import os
import sys
import numpy as np
import pandas as pd
import time
from scipy.stats.qmc import LatinHypercube as lhc

# --- Configuration des chemins ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)
sys.path.append(parent_of_project_dir)

from sven.windTurbine import *
from sven.airfoil import *
from sven.blade import *
from sven.solver import update

outDir = 'outputs_adaptive_hybrid'
if not os.path.exists(outDir):
    os.makedirs(outDir)

def NewMexicoWindTurbine(windVelocity, density, nearWakeLength):
    sign = -1.0; hubRadius = 0.210; nBlades = 3; rotationalVelocity = 44.5163679; bladePitch = sign * 0.040143
    geom_file = os.path.join(script_dir, 'geometry', 'blade.dat')
    data = np.genfromtxt(geom_file, skip_header=1, dtype=str)
    r_targets = data[:, 0].astype(float); twist_targets = -1.0 * data[:, 1].astype(float) 
    chord_targets = np.abs(data[:, 2].astype(float)); airfoil_names = data[:, 3]
    N = len(r_targets)
    nodesRadius = np.zeros(N + 1); nodesChord = np.zeros(N + 1); nodesTwistAngles = np.zeros(N + 1)
    nodesRadius[0] = hubRadius; nodesChord[0] = chord_targets[0]; nodesTwistAngles[0] = twist_targets[0]
    
    for i in range(N):
        nodesRadius[i+1] = 2 * r_targets[i] - nodesRadius[i]
        nodesChord[i+1] = 2 * chord_targets[i] - nodesChord[i]
        nodesTwistAngles[i+1] = 2 * twist_targets[i] - nodesTwistAngles[i]
        
    centersAirfoils = [Airfoil(os.path.join(script_dir, 'geometry', 'Airfoils2', f"{n}.foil"), 1) for n in airfoil_names]
    myWT = windTurbine(nBlades, [0., 0., 0.], hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)
    for b in blades: b.centerChords = chord_targets.copy()
    return blades, myWT, 0.01, 1e-6 

# -----------------------------------------------------------------------------
# Paramètres de la campagne
# -----------------------------------------------------------------------------
base_rotations = 10        # Tours minimum par défaut
max_extra_rotations = 5   # Tours additionnels autorisés
max_rotations = base_rotations + max_extra_rotations  # Soit 20 tours max
DegreesPerTimeStep = 5.0
density = 1.198
N_avg = 3
steps_per_rotation = int(360.0 / DegreesPerTimeStep)
Omega = 44.5163679
R_max = 2.25

# Paramètres de la stratégie "Test & Rollback"
max_picard_iters = 5000  
p_block = 5           
n_block = 5           

## Préparation de l'échantillonnage LHC
sampler = lhc(2, strength = 1, seed = 42)
samples = sampler.random(n = 100)

samples[:,0] = samples[:,0]*8 + 4
samples[:,1] = samples[:,1]*60 - 30

global_start_time = time.time()

file_log = os.path.join(outDir, 'log_sim.txt')
log = open(file_log, 'a', encoding = 'utf-8')

print(f"\n\n\nLancement Campagne Hybrid 'Test & Rollback'")
print(f"Stratégie : {p_block}P + {n_block}N (Budget Picard: {max_picard_iters})")
print(f"Légende   : Win=Vainqueur | J=Succès/Evals | Rel=Relax\n")

log.write("#"*120+"\n")
log.write("#"*120+"\n")
log.write("#"*120+"\n")

log.write(f"\nLancement Campagne Hybrid 'Test & Rollback'\n")
log.write(f"\nStratégie : {p_block}P + {n_block}N (Budget Picard: {max_picard_iters})\n")
log.write(f"\nLégende   : Win=Vainqueur | J=Succès/Evals | Rel=Relax")
log.write("\n\n")

yaws_deg = np.array([5.,10.,15.,20.,25.,30.]) # Angles de lacet
tsrs = np.array([4,6,8,10,12])       # Tip Speed Ratios

current_tsr_dataset = []
for i in range(len(tsrs)) :
    for j in range(len(yaws_deg)) :
        tsr_val = tsrs[i]
        yaw_val = yaws_deg[j]    
        tsr_start = time.time();  
        print(f"#################### ÈME COUPLE (TSR,YAW) : ({tsr_val.round(3)},{yaw_val.round(3)}) ####################")
        log.write(f"\n#################### ÈME COUPLE (TSR,YAW) : ({tsr_val.round(3)},{yaw_val.round(3)}) ####################\n")

        #for yaw_val in yaws_deg:
        uInfty = np.array([((Omega*R_max)/tsr_val)*np.cos(np.radians(yaw_val)), ((Omega*R_max)/tsr_val)*np.sin(np.radians(yaw_val)), 0.0], dtype=np.float32)
            
        Blades, WT, deltaFlts, tol_hybrid = NewMexicoWindTurbine(uInfty, density, 3600)
            
        cR = 0.5 * (WT.nodesRadius[1:] + WT.nodesRadius[:-1])
        tStep = np.radians(DegreesPerTimeStep) / WT.rotationalVelocity
            
        # Le nombre de pas total théorique maximum
        total_max_steps = int((max_rotations * 360.) / DegreesPerTimeStep)

        Fn_history = np.zeros((N_avg, steps_per_rotation, len(cR)))
        Ft_history = np.zeros_like(Fn_history)
        Veff_history = np.zeros_like(Fn_history)
        Alpha_history = np.zeros_like(Fn_history)
        Gamma_history = np.zeros_like(Fn_history) # Nouveau : stockage de Gamma

        current_relax = 0.35 

        for it in range(total_max_steps):
                WT.updateTurbine(WT.rotationalVelocity * tStep * (it+1))
                
                m_err, st, p_its, n_its, win, e_opt, j_ev, j_ok = update(
                    Blades, uInfty, tStep, 0, max_picard_iters, 
                    deltaFlts, global_start_time, [], 
                    algo_type="hybrid", tol=tol_hybrid, 
                    p_block=p_block, n_block=n_block, current_relax=current_relax
                )
                
                if e_opt > 0:
                    current_relax = min(0.35, 0.9 * e_opt)
                else:
                    print(f" [INSTABLE] Pas {it+1:3}/{total_max_steps} | Valeurs propres mixtes (eta_opt = 0) | Err:{m_err:.1e}")
                    log.write(f"\n  [INSTABLE] Pas {it+1:3}/{total_max_steps} | Valeurs propres mixtes (eta_opt = 0) | Err:{m_err:.1e}\n")
                if m_err > tol_hybrid:
                    print(f" [MAXI] Pas {it+1:3}/{total_max_steps} | Précision non atteinte ({p_its}P tentés) | Err:{m_err:.1e}")
                    log.write(f"\n [MAXI] Pas {it+1:3}/{total_max_steps} | Précision non atteinte ({p_its}P tentés) | Err:{m_err:.1e}\n")
                else:
                    if (it + 1) % 60 == 0:
                        status = f"{p_its}P+{n_its}N"
                        win_char = win[0].upper()
                        print(f"        Pas {it+1:3}/{total_max_steps} | {status:<7} | Win:{win_char} | J:{j_ok}/{j_ev} | Rel:{current_relax:.3f} | Err:{m_err:.1e}")
                        log.write(f"\n        Pas {it+1:3}/{total_max_steps} | {status:<7} | Win:{win_char} | J:{j_ok}/{j_ev} | Rel:{current_relax:.3f} | Err:{m_err:.1e}\n")

                # --- STOCKAGE MOYENNAGE (Buffer tournant) ---
                idx_rot = it // steps_per_rotation
                idx_azi = it % steps_per_rotation
                hist_idx = idx_rot % N_avg  # Écrase l'ancien tour (0, 1, 2, 0, 1, 2...)
                
                Fn, Ft = WT.evaluateForces(density)
                Fn_history[hist_idx, idx_azi, :] = Fn
                Ft_history[hist_idx, idx_azi, :] = Ft
                Veff_history[hist_idx, idx_azi, :] = WT.blades[0].effectiveVelocity
                Alpha_history[hist_idx, idx_azi, :] = WT.blades[0].attackAngle
                Gamma_history[hist_idx, idx_azi, :] = WT.blades[0].gammaBound

                # --- ARRÊT ADAPTATIF (À la fin de chaque rotation complète) ---
                if idx_azi == steps_per_rotation - 1:
                    completed_rotations = idx_rot + 1
                    
                    # On ne commence à checker qu'à partir du 10ème tour
                    if completed_rotations >= base_rotations:
                        Gamma_flat = Gamma_history.flatten()
                        Fn_ptp = np.max(np.ptp(Fn_history, axis=0)) 
                        Ft_ptp = np.max(np.ptp(Ft_history, axis=0))
                        Fn_mean_abs = np.mean(np.abs(Fn_history))
                        Ft_mean_abs = np.mean(np.abs(Ft_history))
                        Fn_rel = (Fn_ptp / Fn_mean_abs * 100) if Fn_mean_abs > 0 else 0.0
                        Ft_rel = (Ft_ptp / Ft_mean_abs * 100) if Ft_mean_abs > 0 else 0.0

                        bilan_str = f"        -> [BILAN TOUR {completed_rotations:02d}] Gamma: Min={np.min(Gamma_flat):.2f} Moy={np.mean(Gamma_flat):.2f} Max={np.max(Gamma_flat):.2f} Std={np.std(Gamma_flat):.2f} | Périodicité: Fn={Fn_ptp:.2e} ({Fn_rel:.2f}%) Ft={Ft_ptp:.2e} ({Ft_rel:.2f}%)"
                        print(bilan_str)
                        log.write("\n" + bilan_str + "\n\n") 

                        # Condition de convergence stricte (< 0.1%)
                        if Fn_rel <= 2 and Ft_rel <= 7:
                            success_str = f"        => Convergence périodique atteinte en {completed_rotations} tours ! Fin de la simulation."
                            print(success_str)
                            log.write("\n" + success_str + "\n\n")
                            break
                        
                        # Condition d'échec / limite de budget
                        elif completed_rotations == max_rotations:
                            max_str = f"        => Limite stricte de {max_rotations} tours atteinte. Fin de la simulation."
                            print(max_str)
                            log.write("\n" + max_str + "\n\n")
                            break

        # Compilation TSR
        Fn_mean = np.mean(Fn_history, axis=0)        
        Ft_mean = np.mean(Ft_history, axis=0)
        Veff_mean = np.mean(Veff_history, axis=0)
        Alpha_mean = np.mean(Alpha_history, axis=0)

        for a_idx in range(steps_per_rotation):
                theta = a_idx * DegreesPerTimeStep
                for ir, r_val in enumerate(cR):
                    current_tsr_dataset.append({
                        'Yaw': yaw_val, 'TSR' : tsr_val, 'r': r_val, 'theta': theta,
                        'Fn': Fn_mean[a_idx, ir], 'Ft': Ft_mean[a_idx, ir],
                        'V_eff': Veff_mean[a_idx, ir], 'Alpha_deg': np.degrees(Alpha_mean[a_idx, ir])
                    })

        print(f"\n>> Fichier TSR {tsr_val} généré en {time.time() - tsr_start:.1f}s.\n")
        log.write(f"\n>> Efforts pour le couple ({tsr_val},{yaw_val}) calculé  en {time.time() - tsr_start:.1f}s.\n")
        log.write("\n\n")
        log.write("#"*120+"\n")
        log.write("#"*120+"\n")
        log.write("#"*120+"\n")
        log.write("\n\n")

        if (i+1)%35 == 0 :
            df_tsr = pd.DataFrame(current_tsr_dataset)
            df_tsr.to_csv(os.path.join(outDir, f'results_sanitycheck.csv'), index=False)
            current_tsr_dataset = []

print(f"CAMPAGNE TERMINEE en {time.time() - global_start_time:.1f}s.")
log.write(f"\n\nCAMPAGNE TERMINEE en {time.time() - global_start_time:.1f}s.\n\n")

log.write("#"*120+"\n")
log.write("#"*120+"\n")
log.write("#"*120+"\n")

log.close()