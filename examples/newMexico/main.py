import os
import sys
import numpy as np
import pandas as pd
import time

# --- Configuration des chemins ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)
sys.path.append(parent_of_project_dir)

from sven.windTurbine import *
from sven.airfoil import *
from sven.blade import *
from sven.solver import update

# Dossier de sortie
outDir = 'outputs_campagne_full_newton_hybrid'
if not os.path.exists(outDir):
    os.makedirs(outDir)

def NewMexicoWindTurbine(windVelocity, density, nearWakeLength):
    sign = -1.
    hubRadius = 0.210  
    nBlades = 3
    rotationalVelocity = 44.5163679  
    bladePitch = sign * 0.040143
    geom_file = os.path.join(script_dir, 'geometry', 'blade.dat')
    data = np.genfromtxt(geom_file, skip_header=1, dtype=str)
    r_targets = data[:, 0].astype(float) 
    twist_targets = -1.0 * data[:, 1].astype(float) 
    chord_targets = np.abs(data[:, 2].astype(float)) 
    airfoil_names = data[:, 3]
    N = len(r_targets)
    nodesRadius = np.zeros(N + 1)
    nodesChord = np.zeros(N + 1)
    nodesTwistAngles = np.zeros(N + 1)
    nodesRadius[0] = hubRadius
    nodesChord[0] = chord_targets[0]
    nodesTwistAngles[0] = twist_targets[0]
    
    for i in range(N):
        nodesRadius[i+1] = 2 * r_targets[i] - nodesRadius[i]
        nodesChord[i+1] = 2 * chord_targets[i] - nodesChord[i]
        nodesTwistAngles[i+1] = 2 * twist_targets[i] - nodesTwistAngles[i]
        
    centersAirfoils = []
    for foilName in airfoil_names:
        foil_path = os.path.join(script_dir, 'geometry', 'Airfoils2', f"{foilName}.foil")
        centersAirfoils.append(Airfoil(foil_path, headerLength=1))
        
    myWT = windTurbine(nBlades, [0., 0., 0.], hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)
    
    for b in blades: 
        b.centerChords = chord_targets.copy()
        
    return blades, myWT, 0.01, 1e-5 

# -----------------------------------------------------------------------------
# Paramètres de la campagne
# -----------------------------------------------------------------------------
nRotations = 15.          
DegreesPerTimeStep = 10.  
rotationsKeptInWake = 10  
nearWakeLength = 360 * rotationsKeptInWake
density = 1.198           
N_avg = 3                 
steps_per_rotation = int(360. / DegreesPerTimeStep)
Omega = 44.5163679 
R_max = 2.25 

# Stratégie Full-Newton Hybrid
total_inner_iter = 20
picard_iters_fixed = 15

# Grilles
tsrs = np.array([4, 6, 8, 10, 12])
yaws_deg = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

global_start_time = time.time()

print(f"Lancement Campagne Full-Newton Hybrid (Groupement par TSR)")
print(f"Stratégie : {picard_iters_fixed} iters Picard puis {total_inner_iter - picard_iters_fixed} iters Newton")
print(f"Légende Logs : Win(N/P)=Vainqueur | it*=Iter Argmin (!=rebond) | J=Evals Jacobienne | Rel=Relaxation\n")

# =============================================================================
# BOUCLE EXTERIEURE : TSR
# =============================================================================
for tsr_val in tsrs:
    tsr_start = time.time()
    current_tsr_dataset = [] 
    
    print(f"############################################################")
    print(f" TRAITEMENT TSR : {tsr_val}")
    print(f"############################################################")

    for yaw_val in yaws_deg:
        yaw_rad = np.radians(yaw_val)
        V_mag = (Omega * R_max) / tsr_val
        uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

        print(f"\n--- Yaw {yaw_val}° (V = {V_mag:.2f} m/s) ---")
        Blades, WindTurbine, deltaFlts, tol_hybrid = NewMexicoWindTurbine(uInfty, density, nearWakeLength)
        
        centersRadius = 0.5 * (WindTurbine.nodesRadius[1:] + WindTurbine.nodesRadius[:-1])
        timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
        total_steps = int((nRotations * 360.) / DegreesPerTimeStep)
        start_avg_it = total_steps - (N_avg * steps_per_rotation)

        Fn_history = np.zeros((N_avg, steps_per_rotation, len(centersRadius)))
        Ft_history = np.zeros((N_avg, steps_per_rotation, len(centersRadius)))
        Veff_history = np.zeros((N_avg, steps_per_rotation, len(centersRadius)))
        Alpha_history = np.zeros((N_avg, steps_per_rotation, len(centersRadius)))

        refAzimuth = -WindTurbine.rotationalVelocity * timeStep
        timeSim = 0.
        
        current_relax = 0.35 

        for it in range(total_steps):
            refAzimuth += WindTurbine.rotationalVelocity * timeStep
            WindTurbine.updateTurbine(refAzimuth)
            timeSim += timeStep
            
            # Appel du solveur
            m_err, s_time, p_its, n_its, b_algo, e_opt, early_min, b_iter, j_evals = update(
                Blades, uInfty, timeStep, timeSim, total_inner_iter, 
                deltaFlts, global_start_time, [], 
                algo_type="hybrid", tol=tol_hybrid, 
                picard_iters=picard_iters_fixed, 
                current_relax=current_relax
            )
            
            # --- LOGS COMPACTS ---
            status = f"{p_its}P+{n_its}N" if n_its > 0 else f"{p_its}P"
            win_char = b_algo[0].upper() # 'N' ou 'P'
            rebond = "!" if early_min else " "
            
            # Affichage console tous les 30 pas
            if (it + 1) % 30 == 0: 
                print(f" Pas {it+1:3}/{total_steps} | {status:<7} | Win:{win_char} | it*:{b_iter:>2}{rebond} | J:{j_evals} | Rel:{current_relax:.3f} | Err:{m_err:.1e}")

            # Warning spécifique si Newton a échoué
            if b_algo == "Picard" and n_its > 0 and (it + 1) % 30 != 0:
                print(f" [Alerte] Pas {it+1:3} | Newton a divergé. Picard restaure it*:{b_iter} | Err:{m_err:.1e}")
                
            # Mise à jour du taux de relaxation pour le prochain pas de temps
            if e_opt > 0:
                current_relax = min(0.35, 0.9 * e_opt)

            # --- STOCKAGE MOYENNAGE ---
            if it >= start_avg_it:
                idx_rot = int((it - start_avg_it) // steps_per_rotation)
                idx_azi = int((it - start_avg_it) % steps_per_rotation)
                
                Fn, Ft = WindTurbine.evaluateForces(density)
                Veff = WindTurbine.blades[0].effectiveVelocity
                Alpha = WindTurbine.blades[0].attackAngle
                
                if idx_rot < N_avg:
                    Fn_history[idx_rot, idx_azi, :] = Fn
                    Ft_history[idx_rot, idx_azi, :] = Ft
                    Veff_history[idx_rot, idx_azi, :] = Veff
                    Alpha_history[idx_rot, idx_azi, :] = Alpha

        # --- COMPILATION DU YAW ---
        Fn_mean = np.mean(Fn_history, axis=0)
        Ft_mean = np.mean(Ft_history, axis=0)
        Veff_mean = np.mean(Veff_history, axis=0)
        Alpha_mean = np.mean(Alpha_history, axis=0)

        for a_idx in range(steps_per_rotation):
            theta = a_idx * DegreesPerTimeStep
            for ir, r_val in enumerate(centersRadius):
                current_tsr_dataset.append({
                    'Yaw': yaw_val,
                    'r': r_val,
                    'theta': theta,
                    'Fn': Fn_mean[a_idx, ir], 
                    'Ft': Ft_mean[a_idx, ir],
                    'V_eff': Veff_mean[a_idx, ir],
                    'Alpha_deg': np.degrees(Alpha_mean[a_idx, ir])
                })

    # --- ÉCRITURE DU FICHIER TSR ---
    df_tsr = pd.DataFrame(current_tsr_dataset)
    filename = os.path.join(outDir, f'results_TSR_{tsr_val}.csv')
    df_tsr.to_csv(filename, index=False)
    print(f">> Fichier {filename} généré en {time.time() - tsr_start:.1f}s.\n")

print(f"\nCAMPAGNE TERMINEE en {time.time() - global_start_time:.1f}s.")