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
outDir = 'outputs_adaptive_hybrid'
if not os.path.exists(outDir):
    os.makedirs(outDir)

def NewMexicoWindTurbine(windVelocity, density, nearWakeLength):
    """Initialise la géométrie de la turbine New Mexico."""
    sign = -1.0
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
    
    nodesRadius = np.zeros(N + 1); nodesChord = np.zeros(N + 1); nodesTwistAngles = np.zeros(N + 1)
    nodesRadius[0] = hubRadius; nodesChord[0] = chord_targets[0]; nodesTwistAngles[0] = twist_targets[0]
    
    for i in range(N):
        nodesRadius[i+1] = 2 * r_targets[i] - nodesRadius[i]
        nodesChord[i+1] = 2 * chord_targets[i] - nodesChord[i]
        nodesTwistAngles[i+1] = 2 * twist_targets[i] - nodesTwistAngles[i]
        
    centersAirfoils = [Airfoil(os.path.join(script_dir, 'geometry', 'Airfoils2', f"{n}.foil"), 1) for n in airfoil_names]
    myWT = windTurbine(nBlades, [0., 0., 0.], hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)
    
    for b in blades: 
        b.centerChords = chord_targets.copy()
        
    return blades, myWT, 0.01, 1e-6 

# -----------------------------------------------------------------------------
# Paramètres de la campagne
# -----------------------------------------------------------------------------
nRotations = 15.0
DegreesPerTimeStep = 10.0
density = 1.198
N_avg = 3
steps_per_rotation = int(360.0 / DegreesPerTimeStep)
Omega = 44.5163679
R_max = 2.25

# Paramètres de la stratégie adaptative
max_budget = 35       # Budget total d'itérations autorisé
p_block = 5           # Taille des blocs Picard
n_block = 3           # Taille des blocs Newton

# Grilles (Regroupement par TSR)
tsrs = np.array([4, 6, 8, 10, 12])
yaws_deg = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

global_start_time = time.time()

print(f"Lancement Campagne Adaptive Hybrid")
print(f"Stratégie de cycles : {p_block}P + {n_block}N (Budget max: {max_budget})")
print(f"Légende Logs : Win(N/P)=Vainqueur | it*=Iter Argmin (!=rebond) | J=Succès/Evals | Rel=Relaxation\n")

# =============================================================================
# BOUCLE EXTERIEURE : TSR
# =============================================================================
for tsr_val in tsrs:
    tsr_start = time.time()
    current_tsr_dataset = [] 
    
    print(f"#################### TSR : {tsr_val} ####################")

    for yaw_val in yaws_deg:
        yaw_rad = np.radians(yaw_val)
        V_mag = (Omega * R_max) / tsr_val
        uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

        print(f"\n--- Yaw {yaw_val}° ---")
        Blades, WT, deltaFlts, tol_hybrid = NewMexicoWindTurbine(uInfty, density, 3600)
        
        cR = 0.5 * (WT.nodesRadius[1:] + WT.nodesRadius[:-1])
        tStep = np.radians(DegreesPerTimeStep) / WT.rotationalVelocity
        total_steps = int((nRotations * 360.) / DegreesPerTimeStep)
        start_avg_it = total_steps - (N_avg * steps_per_rotation)

        Fn_history = np.zeros((N_avg, steps_per_rotation, len(cR)))
        Ft_history = np.zeros_like(Fn_history)
        Veff_history = np.zeros_like(Fn_history)
        Alpha_history = np.zeros_like(Fn_history)

        current_relax = 0.35 

        for it in range(total_steps):
            WT.updateTurbine(WT.rotationalVelocity * tStep * (it+1))
            
            # Appel du solveur adaptatif
            m_err, st, its, win, e_opt, early, b_it, j_ev, j_ok, reason = update(
                Blades, uInfty, tStep, 0, max_budget, 
                deltaFlts, global_start_time, [], 
                algo_type="hybrid", tol=tol_hybrid, 
                p_block=p_block, n_block=n_block, 
                current_relax=current_relax
            )
            
            # Mise à jour dynamique de la relaxation pour le prochain pas
            if e_opt > 0:
                current_relax = min(0.35, 0.9 * e_opt)

            # --- 1. ALERTES DE SECURITE (À chaque pas) ---
            if m_err > tol_hybrid:
                if win == "Picard" and its > p_block:
                    # Newton a été tenté mais a échoué par rapport à Picard
                    print(f" [DIV]  Pas {it+1:3}/{total_steps} | Newton a divergé. Picard restaure it*:{b_it} | Err:{m_err:.1e}")
                else:
                    tag = "[STAG]" if reason == "Stagnation" else "[MAXI]"
                    print(f" {tag} Pas {it+1:3}/{total_steps} | Précision non atteinte | Err:{m_err:.1e}")

            # --- 2. LOGS D'ANALYSE (Tous les 30 pas) ---
            if (it + 1) % 30 == 0:
                win_char = win[0].upper()
                reb = "!" if early else " "
                print(f"        Pas {it+1:3}/{total_steps} | Its:{its:2} | Win:{win_char} | it*:{b_it:>2}{reb} | J:{j_ok}/{j_ev} | Rel:{current_relax:.3f} | Err:{m_err:.1e}")

            # --- STOCKAGE MOYENNAGE ---
            if it >= start_avg_it:
                idx_rot = (it - start_avg_it) // steps_per_rotation
                idx_azi = (it - start_avg_it) % steps_per_rotation
                Fn, Ft = WT.evaluateForces(density)
                if idx_rot < N_avg:
                    Fn_history[idx_rot, idx_azi, :] = Fn
                    Ft_history[idx_rot, idx_azi, :] = Ft
                    Veff_history[idx_rot, idx_azi, :] = WT.blades[0].effectiveVelocity
                    Alpha_history[idx_rot, idx_azi, :] = WT.blades[0].attackAngle

        # Compilation des données du TSR
        Fn_mean = np.mean(Fn_history, axis=0)
        Ft_mean = np.mean(Ft_history, axis=0)
        Veff_mean = np.mean(Veff_history, axis=0)
        Alpha_mean = np.mean(Alpha_history, axis=0)

        for a_idx in range(steps_per_rotation):
            theta = a_idx * DegreesPerTimeStep
            for ir, r_val in enumerate(cR):
                current_tsr_dataset.append({
                    'Yaw': yaw_val, 'r': r_val, 'theta': theta,
                    'Fn': Fn_mean[a_idx, ir], 'Ft': Ft_mean[a_idx, ir],
                    'V_eff': Veff_mean[a_idx, ir], 'Alpha_deg': np.degrees(Alpha_mean[a_idx, ir])
                })

    # Sauvegarde du fichier par TSR
    df_tsr = pd.DataFrame(current_tsr_dataset)
    filename = os.path.join(outDir, f'results_TSR_{tsr_val}.csv')
    df_tsr.to_csv(filename, index=False)
    print(f"\n>> Fichier TSR {tsr_val} généré en {time.time() - tsr_start:.1f}s.\n")

print(f"CAMPAGNE TERMINEE en {time.time() - global_start_time:.1f}s.")