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
from sven.solver import update, analyzer

# DÉSACTIVATION du mode automatique global
analyzer.active_eta_opt = False

# Dossier de sortie
outDir = 'outputs_newton_exact'
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
    nodesRadius = np.zeros(N + 1); nodesChord = np.zeros(N + 1); nodesTwistAngles = np.zeros(N + 1)
    nodesRadius[0] = hubRadius; nodesChord[0] = chord_targets[0]; nodesTwistAngles[0] = twist_targets[0]
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
    for b in blades: b.centerChords = chord_targets.copy()
    return blades, myWT, 0.01, 1e-5

# Paramètres globaux
nRotations = 10.          
DegreesPerTimeStep = 10.  
rotationsKeptInWake = 10  
nearWakeLength = 360 * rotationsKeptInWake
innerIter = 15            
density = 1.198           
N_avg = 3                 
steps_per_rotation = int(360. / DegreesPerTimeStep)
Omega = 44.5163679 
R_max = 2.25 

# Stratégie 3 phases
quarter_rotation_steps = int(90. / DegreesPerTimeStep)
warmup_steps = 5 * steps_per_rotation

yaws_deg = np.array([0.0, 15.0, 30.0])
tsrs = np.array([4, 8, 12])

global_dataset = []
global_start_time = time.time()

for yaw_val in yaws_deg:
    yaw_rad = np.radians(yaw_val)
    for tsr_val in tsrs:
        case_start = time.time()
        V_mag = (Omega * R_max) / tsr_val
        uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

        print(f"\n--- Cas: Yaw {yaw_val}°, TSR {tsr_val} ---")
        Blades, WindTurbine, deltaFlts, tol_newton = NewMexicoWindTurbine(uInfty, density, nearWakeLength)
        
        centersRadius = 0.5 * (WindTurbine.nodesRadius[1:] + WindTurbine.nodesRadius[:-1])
        timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
        total_steps = int((nRotations * 360.) / DegreesPerTimeStep)
        start_avg_it = total_steps - (N_avg * steps_per_rotation)

        Fn_history = np.zeros((N_avg, steps_per_rotation, len(centersRadius)))
        Ft_history = np.zeros((N_avg, steps_per_rotation, len(centersRadius)))

        refAzimuth = -WindTurbine.rotationalVelocity * timeStep
        timeSim = 0.
        
        # Liste locale pour le sondage eta_opt
        eta_samples = []
        calculated_relax = 0.05 # Valeur par défaut de secours

        for it in range(total_steps):
            refAzimuth += WindTurbine.rotationalVelocity * timeStep
            WindTurbine.updateTurbine(refAzimuth)
            timeSim += timeStep
            
            # --- LOGIQUE DE CHOIX DU SOLVEUR ---
            if it < quarter_rotation_steps:
                # PHASE 1 : Sondage Newton
                current_algo = "newton"
                current_tol = tol_newton
                
                max_err, solver_time, iters_taken = update(
                    Blades, uInfty, timeStep, timeSim, innerIter, 
                    deltaFlts, global_start_time, [], 
                    algo_type=current_algo, tol=current_tol
                )
                
                # Calcul de l'eta optimal instantané pour échantillonnage
                J = analyzer.compute_jacobian(Blades, deltaFlts)
                e_opt = analyzer.compute_optimal_eta(J)
                if e_opt > 0:
                    eta_samples.append(e_opt)

            elif it < warmup_steps:
                # PHASE 2 : Picard avec relaxation calibrée
                if it == quarter_rotation_steps:
                    # --- AFFICHAGE DES LOGS DE CALIBRATION À 90° ---
                    print(f"  [Calibration] Fin du 1er quart de tour (Sondage Newton).")
                    if eta_samples:
                        calculated_relax = 0.9 * min(eta_samples)
                        print(f"  [Calibration] {len(eta_samples)} pas de temps stables (eta_opt > 0) recensés.")
                        print(f"  [Calibration] Relax appliqué pour Picard : {calculated_relax:.4f} (0.9 * min_eta)")
                    else:
                        print(f"  [Calibration] Attention: Aucun eta_opt > 0 trouvé ! Utilisation du défaut : {calculated_relax}")
                    print(f"  [Calibration] Basculement sur Picard jusqu'au 5ème tour...")
                    # -----------------------------------------------
                    
                    for b in Blades:
                        b.relax = calculated_relax # Application à la pale

                current_algo = "picard"
                current_tol = 1e-3
                max_err, solver_time, iters_taken = update(
                    Blades, uInfty, timeStep, timeSim, innerIter, 
                    deltaFlts, global_start_time, [], 
                    algo_type=current_algo, tol=current_tol
                )

            else:
                # PHASE 3 : Newton final
                current_algo = "newton"
                current_tol = tol_newton
                max_err, solver_time, iters_taken = update(
                    Blades, uInfty, timeStep, timeSim, innerIter, 
                    deltaFlts, global_start_time, [], 
                    algo_type=current_algo, tol=current_tol
                )
                if max_err > current_tol:
                    print(f"  [Warning] Pas {it+1}: Newton non convergé (Err: {max_err:.2e})")

            # Stockage pour moyennage
            if it >= start_avg_it:
                idx_rot = int((it - start_avg_it) // steps_per_rotation)
                idx_azi = int((it - start_avg_it) % steps_per_rotation)
                Fn, Ft = WindTurbine.evaluateForces(density)
                if idx_rot < N_avg:
                    Fn_history[idx_rot, idx_azi, :] = Fn
                    Ft_history[idx_rot, idx_azi, :] = Ft
            
            # Logs tous les 30 pas
            if (it + 1) % 30 == 0: 
                print(f" Pas {it+1}/{total_steps} | {current_algo.capitalize()}: {iters_taken} iters | Err: {max_err:.2e}")

        # Sauvegarde cas par cas
        Fn_mean = np.mean(Fn_history, axis=0)
        Ft_mean = np.mean(Ft_history, axis=0)
        case_res = []
        for a_idx in range(steps_per_rotation):
            theta = a_idx * DegreesPerTimeStep
            for ir, r_val in enumerate(centersRadius):
                case_res.append({'Radius': r_val, 'Azimuth': theta, 'Fn': Fn_mean[a_idx, ir], 'Ft': Ft_mean[a_idx, ir]})
        pd.DataFrame(case_res).to_csv(os.path.join(outDir, f'results_yaw{yaw_val}_tsr{tsr_val}.csv'), index=False)

print(f"\nCampagne terminée. Fichiers sauvegardés dans {outDir}.")