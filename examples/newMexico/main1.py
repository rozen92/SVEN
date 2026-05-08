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

# Désactivation de l'analyseur
analyzer.active_eta_opt = False

# Dossier de sortie spécifique pour la baseline Picard
outDir = 'outputs_campagne_picard_005'
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
    for b in blades: 
        b.centerChords = chord_targets.copy()
        b.relax = 0.05 # FIXE pour cette version main.1.py
    return blades, myWT, 0.01, 1e-3

# -----------------------------------------------------------------------------
# Paramètres globaux
# -----------------------------------------------------------------------------
nRotations = 15.          
DegreesPerTimeStep = 10.  
rotationsKeptInWake = 10  
nearWakeLength = 360 * rotationsKeptInWake
innerIter = 15            
density = 1.198           
N_avg = 3                 
steps_per_rotation = int(360. / DegreesPerTimeStep)
Omega = 44.5163679 
R_max = 2.25 

yaws_deg = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
tsrs = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12])

global_start_time = time.time()

print(f"Lancement Baseline: Picard 0.05 constant")
print(f"Campagne: {len(yaws_deg)} Yaws x {len(tsrs)} TSRs")

for yaw_val in yaws_deg:
    yaw_rad = np.radians(yaw_val)
    yaw_start = time.time()
    current_yaw_dataset = []
    
    print(f"\n>>> YAW : {yaw_val}° (Picard 0.05)")

    for tsr_val in tsrs:
        V_mag = (Omega * R_max) / tsr_val
        uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

        Blades, WindTurbine, deltaFlts, tol_picard = NewMexicoWindTurbine(uInfty, density, nearWakeLength)
        
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
        
        for it in range(total_steps):
            refAzimuth += WindTurbine.rotationalVelocity * timeStep
            WindTurbine.updateTurbine(refAzimuth)
            timeSim += timeStep
            
            # Uniquement Picard
            max_err, solver_time, iters_taken = update(
                Blades, uInfty, timeStep, timeSim, innerIter, 
                deltaFlts, global_start_time, [], 
                algo_type="picard", tol=tol_picard, calc_eta=False
            )

            # Moyennage final
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
            
            if (it + 1) % 30 == 0: 
                print(f"  TSR {tsr_val} | Pas {it+1}/{total_steps} | Picard: {iters_taken} iters | Err: {max_err:.2e}")

        # Compilation des résultats du Yaw
        Fn_mean = np.mean(Fn_history, axis=0)
        Ft_mean = np.mean(Ft_history, axis=0)
        Veff_mean = np.mean(Veff_history, axis=0)
        Alpha_mean = np.mean(Alpha_history, axis=0)

        for a_idx in range(steps_per_rotation):
            theta = a_idx * DegreesPerTimeStep
            for ir, r_val in enumerate(centersRadius):
                current_yaw_dataset.append({
                    'TSR': tsr_val,                 
                    'r': r_val,                     
                    'theta': theta,                 
                    'Fn': Fn_mean[a_idx, ir], 
                    'Ft': Ft_mean[a_idx, ir],
                    'V_eff': Veff_mean[a_idx, ir],  
                    'Alpha_deg': np.degrees(Alpha_mean[a_idx, ir]) 
                })

    # Écriture par Yaw
    df_yaw = pd.DataFrame(current_yaw_dataset)
    filename = os.path.join(outDir, f'results_yaw_{yaw_val}deg.csv')
    df_yaw.to_csv(filename, index=False)
    print(f">> Fichier {filename} généré (Baseline).")

print(f"\nBASELINE TERMINEE en {time.time() - global_start_time:.1f}s.")