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

# Désactivation de l'analyseur mathématique pour la campagne de production
analyzer.active = False

# Dossier de sortie
outDir = 'outputs_newton_exact'
if not os.path.exists(outDir):
    os.makedirs(outDir)

# -----------------------------------------------------------------------------
# Fonction de création de la turbine Mexico
# -----------------------------------------------------------------------------
def NewMexicoWindTurbine(windVelocity, density, nearWakeLength):
    sign = -1.
    hubRadius = 0.210  
    nBlades = 3
    rotationalVelocity = 44.5163679  
    bladePitch = sign * 0.040143
    
    # 1. Charger les données géométriques
    geom_file = os.path.join(script_dir, 'geometry', 'blade.dat')
    data = np.genfromtxt(geom_file, skip_header=1, dtype=str)
    
    # Les données de blade.dat sont en mètres
    r_targets = data[:, 0].astype(float) 
    
    twist_targets = -1.0 * data[:, 1].astype(float) 
    
    chord_targets = np.abs(data[:, 2].astype(float)) 
    airfoil_names = data[:, 3]
    
    N = len(r_targets)
    
    # 2. Reconstruction récursive de la grille 
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
        
    if not np.all(np.diff(nodesRadius) > 0):
        print("ATTENTION: La grille radiale reconstruite n'est pas strictement croissante.")

    # 3. Chargement des profils
    centersAirfoils = []
    for foilName in airfoil_names:
        foil_path = os.path.join(script_dir, 'geometry', 'Airfoils2', f"{foilName}.foil")
        centersAirfoils.append(Airfoil(foil_path, headerLength=1))

    # 4. Initialisation de la turbine SVEN
    myWT = windTurbine(nBlades, [0., 0., 0.], hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)

    # Forçage des cordes aux centres
    for b in blades:
        b.centerChords = chord_targets.copy()

    return blades, myWT, 0.01, 1e-5

# -----------------------------------------------------------------------------
# Paramètres globaux de simulation
# -----------------------------------------------------------------------------
nRotations = 10.          
DegreesPerTimeStep = 10.  
rotationsKeptInWake = 10  
nearWakeLength = 360 * rotationsKeptInWake
innerIter = 15            # Newton converge typiquement en 2-3 itérations
density = 1.198           
N_avg = 3                 # Moyenne sur les 3 derniers tours
steps_per_rotation = int(360. / DegreesPerTimeStep)

Omega = 44.5163679 
R_max = 2.25 # Envergure totale (span)

# Grille de paramètres
yaws_deg = np.array([0.0, 15.0, 30.0])
tsrs = np.array([4, 8, 12])

global_dataset = []
global_start_time = time.time()

print(f"Lancement de la campagne Newton (Géométrie exacte blade.dat)")

for yaw_val in yaws_deg:
    yaw_rad = np.radians(yaw_val)
    for tsr_val in tsrs:
        case_start = time.time()
        
        # Calcul du vecteur vent
        V_mag = (Omega * R_max) / tsr_val
        uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

        print(f"\n--- Cas: Yaw {yaw_val}°, TSR {tsr_val} (V={V_mag:.2f}m/s) ---")

        # Initialisation
        Blades, WindTurbine, deltaFlts, tol_newton = NewMexicoWindTurbine(uInfty, density, nearWakeLength)
        
        centersRadius = 0.5 * (WindTurbine.nodesRadius[1:] + WindTurbine.nodesRadius[:-1])
        timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
        total_steps = int((nRotations * 360.) / DegreesPerTimeStep)
        start_avg_it = total_steps - (N_avg * steps_per_rotation)

        Fn_history = np.zeros((N_avg, steps_per_rotation, len(centersRadius)))
        Ft_history = np.zeros((N_avg, steps_per_rotation, len(centersRadius)))

        refAzimuth = -WindTurbine.rotationalVelocity * timeStep
        timeSim = 0.
        
        for it in range(total_steps):
            refAzimuth += WindTurbine.rotationalVelocity * timeStep
            WindTurbine.updateTurbine(refAzimuth)
            timeSim += timeStep
            
            # Appel du solveur de Newton
            max_err, solver_time, iters_taken = update(
                Blades, uInfty, timeStep, timeSim, innerIter, 
                deltaFlts, global_start_time, [], 
                algo_type="newton", tol=tol_newton
            )

            Fn, Ft = WindTurbine.evaluateForces(density)
            
            if it >= start_avg_it:
                t_idx = int((it - start_avg_it) // steps_per_rotation)
                a_idx = int((it - start_avg_it) % steps_per_rotation)
                if t_idx < N_avg:
                    Fn_history[t_idx, a_idx, :] = Fn
                    Ft_history[t_idx, a_idx, :] = Ft
            
            if (it + 1) % 50 == 0: 
                print(f" Pas {it+1}/{total_steps} | Newton: {iters_taken} iters | Err: {max_err:.2e}")

        # Moyennage et stockage
        Fn_mean = np.mean(Fn_history, axis=0)
        Ft_mean = np.mean(Ft_history, axis=0)

        for a_idx in range(steps_per_rotation):
            theta = a_idx * DegreesPerTimeStep
            for ir, r_val in enumerate(centersRadius):
                global_dataset.append({
                    'r': r_val, 'theta': theta, 'yaw': yaw_val, 'TSR': tsr_val,
                    'Fn': Fn_mean[a_idx, ir], 'Ft': Ft_mean[a_idx, ir]
                })
        
        print(f" Cas terminé en {time.time() - case_start:.1f}s")

# Sauvegarde finale
df_dataset = pd.DataFrame(global_dataset)
df_dataset.to_excel(os.path.join(outDir, 'dataset_mexico_newton.xlsx'), index=False)
print(f"\nCampagne terminée. Dataset sauvegardé dans {outDir}.")