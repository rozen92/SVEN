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
outDir = 'outputs_campagne_complete'
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

# -----------------------------------------------------------------------------
# Paramètres globaux de la campagne
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

# Stratégie 3 phases (15 tours total)
calibration_steps = 1 * steps_per_rotation    # Phase 1: Tour 1 (Calibration Newton)
picard_warmup_steps = 6 * steps_per_rotation  # Phase 2: Tours 2 à 6 (5 tours de Picard)
# Phase 3: Tours 7 à 15 (Newton)

# Grilles de la campagne (Asymétrique pour le Yaw !)
yaws_deg = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
tsrs = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12])

global_start_time = time.time()

print(f"Lancement Campagne: {len(yaws_deg)} Yaws x {len(tsrs)} TSRs ({len(yaws_deg)*len(tsrs)} cas)")
print("Séquence : 1 tour Newton (Calib) -> 5 tours Picard (Warmup) -> Newton (Tour 7 à 15)")

# Boucle Principale sur le YAW 
for yaw_val in yaws_deg:
    yaw_rad = np.radians(yaw_val)
    yaw_start = time.time()
    
    current_yaw_dataset = []
    
    print(f"\n============================================================")
    print(f" TRAITEMENT YAW : {yaw_val}°")
    print(f"============================================================")

    for tsr_val in tsrs:
        V_mag = (Omega * R_max) / tsr_val
        uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

        print(f"\n--- TSR {tsr_val} (V = {V_mag:.2f} m/s) ---")
        Blades, WindTurbine, deltaFlts, tol_newton = NewMexicoWindTurbine(uInfty, density, nearWakeLength)
        
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
        
        # Variables de calibration
        eta_samples = []
        calculated_relax = 0.3 # Valeur de repli

        for it in range(total_steps):
            refAzimuth += WindTurbine.rotationalVelocity * timeStep
            WindTurbine.updateTurbine(refAzimuth)
            timeSim += timeStep
            
            # --- LOGIQUE DE CHOIX DU SOLVEUR ---
            if it < calibration_steps:
                # PHASE 1 : Tour 1 (Sondage Newton)
                current_algo = "newton"
                current_tol = tol_newton
                do_calc_eta = True # Activation de l'interrupteur
                
            elif it < picard_warmup_steps:
                # PHASE 2 : Tours 2 à 6 (Picard 5 tours)
                if it == calibration_steps:
                    # --- FIN DU TOUR 1 : BILAN DE CALIBRATION ---
                    if eta_samples:
                        calculated_relax = 0.9 * min(eta_samples)
                        print(f"  [Calib] {len(eta_samples)}/{calibration_steps} pas de temps exploitables.")
                        print(f"  [Calib] Relax appliqué : {calculated_relax:.4f} (Basé sur 0.9 * min_eta)")
                    else:
                        print(f"  [Calib] Attention: 0/{calibration_steps} pas exploitables ! Défaut : {calculated_relax}")
                    print(f"  [Transition] Basculement Picard pour 5 tours.")
                    
                    for b in Blades:
                        b.relax = calculated_relax

                current_algo = "picard"
                current_tol = 1e-3
                do_calc_eta = False # Désactivé pour économiser du CPU
                
            else:
                # PHASE 3 : Tours 7 à 15 (Newton Final)
                if it == picard_warmup_steps:
                    print(f"  [Transition] Début Tour 7: Retour à Newton.")
                    
                current_algo = "newton"
                current_tol = tol_newton
                do_calc_eta = False 

            # Appel du solveur 
            max_err, solver_time, iters_taken = update(
                Blades, uInfty, timeStep, timeSim, innerIter, 
                deltaFlts, global_start_time, [], 
                algo_type=current_algo, tol=current_tol, calc_eta=do_calc_eta
            )
            
            # Récupération de l'eta calculé (uniquement en phase 1)
            if do_calc_eta and hasattr(analyzer, 'last_eta_opt') and analyzer.last_eta_opt > 0:
                eta_samples.append(analyzer.last_eta_opt)

            # Warning de Newton seulement en régime final
            if current_algo == "newton" and max_err > current_tol and it >= picard_warmup_steps:
                print(f"  [Warning] Pas {it+1}: Newton non convergé (Err: {max_err:.2e})")

            # --- STOCKAGE MOYENNAGE (Tours 13, 14, 15) ---
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
            
            # Affichage console tous les 30 pas
            if (it + 1) % 30 == 0: 
                print(f"  Pas {it+1}/{total_steps} | {current_algo.capitalize()}: {iters_taken} iters | Err: {max_err:.2e}")

        # --- SAUVEGARDE DU CAS ---
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

    # --- ÉCRITURE DU FICHIER YAW ---
    df_yaw = pd.DataFrame(current_yaw_dataset)
    filename = os.path.join(outDir, f'results_yaw_{yaw_val}deg.csv')
    df_yaw.to_csv(filename, index=False)
    print(f">> Fichier {filename} généré avec succès en {time.time() - yaw_start:.1f}s.")

print(f"\nCAMPAGNE TERMINEE en {time.time() - global_start_time:.1f}s.")