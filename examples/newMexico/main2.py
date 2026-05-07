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

# ACTIVATION de l'analyseur allégé pour récupérer l'historique des eta
analyzer.active_eta_opt = True

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

print(f"Lancement de la campagne Newton (Géométrie exacte blade.dat, analyseur épuré)")

for yaw_val in yaws_deg:
    yaw_rad = np.radians(yaw_val)
    for tsr_val in tsrs:
        case_start = time.time()
        
        # Réinitialisation de l'historique des eta pour ce nouveau cas
        analyzer.reset()

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

        # Tableaux pour stocker les variables locales
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
            
            # Appel du solveur de Newton
            max_err, solver_time, iters_taken = update(
                Blades, uInfty, timeStep, timeSim, innerIter, 
                deltaFlts, global_start_time, [], 
                algo_type="newton", tol=tol_newton
            )
            
            # Avertissement si Newton n'atteint pas la tolérance
            if max_err > tol_newton:
                print(f"  [Avertissement] Pas de temps {it+1}: Newton n'a pas atteint la tolérance (Erreur = {max_err:.2e})")

            # Récupération des données du pas de temps courant
            Fn, Ft = WindTurbine.evaluateForces(density)
            Veff = WindTurbine.blades[0].effectiveVelocity
            Alpha = WindTurbine.blades[0].attackAngle
            
            if it >= start_avg_it:
                t_idx = int((it - start_avg_it) // steps_per_rotation)
                a_idx = int((it - start_avg_it) % steps_per_rotation)
                if t_idx < N_avg:
                    Fn_history[t_idx, a_idx, :] = Fn
                    Ft_history[t_idx, a_idx, :] = Ft
                    Veff_history[t_idx, a_idx, :] = Veff
                    Alpha_history[t_idx, a_idx, :] = Alpha
            
            if (it + 1) % 50 == 0: 
                print(f" Pas {it+1}/{total_steps} | Newton: {iters_taken} iters | Err: {max_err:.2e}")

        # ---------------------------------------------------------
        # Sauvegarde des résultats SPÉCIFIQUES À CE CAS
        # ---------------------------------------------------------
        
        # 1. Historique des eta_opt
        n_eta = min(len(analyzer.eta_relax_init_history), len(analyzer.eta_relax_sol_history), total_steps)
        df_eta = pd.DataFrame({
            'Time_Step': range(1, n_eta + 1),
            'Eta_Opt_Init': analyzer.eta_relax_init_history[:n_eta],
            'Eta_Opt_Sol': analyzer.eta_relax_sol_history[:n_eta]
        })
        df_eta.to_csv(os.path.join(outDir, f'eta_history_yaw{yaw_val}_tsr{tsr_val}.csv'), index=False)

        # 2. Moyennage et stockage des forces/vitesses/alphas (un fichier par cas)
        Fn_mean = np.mean(Fn_history, axis=0)
        Ft_mean = np.mean(Ft_history, axis=0)
        Veff_mean = np.mean(Veff_history, axis=0)
        Alpha_mean = np.mean(Alpha_history, axis=0)

        case_results = []
        for a_idx in range(steps_per_rotation):
            theta = a_idx * DegreesPerTimeStep
            for ir, r_val in enumerate(centersRadius):
                # On stocke pour le fichier spécifique
                case_results.append({
                    'Radius': r_val, 
                    'Azimuth': theta, 
                    'Fn': Fn_mean[a_idx, ir], 
                    'Ft': Ft_mean[a_idx, ir],
                    'V_eff': Veff_mean[a_idx, ir],
                    'Alpha_deg': np.degrees(Alpha_mean[a_idx, ir])
                })
                global_dataset.append({
                    'r': r_val, 'theta': theta, 'yaw': yaw_val, 'TSR': tsr_val,
                    'Fn': Fn_mean[a_idx, ir], 'Ft': Ft_mean[a_idx, ir]
                })
        
        df_case = pd.DataFrame(case_results)
        df_case.to_csv(os.path.join(outDir, f'results_yaw{yaw_val}_tsr{tsr_val}.csv'), index=False)
        
        print(f" Cas terminé en {time.time() - case_start:.1f}s. Fichiers 'results_...' et 'eta_history_...' créés.")

# Sauvegarde finale globale
df_dataset = pd.DataFrame(global_dataset)
df_dataset.to_excel(os.path.join(outDir, 'dataset_mexico_newton_global.xlsx'), index=False)
print(f"\nCampagne terminée. Fichiers sauvegardés dans {outDir}.")