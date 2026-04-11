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
from sven.solver import *
from scipy import interpolate

# Dossier de sortie
outDir = 'outputs'
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
    
    dataAirfoils = np.genfromtxt('./geometry/mexico.blade', skip_header=1, usecols=(7), dtype='U')
    intAirfoils = np.arange(0, len(dataAirfoils))
    data = np.genfromtxt('./geometry/mexico.blade', skip_header=1)
    refRadius = data[:,2] 
    inputNodesTwistAngles = -sign * np.radians(data[:, 5])
    inputNodesChord = data[:, 6]

    f = interpolate.interp1d(data[:, 2], intAirfoils, kind='nearest')
    centersAirfoils = []
    for i in range(len(refRadius)):
        foilName = str(dataAirfoils[int(f(refRadius[i]))])
        centersAirfoils.append(Airfoil('./geometry/' + foilName, headerLength=1))

    nodesRadius = hubRadius + refRadius
    nodesTwistAngles = np.interp(refRadius, data[:, 2], inputNodesTwistAngles)
    nodesChord = np.interp(refRadius, data[:, 2], inputNodesChord)

    myWT = windTurbine(nBlades, [0., 0., 0.], hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)

    return blades, myWT, windVelocity, density, 0.01, 1e-4

# -----------------------------------------------------------------------------
# Paramètres globaux de simulation
# -----------------------------------------------------------------------------
nRotations = 10.          #
DegreesPerTimeStep = 10.  # Résolution azimutale
rotationsKeptInWake = 10  # Longueur du sillage
nearWakeLength = 360 * rotationsKeptInWake
innerIter = 12
density = 1.191           # kg/m3
N_avg = 3                 # Moyenne sur les 3 derniers tours
steps_per_rotation = int(360. / DegreesPerTimeStep)

# Extraction du rayon max pour calcul du TSR
data_geom = np.genfromtxt('./geometry/mexico.blade', skip_header=1)
R_max = 0.210 + np.max(data_geom[:,2]) 
Omega = 44.5163679 

# -----------------------------------------------------------------------------
# Grille de paramètres
# -----------------------------------------------------------------------------
yaws_deg = np.array([0.0, 15.0, 30.0]) # Angles de lacet (yaw)
tsrs = np.array([4, 8, 12])       # Tip Speed Ratios (TSR)

# Liste pour stocker toutes les données globales du dataset final
global_dataset = []
global_start_time = time.time()

print(f"Lancement de la campagne : {len(yaws_deg) * len(tsrs)} simulations.")

# -----------------------------------------------------------------------------
# Boucles paramétriques
# -----------------------------------------------------------------------------
for i_yaw, yaw_val in enumerate(yaws_deg):
    yaw_rad = np.radians(yaw_val)
    
    for i_tsr, tsr_val in enumerate(tsrs):
        case_start = time.time()
        
        # Calcul du vecteur vent à partir du TSR et du Yaw
        V_mag = (Omega * R_max) / tsr_val
        uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

        print(f"\n--- Cas: Yaw {yaw_val}°, TSR {tsr_val} (V={V_mag:.2f}m/s) ---")

        # Initialisation
        Blades, WindTurbine, _, _, deltaFlts, _ = NewMexicoWindTurbine(uInfty, density, nearWakeLength)
        
        # Récupération des rayons au centre des éléments
        centersRadius = 0.5 * (WindTurbine.nodesRadius[1:] + WindTurbine.nodesRadius[:-1])
        
        timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
        timeEnd = np.radians(nRotations * 360.) / WindTurbine.rotationalVelocity
        timeSteps = np.arange(0., timeEnd, timeStep)

        num_sections = len(Blades[0].centers)
        total_steps = len(timeSteps)
        start_avg_it = total_steps - (N_avg * steps_per_rotation)

        # Historiques locaux pour ce cas de TSR/Yaw
        conv_data = [] 
        Fn_history = np.zeros((N_avg, steps_per_rotation, num_sections))
        Ft_history = np.zeros((N_avg, steps_per_rotation, num_sections))
        Veff_history = np.zeros((N_avg, steps_per_rotation, num_sections))
        Alpha_history = np.zeros((N_avg, steps_per_rotation, num_sections))

        # Boucle temporelle
        refAzimuth = -WindTurbine.rotationalVelocity * timeStep
        timeSim = 0.
        for it in range(total_steps):
            refAzimuth += WindTurbine.rotationalVelocity * timeStep
            WindTurbine.updateTurbine(refAzimuth)
            timeSim += timeStep
            
            # Mise à jour du solveur (SVEN)
            update(Blades, uInfty, timeStep, timeSim, innerIter, deltaFlts, case_start, [])

            # Calcul des forces Fn/Ft 
            Fn, Ft = WindTurbine.evaluateForces(density)
            
            # Extraction des données dynamiques pour les variables globales (sur la pale 0)
            Veff = WindTurbine.blades[0].effectiveVelocity
            Alpha = WindTurbine.blades[0].attackAngle
            
            # --- MODIFICATION ICI : Calcul de l'erreur pour les 3 pales ---
            err_b0 = WindTurbine.blades[0].f_Gamma - WindTurbine.blades[0].gammaBound
            err_b1 = WindTurbine.blades[1].f_Gamma - WindTurbine.blades[1].gammaBound
            err_b2 = WindTurbine.blades[2].f_Gamma - WindTurbine.blades[2].gammaBound

            # Enregistrement pour le fichier de convergence détaillé
            for ir, r_val in enumerate(centersRadius):
                conv_data.append({
                    'r': r_val,
                    't': timeSim,
                    'V_eff': Veff[ir],
                    'alpha': np.degrees(Alpha[ir]),
                    'Fn': Fn[ir],
                    'Ft': Ft[ir],
                    'err_point_fixe_b0': err_b0[ir],
                    'err_point_fixe_b1': err_b1[ir],
                    'err_point_fixe_b2': err_b2[ir]
                })

            # Capture pour la moyenne périodique (les 3 derniers tours)
            if it >= start_avg_it:
                t_idx = int((it - start_avg_it) // steps_per_rotation)
                a_idx = int((it - start_avg_it) % steps_per_rotation)
                if t_idx < N_avg:
                    Fn_history[t_idx, a_idx, :] = Fn
                    Ft_history[t_idx, a_idx, :] = Ft
                    Veff_history[t_idx, a_idx, :] = Veff
                    Alpha_history[t_idx, a_idx, :] = Alpha
            
            if it % 50 == 0: print(f" Itération {it}/{total_steps}")

        # 1. Sauvegarde du fichier de convergence en Excel (.xlsx)
        df_conv = pd.DataFrame(conv_data)
        conv_path = os.path.join(outDir, f'conv_yaw{yaw_val}_tsr{tsr_val}.xlsx')
        df_conv.to_excel(conv_path, index=False)

        # 2. Moyennes temporelles et calcul d'erreurs pour le dataset global
        Fn_mean = np.mean(Fn_history, axis=0)
        Ft_mean = np.mean(Ft_history, axis=0)
        Veff_mean = np.mean(Veff_history, axis=0)
        Alpha_mean = np.mean(Alpha_history, axis=0)
        
        # Erreur de périodicité
        err_period_n = np.max(Fn_history, axis=0) - np.min(Fn_history, axis=0)
        err_period_t = np.max(Ft_history, axis=0) - np.min(Ft_history, axis=0)

        # Stockage dans la structure du dataset global
        for a_idx in range(steps_per_rotation):
            theta = a_idx * DegreesPerTimeStep
            for ir, r_val in enumerate(centersRadius):
                global_dataset.append({
                    'r': r_val,
                    'theta': theta,
                    'yaw': yaw_val,
                    'TSR': tsr_val,
                    'V_eff': Veff_mean[a_idx, ir],
                    'alpha': np.degrees(Alpha_mean[a_idx, ir]),
                    'Fn': Fn_mean[a_idx, ir],
                    'Ft': Ft_mean[a_idx, ir],
                    'err_period_n': err_period_n[a_idx, ir],
                    'err_period_t': err_period_t[a_idx, ir]
                })
        
        print(f" Terminé en {time.time() - case_start:.1f}s")

# -----------------------------------------------------------------------------
# Sauvegarde finale du Dataset global en Excel (.xlsx)
# -----------------------------------------------------------------------------
df_dataset = pd.DataFrame(global_dataset)
dataset_path = os.path.join(outDir, 'dataset_forces_mexico.xlsx')
df_dataset.to_excel(dataset_path, index=False)

print(f"\nSimulation globale achevée en {(time.time() - global_start_time)/60:.1f} min.")
print(f"Dataset sauvegardé : {dataset_path} (Shape: {df_dataset.shape})")