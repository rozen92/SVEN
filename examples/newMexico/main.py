import os
import sys
import torch
import numpy as np
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
yaws_deg = np.array([0.0, 15.0, 30.0]) # Angles de lacet
tsrs = np.array([4.5, 6.6, 9.0])       # Tip Speed Ratios

global_forces = None 
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
        
        timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
        timeEnd = np.radians(nRotations * 360.) / WindTurbine.rotationalVelocity
        timeSteps = np.arange(0., timeEnd, timeStep)

        num_sections = len(Blades[0].centers)
        total_steps = len(timeSteps)
        start_avg_it = total_steps - (N_avg * steps_per_rotation)

        # Allocation mémoire pour ce cas
        if global_forces is None:
            global_forces = np.zeros((len(yaws_deg), len(tsrs), 2, steps_per_rotation, num_sections))
        
        full_history = np.zeros((total_steps, 2, num_sections))
        Fn_history = np.zeros((N_avg, steps_per_rotation, num_sections))
        Ft_history = np.zeros((N_avg, steps_per_rotation, num_sections))

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
            
            # Stockage de l'historique complet (RAM)
            full_history[it, 0, :] = Fn
            full_history[it, 1, :] = Ft

            # Capture pour la moyenne périodique
            if it >= start_avg_it:
                t_idx = int((it - start_avg_it) // steps_per_rotation)
                a_idx = int((it - start_avg_it) % steps_per_rotation)
                if t_idx < N_avg:
                    Fn_history[t_idx, a_idx, :] = Fn
                    Ft_history[t_idx, a_idx, :] = Ft
            
            if it % 50 == 0: print(f" Itération {it}/{total_steps}")

        # Sauvegarde de l'historique de convergence (un seul fichier par cas)
        conv_path = os.path.join(outDir, f'conv_yaw{yaw_val}_tsr{tsr_val}.pt')
        torch.save(torch.tensor(full_history, dtype=torch.float32), conv_path)

        # Stockage dans le tenseur global (moyenne atemporelle)
        global_forces[i_yaw, i_tsr, 0, :, :] = np.mean(Fn_history, axis=0)
        global_forces[i_yaw, i_tsr, 1, :, :] = np.mean(Ft_history, axis=0)
        
        print(f" Terminé en {time.time() - case_start:.1f}s")

# -----------------------------------------------------------------------------
# Sauvegarde finale du Dataset
# -----------------------------------------------------------------------------
dataset_path = os.path.join(outDir, 'dataset_forces_mexico.pt')
torch.save(torch.tensor(global_forces, dtype=torch.float32), dataset_path)

print(f"\nSimulation globale achevée en {(time.time() - global_start_time)/60:.1f} min.")
print(f"Dataset sauvegardé : {dataset_path} (Shape: {global_forces.shape})")