import os
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# --- Configuration des chemins ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)
sys.path.append(parent_of_project_dir)

from sven.windTurbine import *
from sven.airfoil import *
from sven.blade import *
from sven.solver import update, analyzer # <-- Import de l'analyseur global
from scipy import interpolate

# Dossier de sortie
outDir = 'outputs_maths'
if not os.path.exists(outDir):
    os.makedirs(outDir)

# -----------------------------------------------------------------------------
# Fonction de création de la turbine Mexico (identique à main.py)
# -----------------------------------------------------------------------------
def NewMexicoWindTurbine(windVelocity, density, nearWakeLength):
    sign = -1.
    hubRadius = 0.210
    nBlades = 3
    rotationalVelocity = 44.5163679  
    bladePitch = sign * 0.040143
    
    dataAirfoils = np.genfromtxt('./geometry/mexico.blade', skip_header=1, usecols=(7), dtype='U')
    intAirfoils = np.arange(0, len(dataAirfoils))
    data_orig = np.genfromtxt('./geometry/mexico.blade', skip_header=1)
    
    refRadius_orig = data_orig[:, 2] 
    r_nodes_orig = hubRadius + refRadius_orig
    twist_orig = -sign * np.radians(data_orig[:, 5])
    chord_orig = data_orig[:, 6]

    df_aero = pd.read_csv('./data_AeroDeeP.csv', skiprows=7)
    r_targets = df_aero['radius'].values 
    
    N = len(r_targets)
    nodesRadius_new = np.zeros(N + 1)
    nodesRadius_new[0] = hubRadius
    
    for i in range(N):
        nodesRadius_new[i+1] = 2 * r_targets[i] - nodesRadius_new[i]

    nodesTwistAngles = np.interp(nodesRadius_new, r_nodes_orig, twist_orig)
    nodesChord = np.interp(nodesRadius_new, r_nodes_orig, chord_orig)
    
    f_foil = interpolate.interp1d(r_nodes_orig, intAirfoils, kind='nearest', fill_value="extrapolate")
    centersAirfoils = []

    for i in range(len(r_targets)):
        idx_foil = int(f_foil(r_targets[i]))
        foilName = str(dataAirfoils[idx_foil])
        centersAirfoils.append(Airfoil('./geometry/' + foilName, headerLength=1))

    myWT = windTurbine(nBlades, [0., 0., 0.], hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius_new, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)

    return blades, myWT, windVelocity, density, 0.01, 1e-4

# -----------------------------------------------------------------------------
# Paramètres d'analyse mathématique (Un seul point de fonctionnement)
# -----------------------------------------------------------------------------
nRotations = 1.0          # 1 rotation suffit pour observer la dynamique de convergence
DegreesPerTimeStep = 10.  
rotationsKeptInWake = 3   # Sillage plus court pour alléger les calculs de Biot-Savart
nearWakeLength = 360 * rotationsKeptInWake
innerIter = 10            # Itérations internes du point fixe
density = 1.191           
total_steps = int((nRotations * 360.) / DegreesPerTimeStep)

Omega = 44.5163679 
R_max = 2.46 
tsr_val = 8.0             # TSR nominal
yaw_rad = 0.0             # Pas de Yaw pour l'instant

V_mag = (Omega * R_max) / tsr_val
uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

print(f"--- DÉMARRAGE ANALYSE MATHÉMATIQUE ---")
print(f"Configuration : TSR = {tsr_val}, V_vent = {V_mag:.2f} m/s, Rotations = {nRotations}")
print(f"Attention : Le calcul de la Jacobienne exacte ralentit significativement l'exécution.\n")

# -----------------------------------------------------------------------------
# Initialisation et Boucle Temporelle
# -----------------------------------------------------------------------------
Blades, WindTurbine, _, _, deltaFlts, _ = NewMexicoWindTurbine(uInfty, density, nearWakeLength)

timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
refAzimuth = -WindTurbine.rotationalVelocity * timeStep
timeSim = 0.
global_start = time.time()

for it in range(total_steps):
    step_start = time.time()
    
    refAzimuth += WindTurbine.rotationalVelocity * timeStep
    WindTurbine.updateTurbine(refAzimuth)
    timeSim += timeStep
    
    # Appel de SVEN (qui appelle notre analyzer en interne)
    update(Blades, uInfty, timeStep, timeSim, innerIter, deltaFlts, global_start, [])
    
    print(f"Pas {it+1}/{total_steps} achevé en {time.time() - step_start:.1f} s")

# -----------------------------------------------------------------------------
# Extraction et Sauvegarde des Diagnostics Mathématiques
# -----------------------------------------------------------------------------
print(f"\n--- EXTRACTION DES DONNÉES MATHÉMATIQUES ---")

# 1. Création d'un DataFrame global pour les métriques par pas de temps
df_metrics = pd.DataFrame({
    'Time_Step': range(1, total_steps + 1),
    'Max_Eta_Prime': analyzer.eta_primes_history,
    'K_Spectral_Radius': analyzer.spectral_radii_K,
    'K_Infinity_Norm': analyzer.K_infinity_norms,
    'Empirical_Lipschitz': analyzer.lipschitz_empirical
})

metrics_path = os.path.join(outDir, 'math_metrics_history.xlsx')
df_metrics.to_excel(metrics_path, index=False)
print(f"-> Métriques globales sauvegardées : {metrics_path}")

# 2. Sauvegarde détaillée des résidus (Dernier pas de temps)
# On extrait les courbes de convergence de la toute dernière itération temporelle
last_picard = analyzer.picard_residuals[-1]
last_newton = analyzer.newton_residuals[-1]

df_residuals = pd.DataFrame({
    'Iteration': range(1, innerIter + 1),
    'Picard_Pur_Residual': last_picard + [np.nan]*(innerIter - len(last_picard)),
    'Newton_Residual': last_newton + [np.nan]*(innerIter - len(last_newton))
})

residuals_path = os.path.join(outDir, 'convergence_residuals_last_step.csv')
df_residuals.to_csv(residuals_path, index=False)
print(f"-> Résidus intra-pas sauvegardés : {residuals_path}")

print(f"\nAnalyse terminée en {(time.time() - global_start)/60:.1f} minutes.")