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
from scipy import interpolate

# Dossier de sortie
outDir = 'outputs_maths'
if not os.path.exists(outDir):
    os.makedirs(outDir)

# -----------------------------------------------------------------------------
# Fonction de création de la turbine Mexico (Maillage Uniforme Adaptatif)
# -----------------------------------------------------------------------------
def NewMexicoWindTurbine(windVelocity, density, nearWakeLength, n_sections):
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

    R_max = r_nodes_orig[-1]

    # Discrétisation spatiale uniforme
    nodesRadius_new = np.linspace(hubRadius, R_max, n_sections + 1)
    centersRadius = 0.5 * (nodesRadius_new[1:] + nodesRadius_new[:-1])

    nodesTwistAngles = np.interp(nodesRadius_new, r_nodes_orig, twist_orig)
    nodesChord = np.interp(nodesRadius_new, r_nodes_orig, chord_orig)
    
    f_foil = interpolate.interp1d(r_nodes_orig, intAirfoils, kind='nearest', fill_value="extrapolate")
    centersAirfoils = []

    for i in range(len(centersRadius)):
        idx_foil = int(f_foil(centersRadius[i]))
        foilName = str(dataAirfoils[idx_foil])
        centersAirfoils.append(Airfoil('./geometry/' + foilName, headerLength=1))

    myWT = windTurbine(nBlades, [0., 0., 0.], hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius_new, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)

    return blades, myWT, windVelocity, density, 0.01, 1e-4

# -----------------------------------------------------------------------------
# Paramètres de simulation
# -----------------------------------------------------------------------------
nRotations = 1.0          
DegreesPerTimeStep = 10.  
rotationsKeptInWake = 3   
nearWakeLength = 360 * rotationsKeptInWake
innerIter = 10            
density = 1.191           
total_steps = int((nRotations * 360.) / DegreesPerTimeStep)

Omega = 44.5163679 
R_max = 2.46 
tsr_val = 8.0             
yaw_rad = 0.0             

V_mag = (Omega * R_max) / tsr_val
uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

resolutions_n = [10, 20, 40]

print(f"--- DÉMARRAGE DES ÉTUDES DE SENSIBILITÉ ---")
global_start = time.time()

for n in resolutions_n:
    print(f"\n=============================================")
    print(f" SIMULATION POUR n = {n} SECTIONS")
    print(f"=============================================")
    
    # 1. On réinitialise l'analyseur global
    analyzer.reset()

    # 2. Initialisation de la turbine pour ce n
    Blades, WindTurbine, _, _, deltaFlts, _ = NewMexicoWindTurbine(uInfty, density, nearWakeLength, n)

    timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
    refAzimuth = -WindTurbine.rotationalVelocity * timeStep
    timeSim = 0.
    
    for it in range(total_steps):
        step_start = time.time()
        
        refAzimuth += WindTurbine.rotationalVelocity * timeStep
        WindTurbine.updateTurbine(refAzimuth)
        timeSim += timeStep
        
        update(Blades, uInfty, timeStep, timeSim, innerIter, deltaFlts, global_start, [])
        print(f"  Pas {it+1}/{total_steps} achevé en {time.time() - step_start:.1f} s")

    # -------------------------------------------------------------------------
    # Extractions des résultats pour le n courant
    # -------------------------------------------------------------------------
    # A. Métriques globales (eta, L, rho)
    df_metrics = pd.DataFrame({
        'Time_Step': range(1, total_steps + 1),
        'Max_Eta_Prime': analyzer.eta_primes_history,
        'K_Spectral_Radius': analyzer.spectral_radii_K,
        'K_Infinity_Norm': analyzer.K_infinity_norms,
        'Empirical_Lipschitz': analyzer.lipschitz_empirical
    })
    df_metrics.to_excel(os.path.join(outDir, f'math_metrics_n{n}.xlsx'), index=False)

    # B. Résidus complets (Tous les algos, pour TOUS les pas de temps)
    all_residuals = []
    for t_idx in range(total_steps):
        for iter_idx in range(innerIter):
            # Sécurité pour éviter les index out of bounds si Newton s'arrête avant innerIter
            res_pic = analyzer.picard_residuals[t_idx][iter_idx] if iter_idx < len(analyzer.picard_residuals[t_idx]) else np.nan
            res_relax = analyzer.picard_relax_residuals[t_idx][iter_idx] if iter_idx < len(analyzer.picard_relax_residuals[t_idx]) else np.nan
            res_newton = analyzer.newton_residuals[t_idx][iter_idx] if iter_idx < len(analyzer.newton_residuals[t_idx]) else np.nan
            
            all_residuals.append({
                'Time_Step': t_idx + 1,
                'Iteration': iter_idx + 1,
                'Picard_Pur': res_pic,
                'Picard_Relax': res_relax,
                'Newton': res_newton
            })
    pd.DataFrame(all_residuals).to_csv(os.path.join(outDir, f'residuals_all_steps_n{n}.csv'), index=False)

    # C. Valeurs propres (Pour TOUS les pas de temps)
    all_eigvals = []
    for t_idx in range(total_steps):
        eigs = analyzer.jacobian_eigenvalues_history[t_idx]
        for e in eigs:
            all_eigvals.append({
                'Time_Step': t_idx + 1,
                'Real': np.real(e),
                'Imag': np.imag(e)
            })
    pd.DataFrame(all_eigvals).to_csv(os.path.join(outDir, f'eigenvalues_all_steps_n{n}.csv'), index=False)

    print(f"-> Fichiers exportés pour n={n}.")

print(f"\nCampagne complète achevée en {(time.time() - global_start)/60:.1f} minutes.")