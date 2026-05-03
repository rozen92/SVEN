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
# Paramètres de simulation MODIFIÉS
# -----------------------------------------------------------------------------
nRotations = 5.0          
DegreesPerTimeStep = 10.  
rotationsKeptInWake = 5   
nearWakeLength = 360 * rotationsKeptInWake
innerIter = 20            
density = 1.191           
total_steps = int((nRotations * 360.) / DegreesPerTimeStep)
steps_per_rotation = int(360. / DegreesPerTimeStep)

start_analysis_step = 2 * steps_per_rotation 

Omega = 44.5163679 
R_max = 2.46 
tsr_val = 8.0             
yaw_rad = 0.0             

V_mag = (Omega * R_max) / tsr_val
uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

resolutions_n = [5, 10, 20, 40]
# Dictionnaire adaptatif pour eta_relax en fonction de la résolution
eta_relax_dict = {5: 1.0, 10: 0.85, 20: 0.55, 40: 0.30}

print(f"--- DÉMARRAGE DES ÉTUDES DE SENSIBILITÉ ---")
global_start = time.time()

for n in resolutions_n:
    print(f"\n=============================================")
    print(f" SIMULATION POUR n = {n} SECTIONS")
    print(f"=============================================")
    
    analyzer.reset()
    Blades, WindTurbine, _, _, deltaFlts, _ = NewMexicoWindTurbine(uInfty, density, nearWakeLength, n)
    
    # Application du facteur de relaxation adapté à n
    current_eta = eta_relax_dict.get(n, 0.05)
    for b in Blades:
        b.relax = current_eta
    print(f" -> Facteur de relaxation appliqué : {current_eta}")

    timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
    refAzimuth = -WindTurbine.rotationalVelocity * timeStep
    timeSim = 0.
    
    analyzed_time_steps = [] # Liste pour garder trace précise des pas enregistrés
    
    for it in range(total_steps):
        step_start = time.time()
        
        refAzimuth += WindTurbine.rotationalVelocity * timeStep
        WindTurbine.updateTurbine(refAzimuth)
        timeSim += timeStep
        
        # Activation au tout 1er pas (it=0) ET après le 2ème tour
        analyzer.active = (it == 0) or (it >= start_analysis_step)
        
        update(Blades, uInfty, timeStep, timeSim, innerIter, deltaFlts, global_start, [])
        
        if analyzer.active:
            analyzed_time_steps.append(it + 1)
            
        if it % 5 == 0 or analyzer.active:
            print(f"  Pas {it+1}/{total_steps} achevé en {time.time() - step_start:.1f} s {'(Analyse Math en cours)' if analyzer.active else ''}")

    # -------------------------------------------------------------------------
    # Extractions 
    # -------------------------------------------------------------------------
    # A. Métriques globales
    df_metrics = pd.DataFrame({
        'Time_Step': analyzed_time_steps,
        'Max_Eta_Prime': analyzer.eta_primes_history,
        'K_Spectral_Radius': analyzer.spectral_radii_K,
        'K_Infinity_Norm': analyzer.K_infinity_norms,
        'K_Prime_Init_Spectral_Radius': analyzer.spectral_radii_K_prime_init, 
        'K_Prime_Sol_Spectral_Radius': analyzer.spectral_radii_K_prime_sol,   
        'Empirical_Lipschitz_Pur': analyzer.lipschitz_empirical,
        'Empirical_Lipschitz_Relax': analyzer.lipschitz_empirical_relax,      
        'Condition_Number_Init': analyzer.jacobian_condition_numbers_init,
        'Condition_Number_Sol': analyzer.jacobian_condition_numbers_sol,
        'FD_Verification_Error': analyzer.fd_verification_errors              
    })
    df_metrics.to_excel(os.path.join(outDir, f'math_metrics_n{n}.xlsx'), index=False)

    # B. Résidus complets
    all_residuals = []
    for i, t_idx_real in enumerate(analyzed_time_steps):
        for iter_idx in range(innerIter):
            res_pic = analyzer.picard_residuals[i][iter_idx] if iter_idx < len(analyzer.picard_residuals[i]) else np.nan
            res_relax = analyzer.picard_relax_residuals[i][iter_idx] if iter_idx < len(analyzer.picard_relax_residuals[i]) else np.nan
            res_newton = analyzer.newton_residuals[i][iter_idx] if iter_idx < len(analyzer.newton_residuals[i]) else np.nan
            
            all_residuals.append({
                'Time_Step': t_idx_real,
                'Iteration': iter_idx + 1,
                'Picard_Pur': res_pic,
                'Picard_Relax': res_relax,
                'Newton': res_newton
            })
    pd.DataFrame(all_residuals).to_csv(os.path.join(outDir, f'residuals_all_steps_n{n}.csv'), index=False)

    # C. Valeurs propres (f non relaxée, évaluées en Init et Sol)
    all_eigvals = []
    for i, t_idx_real in enumerate(analyzed_time_steps):
        # VP Initiales
        eigs_init = analyzer.jacobian_eigenvalues_init[i]
        for e in eigs_init:
            all_eigvals.append({
                'Time_Step': t_idx_real,
                'State': 'Init',
                'Real': np.real(e),
                'Imag': np.imag(e)
            })
            
        # VP Solution
        eigs_sol = analyzer.jacobian_eigenvalues_sol[i]
        for e in eigs_sol:
            all_eigvals.append({
                'Time_Step': t_idx_real,
                'State': 'Sol',
                'Real': np.real(e),
                'Imag': np.imag(e)
            })
            
    pd.DataFrame(all_eigvals).to_csv(os.path.join(outDir, f'eigenvalues_all_steps_n{n}.csv'), index=False)

    print(f"-> Fichiers exportés pour n={n}.")

print(f"\nCampagne complète achevée en {(time.time() - global_start)/60:.1f} minutes.")