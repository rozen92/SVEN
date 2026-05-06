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
# Paramètres de simulation
# -----------------------------------------------------------------------------
nRotations = 5.0          
DegreesPerTimeStep = 10.  
rotationsKeptInWake = 5   
nearWakeLength = 360 * rotationsKeptInWake
innerIter = 20            
density = 1.191           
total_steps = int((nRotations * 360.) / DegreesPerTimeStep)
steps_per_rotation = int(360. / DegreesPerTimeStep)

# Début de l'analyse stabilisée au 3ème tour
start_analysis_step = 2 * steps_per_rotation 

Omega = 44.5163679 
R_max = 2.46 
tsr_val = 8.0             
yaw_rad = 0.0             

V_mag = (Omega * R_max) / tsr_val
uInfty = np.array([V_mag * np.cos(yaw_rad), V_mag * np.sin(yaw_rad), 0.0], dtype=np.float32)

resolutions_n = [5, 10, 20, 40]
# Dictionnaire des facteurs de relaxation optimisés basés sur l'étude de stabilité
eta_relax_dict = {5: 1.0, 10: 0.85, 20: 0.55, 40: 0.30}

print(f"--- DÉMARRAGE DE LA CAMPAGNE D'ANALYSE MATHÉMATIQUE ---")
global_start = time.time()

for n in resolutions_n:
    # On définit les facteurs à tester pour cette résolution
    eta_opt = eta_relax_dict.get(n, 0.05)
    eta_list = [eta_opt]
    if eta_opt != 0.05:
        eta_list.append(0.05) # On rajoute le cas par défaut pour comparaison
        
    for current_eta in eta_list:
        print(f"\n========================================================")
        print(f" CONFIGURATION : n = {n} | eta_relax = {current_eta}")
        print(f"========================================================")
        
        analyzer.reset()
        Blades, WindTurbine, _, _, deltaFlts, _ = NewMexicoWindTurbine(uInfty, density, nearWakeLength, n)
        
        # Application du facteur de relaxation
        for b in Blades:
            b.relax = current_eta

        timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
        refAzimuth = -WindTurbine.rotationalVelocity * timeStep
        timeSim = 0.
        analyzed_time_steps = [] 
        
        for it in range(total_steps):
            step_start = time.time()
            
            refAzimuth += WindTurbine.rotationalVelocity * timeStep
            WindTurbine.updateTurbine(refAzimuth)
            timeSim += timeStep
            
            # Activation au Pas 1 (Choc) ET régime établi (Tours 3 à 5)
            analyzer.active = (it == 0) or (it >= start_analysis_step)
            
            # Mise à jour du sillage et résolution du point fixe
            update(Blades, uInfty, timeStep, timeSim, innerIter, deltaFlts, global_start, [])
            
            if analyzer.active:
                analyzed_time_steps.append(it + 1)
                
            if it % 10 == 0 or analyzer.active:
                msg = f"  Pas {it+1}/{total_steps}"
                if it == 0: msg += " (Analyse Choc Initial)"
                elif it >= start_analysis_step: msg += " (Analyse Stabilisée)"
                print(msg)

        # -------------------------------------------------------------------------
        # Exportation des données
        # -------------------------------------------------------------------------
        file_suffix = f"n{n}_eta{current_eta}"
        print(f" -> Exportation des résultats pour {file_suffix}...")

        # 1. Métriques globales (Conditionnement, Bornes K, FD error, Symétrie, Diagonale Dominante)
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
            'FD_Verification_Error': analyzer.fd_verification_errors,
            'J_Symmetry_Error_Init': analyzer.jacobian_symmetry_errors_init,
            'J_Symmetry_Error_Sol': analyzer.jacobian_symmetry_errors_sol,
            'J_Diag_Dominant_Init': analyzer.J_diag_dominant_init,
            'J_Diag_Dominant_Sol': analyzer.J_diag_dominant_sol
        })
        df_metrics.to_excel(os.path.join(outDir, f'math_metrics_{file_suffix}.xlsx'), index=False)

        # 1.bis. Historique de dCL_dalpha
        all_dCL = []
        for i, t_idx_real in enumerate(analyzed_time_steps):
            dCL_array = analyzer.dCL_dalpha_history[i]
            for sec_idx, val in enumerate(dCL_array):
                all_dCL.append({
                    'Time_Step': t_idx_real,
                    'Section_Index': sec_idx,
                    'dCL_dalpha': val
                })
        pd.DataFrame(all_dCL).to_csv(os.path.join(outDir, f'dCL_dalpha_history_{file_suffix}.csv'), index=False)

        # 2. Résidus détaillés par itération interne
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
        pd.DataFrame(all_residuals).to_csv(os.path.join(outDir, f'residuals_all_steps_{file_suffix}.csv'), index=False)

        # 3. Valeurs propres de la Jacobienne (f non relaxée)
        all_eigvals = []
        for i, t_idx_real in enumerate(analyzed_time_steps):
            # VP Initiales (Init)
            for e in analyzer.jacobian_eigenvalues_init[i]:
                all_eigvals.append({
                    'Time_Step': t_idx_real, 'State': 'Init', 'Real': np.real(e), 'Imag': np.imag(e)
                })
            # VP à la solution convergée (Sol)
            for e in analyzer.jacobian_eigenvalues_sol[i]:
                all_eigvals.append({
                    'Time_Step': t_idx_real, 'State': 'Sol', 'Real': np.real(e), 'Imag': np.imag(e)
                })
        pd.DataFrame(all_eigvals).to_csv(os.path.join(outDir, f'eigenvalues_all_steps_{file_suffix}.csv'), index=False)

print(f"\nCampagne complète achevée en {(time.time() - global_start)/60:.1f} minutes.")