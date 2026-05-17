import os
import sys
import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d

# --- Configuration des chemins ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)
sys.path.append(parent_of_project_dir)

from sven.windTurbine import *
from sven.airfoil import *
from sven.blade import *
from sven.solver import update

outDir = 'outputs_discretization_study'
if not os.path.exists(outDir):
    os.makedirs(outDir)

def NewMexicoWindTurbine(windVelocity, density, nearWakeLength):
    sign = -1.0; hubRadius = 0.210; nBlades = 3; rotationalVelocity = 44.5163679; bladePitch = sign * 0.040143
    
    # 1. Chargement des rayons exacts de CASTOR pour caler les milieux de sections
    castor_file = os.path.join(script_dir, 'CASTOR_data.dat')
    castor_raw = np.genfromtxt(castor_file, skip_header=3)
    r_targets = castor_raw[:, 0]
    N = len(r_targets)
    
    # 2. Chargement de la géométrie originale pour interpolation
    geom_file = os.path.join(script_dir, 'geometry', 'blade.dat')
    data_geom = np.genfromtxt(geom_file, skip_header=1, dtype=str)
    r_orig = data_geom[:, 0].astype(float)
    twist_orig = -1.0 * data_geom[:, 1].astype(float) 
    chord_orig = np.abs(data_geom[:, 2].astype(float))
    airfoil_orig = data_geom[:, 3]
    
    # Interpolation linéaire des propriétés de la pale sur les rayons CASTOR
    f_twist = interp1d(r_orig, twist_orig, kind='linear', fill_value="extrapolate")
    f_chord = interp1d(r_orig, chord_orig, kind='linear', fill_value="extrapolate")
    
    twist_targets = f_twist(r_targets)
    chord_targets = f_chord(r_targets)
    
    # Mapping des profils aérodynamiques (plus proche voisin)
    airfoil_names = []
    for r in r_targets:
        idx_nearest = np.argmin(np.abs(r_orig - r))
        airfoil_names.append(airfoil_orig[idx_nearest])
        
    # 3. Reconstruction des nœuds géométriques pour forcer le milieu exact
    nodesRadius = np.zeros(N + 1); nodesChord = np.zeros(N + 1); nodesTwistAngles = np.zeros(N + 1)
    nodesRadius[0] = hubRadius; nodesChord[0] = chord_targets[0]; nodesTwistAngles[0] = twist_targets[0]
    
    for i in range(N):
        nodesRadius[i+1] = 2 * r_targets[i] - nodesRadius[i]
        nodesChord[i+1] = 2 * chord_targets[i] - nodesChord[i]
        nodesTwistAngles[i+1] = 2 * twist_targets[i] - nodesTwistAngles[i]
        
    centersAirfoils = [Airfoil(os.path.join(script_dir, 'geometry', 'Airfoils2', f"{n}.foil"), 1) for n in airfoil_names]
    myWT = windTurbine(nBlades, [0., 0., 0.], hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)
    for b in blades: b.centerChords = chord_targets.copy()
    
    return blades, myWT, 0.01, 1e-6 

# -----------------------------------------------------------------------------
# Paramètres de l'étude de discrétisation
# -----------------------------------------------------------------------------
base_rotations = 10        
max_extra_rotations = 10   
max_rotations = base_rotations + max_extra_rotations  
DegreesPerTimeStep = 5.0   
N_avg = 3
steps_per_rotation = int(360.0 / DegreesPerTimeStep)

# Paramètres de la stratégie "Test & Rollback"
max_picard_iters = 5000  
p_block = 5           
n_block = 5           

# Les 3 cas EXACTS de Caroline (Vitesse, Densité, Index CASTOR)
caroline_cases = [
    {'label': '10', 'V': 10.05, 'rho': 1.197},
    {'label': '15', 'V': 15.06, 'rho': 1.191},
    {'label': '24', 'V': 24.05, 'rho': 1.195}
]

global_start_time = time.time()
file_log = os.path.join(outDir, 'log_discretization.txt')
log = open(file_log, 'a', encoding='utf-8')

print(f"Lancement de la Campagne d'Étude de Discrétisation (Yaw = 0°)")
log.write(f"Lancement de la Campagne d'Étude de Discrétisation (Yaw = 0°)\n\n")

# Chargement initial des données CASTOR
castor_file = os.path.join(script_dir, 'CASTOR_data.dat')
castor_raw = np.genfromtxt(castor_file, skip_header=3)

for case in caroline_cases:
    case_start = time.time()
    V_exact = case['V']
    rho_exact = case['rho']
    case_label = case['label']
    
    uInfty = np.array([V_exact, 0.0, 0.0], dtype=np.float32)
    
    print(f"\n#################### CAS : {case_label} (V={V_exact} m/s, rho={rho_exact}) ####################")
    log.write(f"#################### CAS : {case_label} (V={V_exact} m/s, rho={rho_exact}) ####################\n")
    
    # On passe la densité exacte à l'initialisation
    Blades, WT, deltaFlts, tol_hybrid = NewMexicoWindTurbine(uInfty, rho_exact, 3600)
    cR = 0.5 * (WT.nodesRadius[1:] + WT.nodesRadius[:-1])
    tStep = np.radians(DegreesPerTimeStep) / WT.rotationalVelocity
    total_max_steps = int((max_rotations * 360.) / DegreesPerTimeStep)

    Fn_history = np.zeros((N_avg, steps_per_rotation, len(cR)))
    Ft_history = np.zeros_like(Fn_history)

    Fn_current_rot = np.zeros((steps_per_rotation, len(cR)))
    Ft_current_rot = np.zeros_like(Fn_current_rot)

    current_relax = 0.35 

    for it in range(total_max_steps):
        WT.updateTurbine(WT.rotationalVelocity * tStep * (it+1))
        
        m_err, st, p_its, n_its, win, e_opt, j_ev, j_ok = update(
            Blades, uInfty, tStep, 0, max_picard_iters, 
            deltaFlts, global_start_time, [], 
            algo_type="hybrid", tol=tol_hybrid, 
            p_block=p_block, n_block=n_block, current_relax=current_relax
        )
        
        if e_opt > 0:
            current_relax = min(0.35, 0.9 * e_opt)

        if m_err > tol_hybrid:
            log.write(f" [MAXI] Pas {it+1:3} | Précision non atteinte | Err:{m_err:.1e}\n")

        idx_rot = it // steps_per_rotation
        idx_azi = it % steps_per_rotation
        hist_idx = idx_rot % N_avg  
        
        # Évaluation des forces avec la densité EXACTE du cas
        Fn, Ft = WT.evaluateForces(rho_exact)
        
        Fn_history[hist_idx, idx_azi, :] = Fn
        Ft_history[hist_idx, idx_azi, :] = Ft
        Fn_current_rot[idx_azi, :] = Fn
        Ft_current_rot[idx_azi, :] = Ft

        if idx_azi == steps_per_rotation - 1:
            completed_rotations = idx_rot + 1
            
            Fn_ptp_intra = np.max(np.ptp(Fn_current_rot, axis=0))
            Ft_ptp_intra = np.max(np.ptp(Ft_current_rot, axis=0))
            
            if completed_rotations >= base_rotations:
                Fn_ptp_inter = np.max(np.ptp(Fn_history, axis=0)) 
                Ft_ptp_inter = np.max(np.ptp(Ft_history, axis=0))
                Fn_mean_inter = np.mean(np.abs(Fn_history))
                Ft_mean_inter = np.mean(np.abs(Ft_history))
                Fn_rel = (Fn_ptp_inter / Fn_mean_inter * 100) if Fn_mean_inter > 0 else 0.0
                Ft_rel = (Ft_ptp_inter / Ft_mean_inter * 100) if Ft_mean_inter > 0 else 0.0

                bilan_str = (f"        -> [BILAN TOUR {completed_rotations:02d}] "
                             f"Périodicité Inter: Fn={Fn_rel:.3f}% Ft={Ft_rel:.3f}% | "
                             f"Erreur Amplitude Intra: Fn={Fn_ptp_intra:.2e} Ft={Ft_ptp_intra:.2e}")
                print(bilan_str)
                log.write(bilan_str + "\n")

                if Fn_rel <= 0.1 and Ft_rel <= 0.1:
                    print(f"        => Convergence atteinte en {completed_rotations} tours !")
                    break
                elif completed_rotations == max_rotations:
                    print(f"        => Limite de {max_rotations} tours atteinte.")
                    break

    Fn_final_sven = np.mean(Fn_current_rot, axis=0)
    Ft_final_sven = np.mean(Ft_current_rot, axis=0)
    
    if case_label == '10':
        Fn_castor = castor_raw[:, 1]; Ft_castor = castor_raw[:, 2]
    elif case_label == '15':
        Fn_castor = castor_raw[:, 3]; Ft_castor = castor_raw[:, 4]
    else:
        Fn_castor = castor_raw[:, 5]; Ft_castor = castor_raw[:, 6]
        
    rmse_Fn = np.sqrt(np.mean((Fn_final_sven - Fn_castor) ** 2))
    rmse_Ft = np.sqrt(np.mean((Ft_final_sven - Ft_castor) ** 2))
    
    rmse_str = f">> [RÉSULTAT CAS {case_label}] RMSE -> Fn: {rmse_Fn:.4f} | Ft: {rmse_Ft:.4f} (en {time.time() - case_start:.1f}s)"
    print(rmse_str)
    log.write("\n" + rmse_str + "\n\n" + "#"*100 + "\n")
    
    df_res = pd.DataFrame({
        'r_target': cR,
        'Fn_SVEN_mean': Fn_final_sven,
        'Ft_SVEN_mean': Ft_final_sven,
        'Fn_CASTOR': Fn_castor,
        'Ft_CASTOR': Ft_castor,
        'Fn_Osci_Intra_PTP': np.ptp(Fn_current_rot, axis=0),
        'Ft_Osci_Intra_PTP': np.ptp(Ft_current_rot, axis=0)
    })
    df_res.to_csv(os.path.join(outDir, f'radiales_forces_cas_{case_label}.csv'), index=False)

print(f"\nETUDE DE DISCRETISATION COMPLETEE en {time.time() - global_start_time:.1f}s.")
log.write(f"\nETUDE DE DISCRETISATION COMPLETEE en {time.time() - global_start_time:.1f}s.")
log.close()