import os
import sys
import numpy as np
import pandas as pd
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)
sys.path.append(parent_of_project_dir)

from sven.windTurbine import *
from sven.airfoil import *
from sven.blade import *
from sven.solver import update, analyzer
from scipy import interpolate

outDir = 'outputs_comparaison'
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

    return blades, myWT

# --- PARAMÈTRES DU TEST ---
n_sections = 40
yaw_val = 0.0
tsr_val = 8.0
density = 1.191

nRotations = 10.0          
DegreesPerTimeStep = 10.  
nearWakeLength = 360 * int(nRotations)
innerIter = 12
tol_point_fixe = 1e-5

Omega = 44.5163679 
R_max = 2.46 
V_mag = (Omega * R_max) / tsr_val
uInfty = np.array([V_mag, 0.0, 0.0], dtype=np.float32)

total_steps = int((nRotations * 360.) / DegreesPerTimeStep)

# --- SCÉNARIOS ---
scenarios = [
    {"name": "Picard_eta0.05", "algo": "picard", "eta": 0.05},
    {"name": "Picard_eta0.30", "algo": "picard", "eta": 0.30},
    {"name": "Newton",         "algo": "newton", "eta": 1.00} # eta ignoré par Newton
]

print(f"=== COMPARAISON DES PERFORMANCES DES SOLVEURS ===")
print(f"Configuration : n={n_sections}, TSR={tsr_val}, Yaw={yaw_val}°, {total_steps} pas de temps.\n")

for sc in scenarios:
    print(f"Lancement de la simulation : {sc['name']}")
    
    # On désactive l'analyse mathématique fantôme pour mesurer la vraie vitesse
    analyzer.active = False 
    
    Blades, WindTurbine = NewMexicoWindTurbine(uInfty, density, nearWakeLength, n_sections)
    centersRadius = 0.5 * (WindTurbine.nodesRadius[1:] + WindTurbine.nodesRadius[:-1])
    
    for b in Blades:
        b.relax = sc['eta']
        
    timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
    refAzimuth = -WindTurbine.rotationalVelocity * timeStep
    timeSim = 0.
    
    results = []
    global_start = time.time()
    
    for it in range(total_steps):
        refAzimuth += WindTurbine.rotationalVelocity * timeStep
        WindTurbine.updateTurbine(refAzimuth)
        timeSim += timeStep
        
        # Le solveur renvoie maintenant l'erreur max et son propre temps d'exécution
        max_err, solver_t = update(
            Blades, uInfty, timeStep, timeSim, innerIter, 
            0.01, global_start, [], 
            algo_type=sc['algo'], tol=tol_point_fixe
        )
        
        # Évaluation des efforts
        Fn, Ft = WindTurbine.evaluateForces(density)
        
        # Extraction de la pale 0 pour le profil de la pale
        Veff = WindTurbine.blades[0].effectiveVelocity
        Alpha = WindTurbine.blades[0].attackAngle
        
        # Erreur de point fixe propre à chaque section de la pale 0
        err_sections = np.abs(WindTurbine.blades[0].f_Gamma - WindTurbine.blades[0].gammaBound)
        
        for ir, r_val in enumerate(centersRadius):
            results.append({
                'Time_Step': it + 1,
                'Time_s': timeSim,
                'Radius': r_val,
                'Solver_Time_s': solver_t,      # Temps pris par le solveur à ce pas
                'Max_Global_Err': max_err,      # Erreur max toutes pales confondues
                'Section_PF_Err': err_sections[ir],
                'Fn': Fn[ir],
                'Ft': Ft[ir],
                'V_eff': Veff[ir],
                'Alpha_deg': np.degrees(Alpha[ir])
            })
            
        if (it + 1) % 50 == 0:
            print(f"  Pas {it+1}/{total_steps} achevé.")

    # Sauvegarde
    df = pd.DataFrame(results)
    save_path = os.path.join(outDir, f"perf_{sc['name']}.csv")
    df.to_csv(save_path, index=False)
    
    print(f"{sc['name']} terminé en {time.time() - global_start:.1f}s. Données sauvegardées.\n")

print("Toutes les comparaisons sont terminées !")