import os
import sys
from ctypes import CDLL, RTLD_GLOBAL


## Changer qblade_root en fonction de la localisation du logiciel dans la machine
## Pour régler le warning dans PROBLEMS, ajouter dans settings.json     "python.analysis.extraPaths": [
##       chemin/vers/dir/QBLADE
##    ]
qblade_root = "/home/arthur/Bureau/QBladeCE_2.0.9.7_unix/QBladeCE_2.0.9.7"
lib_dir = os.path.join(qblade_root, "Libraries")
sil_dir = os.path.join(qblade_root, "SIL_Interface")

from ctypes import *
from QBladeLibrary import QBladeLibrary as qbl
import numpy as np
import pandas as pd
import time
from scipy.stats.qmc import LatinHypercube as lhc

###########################################################################################################################################
################        0. Charger un objet QBlade                                                                         ################
###########################################################################################################################################

SIM = qbl(qblade_root)      # Objet Qbldade
SIM.createInstance()        # Charger une simulation selon le hardware disponible

###########################################################################################################################################
################        1. Charger la géométrie de l'éolienne                                                              ################
###########################################################################################################################################

qbl.loadProjet("Mexico.qpr")



# -----------------------------------------------------------------------------
# Paramètres de la campagne
# -----------------------------------------------------------------------------
base_rotations = 10        # Tours minimum par défaut
max_extra_rotations = 10   # Tours additionnels autorisés
max_rotations = base_rotations + max_extra_rotations  # Soit 20 tours max
DegreesPerTimeStep = 5.0
density = 1.198
N_avg = 3
steps_per_rotation = int(360.0 / DegreesPerTimeStep)
Omega = 44.5163679
R_max = 2.25


## Préparation de l'échantillonnage LHC
sampler = lhc(2, strength = 1, seed = 42)
samples = sampler.random(n = 100)

samples[:,0] = samples[:,0]*8 + 4
samples[:,1] = samples[:,1]*60 - 30

outDir = 'outputs_adaptive_hybrid'
if not os.path.exists(outDir):
    os.makedirs(outDir)
file_log = os.path.join(outDir, 'log_sim.txt')
log = open(file_log, 'a', encoding = 'utf-8')

current_tsr_dataset = []
for i in range(10, 30) :
    tsr_val = samples[i,0]
    yaw_val = samples[i,1]    

    tsr_start = time.time();  
    print(f"#################### (TSR,YAW) : ({tsr_val.round(3)},{yaw_val.round(3)}) ####################")
    log.write(f"\n#################### (TSR,YAW) : ({tsr_val.round(3)},{yaw_val.round(3)}) ####################\n")

    path_sim = simMaker(tsr_val, yaw_val)    ## simMaker doit pouvoir écrire un fichier .sim qui contient les paramètres de la simulation (on fait varier juste le yaw et le tsr) et retourne un str qui pointe sur le fichier contenant les infos sur la sim 
    qbl.loadSimDefinition(path_sim)         

    for i in range(500) : # Chaque i indique un pas de temps, éventuellement à paramétrer plus proprement

        is_success = qbl.advanceTurbineSImulation()

        if not is_success:  # If success is False, exit the loop
            print(f"Échec de la simulation pour (TSR, yaw) = ({tsr_val, yaw_val}), au {i}-ème pas de temps. Arrêt total.")
            break
