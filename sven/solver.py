import time
import numpy as np
from sven.inductions import *
from sven.math_analyzer import MathAnalyzer
analyzer = MathAnalyzer()

def update(
    blades, uInfty, timeStep, timeSimulation, innerIter, 
    deltaFlts, startTime, iterationVect, algo_type="picard", tol=0.0):

    iterationTime = time.time()
    
    # 1. Initialize all inductions
    for (iBlade, blade) in enumerate(blades):
        blade.inductionsFromWake[:, :] = 0.
        blade.inductionsAtNodes[:, :] = 0.
        blade.wakeNodesInductions[:, :, :] = 0.

    # 2. Update first wake row
    for blade in blades:
        blade.updateFirstWakeRow()
        nearWakeLength = blade.nearWakeLength

    # 3. Inductions on blade
    if (nearWakeLength > 2):
        wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, "blade")

    # 4. Initialize for convergence loop
    for blade in blades:
        blade.updateSheds(blade.gammaBound)
        blade.updateTrails(blade.gammaBound)
   
    # 5. Analyses mathématiques en arrière-plan (si actif)
    if analyzer.active:
        analyzer.run_shadow_convergence(blades, uInfty, deltaFlts, max_iter=innerIter)
        analyzer.extract_eta_prime(blades)

    # =========================================================================
    # BOUCLE DE CONVERGENCE (Intégration de Newton et du Break)
    # =========================================================================
    t_solver_start = time.time()
    bladesGammaBounds = [0.] * len(blades)
    max_err = 0.0
    iters_taken = innerIter # Par défaut, on suppose qu'on fait toutes les itérations

    if algo_type == "picard":
        for i in range(innerIter):
            iters_taken = i + 1 # Mise à jour du nombre d'itérations réelles
            nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
            max_err = 0.0
            
            iBlade = 0
            for (blade, ind) in zip(blades, nearWakeInducedVelocities):
                old_g = blade.gammaBound.copy()
                bladesGammaBounds[iBlade] = blade.estimateGammaBound(uInfty, ind)
                
                err = np.max(np.abs(blade.f_Gamma - old_g))
                max_err = max(max_err, err)

                blade.updateSheds(bladesGammaBounds[iBlade])
                blade.updateTrails(bladesGammaBounds[iBlade])
                blade.gammaBound = bladesGammaBounds[iBlade]
                iBlade += 1
                
            if tol > 0 and max_err < tol:
                break

    elif algo_type == "newton":
        total_n = sum(len(b.centers) for b in blades)
        for i in range(innerIter):
            iters_taken = i + 1 # Mise à jour du nombre d'itérations réelles
            nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
            f_g_list = []
            current_g_list = []
            
            for blade, ind in zip(blades, nearWakeInducedVelocities):
                old_g = blade.gammaBound.copy()
                blade.estimateGammaBound(uInfty, ind)
                blade.gammaBound = old_g 
                blade.newGammaBound = old_g 
                
                f_g_list.append(blade.f_Gamma)
                current_g_list.append(old_g)
                
            F_G = np.concatenate(f_g_list)
            Gamma = np.concatenate(current_g_list)
            
            err_vector = F_G - Gamma
            max_err = np.max(np.abs(err_vector))
            if tol > 0 and max_err < tol:
                break
                
            J, _, _ = analyzer.compute_jacobian_and_K(blades, deltaFlts, compute_K=False)
            A = np.eye(total_n) - J
            try:
                dGamma = np.linalg.solve(A, err_vector)
                new_Gamma = Gamma + dGamma
            except np.linalg.LinAlgError:
                new_Gamma = F_G 
                
            idx = 0
            for ib, blade in enumerate(blades):
                n_sec = len(blade.centers)
                new_g = new_Gamma[idx : idx+n_sec]
                
                blade.gammaBound = new_g
                blade.newGammaBound = new_g.copy()
                blade.updateSheds(new_g)
                blade.updateTrails(new_g)
                bladesGammaBounds[ib] = new_g
                idx += n_sec

    solver_time = time.time() - t_solver_start
    # =========================================================================

    for (iBlade, blade) in enumerate(blades):
        blade.storeOldGammaBound(bladesGammaBounds[iBlade])

    # 6. Inductions on wake
    if (nearWakeLength > 2):
        wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, "wake")

    bladeInductionsOnWake(blades, deltaFlts)
    
    # 7. Advection and Splicing
    if (nearWakeLength > 2):
        for blade in blades:
            blade.advectFilaments(uInfty, timeStep)
            blade.spliceNearWake()
            blade.updateFilamentCirulations()

    iterationVect.append([time.time() - iterationTime, time.time()-startTime])

  
    return max_err, solver_time, iters_taken