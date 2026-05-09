import time
import numpy as np
from sven.inductions import *
from sven.math_analyzer import MathAnalyzer

analyzer = MathAnalyzer()

def update(
    blades, uInfty, timeStep, timeSimulation, innerIter, 
    deltaFlts, startTime, iterationVect, algo_type="hybrid", tol=0.0, 
    picard_iters=10, current_relax=0.3):

    t_solver_start = time.time()
    
    # 1. Initialize all inductions
    for blade in blades:
        blade.inductionsFromWake[:, :] = 0.
        blade.inductionsAtNodes[:, :] = 0.
        blade.wakeNodesInductions[:, :, :] = 0.
        blade.updateFirstWakeRow()

    nearWakeLength = blades[0].nearWakeLength

    # 3. Inductions on blade
    if nearWakeLength > 2:
        wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, "blade")

    # 4. Initialize for convergence loop
    for blade in blades:
        blade.updateSheds(blade.gammaBound)
        blade.updateTrails(blade.gammaBound)
   
    # =========================================================================
    # BOUCLE DE CONVERGENCE (Mode Full-Newton Hybrid Intra-Step)
    # =========================================================================
    bladesGammaBounds = [0.] * len(blades)
    total_n = sum(len(b.centers) for b in blades)
    
    best_err = float('inf')
    best_gammas = None
    best_algo = ""
    last_eta_opt = 0.0
    
    picard_count = 0
    newton_count = 0
    eta_evals = 0
    best_iter = 0
    
    # --- PHASE 1 : Pré-conditionnement Picard ---
    for i in range(picard_iters):
        picard_count += 1 
        nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
        max_err = 0.0
        
        for blade, ind in zip(blades, nearWakeInducedVelocities):
            old_g = blade.gammaBound.copy()
            f_g = blade.compute_f_Gamma(uInfty, ind)
            new_g = blade.apply_Picard_relaxation(f_g, custom_relax=current_relax)
            
            err = np.max(np.abs(f_g - old_g))
            max_err = max(max_err, err)

            blade.updateSheds(new_g)
            blade.updateTrails(new_g)
            
        # Mise à jour de l'Argmin
        if max_err < best_err:
            best_err = max_err
            best_gammas = [b.gammaBound.copy() for b in blades]
            best_algo = "Picard"
            best_iter = picard_count
            
        if tol > 0 and max_err < tol:
            break
            
    # --- PHASE 2 : Affinage Full Newton ---
    if best_err > tol and picard_count < innerIter:
        # On restaure la meilleure solution de Picard comme point de départ
        for b, g in zip(blades, best_gammas):
            b.gammaBound = g.copy()
            b.newGammaBound = g.copy()
            b.updateSheds(g)
            b.updateTrails(g)

        for i in range(innerIter - picard_iters):
            newton_count += 1
            nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
            f_g_list = []
            current_g_list = []
            
            for blade, ind in zip(blades, nearWakeInducedVelocities):
                f_g = blade.compute_f_Gamma(uInfty, ind)
                f_g_list.append(f_g)
                current_g_list.append(blade.gammaBound.copy())
                
            F_G = np.concatenate(f_g_list)
            Gamma = np.concatenate(current_g_list)
            
            err_vector = F_G - Gamma
            max_err = np.max(np.abs(err_vector))
            
            # Mise à jour de l'Argmin
            if max_err < best_err:
                best_err = max_err
                best_gammas = [b.gammaBound.copy() for b in blades]
                best_algo = "Newton"
                best_iter = picard_count + newton_count
                
            if tol > 0 and max_err < tol:
                break
                
            # FULL NEWTON : Calcul de la Jacobienne à CHAQUE itération
            J = analyzer.compute_jacobian(blades, deltaFlts)
            last_eta_opt = analyzer.compute_optimal_eta(J)
            eta_evals += 1
            
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
                idx += n_sec

    # --- DIAGNOSTIC ARGMIN (Rebond) ---
    total_iters_done = picard_count + newton_count
    early_argmin = False
    if best_err > tol and best_iter < total_iters_done:
        early_argmin = True

    # --- RESTAURATION ARGMIN FINAL ---
    for ib, (b, g) in enumerate(zip(blades, best_gammas)):
        b.gammaBound = g.copy()
        b.newGammaBound = g.copy()
        b.updateSheds(g)
        b.updateTrails(g)
        bladesGammaBounds[ib] = g.copy()
        
    # =========================================================================

    solver_time = time.time() - t_solver_start

    for iBlade, blade in enumerate(blades):
        blade.storeOldGammaBound(bladesGammaBounds[iBlade])

    # 6. Inductions on wake
    if nearWakeLength > 2:
        wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, "wake")

    bladeInductionsOnWake(blades, deltaFlts)
    
    # 7. Advection and Splicing
    if nearWakeLength > 2:
        for blade in blades:
            blade.advectFilaments(uInfty, timeStep)
            blade.spliceNearWake()
            blade.updateFilamentCirulations()

    iterationVect.append([time.time() - iterationTime, time.time() - startTime])

    return best_err, solver_time, picard_count, newton_count, best_algo, last_eta_opt, early_argmin, best_iter, eta_evals