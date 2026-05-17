import time
import numpy as np
from sven.inductions import *
from sven.math_analyzer import MathAnalyzer

analyzer = MathAnalyzer()

def update(
    blades, uInfty, timeStep, timeSimulation, max_picard_iters, 
    deltaFlts, startTime, iterationVect, algo_type="hybrid", tol=1e-6, 
    p_block=5, n_block=5, current_relax=0.3):

    iterationTime = time.time()
    t_solver_start = time.time()
    
    # 1. Initialisation des inductions et sillage
    for blade in blades:
        blade.inductionsFromWake[:, :] = 0.
        blade.inductionsAtNodes[:, :] = 0.
        blade.wakeNodesInductions[:, :, :] = 0.
        blade.updateFirstWakeRow()

    nearWakeLength = blades[0].nearWakeLength

    if nearWakeLength > 2:
        wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, "blade")

    for blade in blades:
        blade.updateSheds(blade.gammaBound)
        blade.updateTrails(blade.gammaBound)
   
    # =========================================================================
    # BOUCLE HYBRIDE AVEC ARBITRAGE (N-ème Picard vs Argmin Newton)
    # =========================================================================
    total_n = sum(len(b.centers) for b in blades)
    
    picard_count = 0
    eta_evals = 0
    valid_eta_count = 0
    
    best_algo_overall = "Picard"
    best_p_count = 0
    best_n_count = 0

    final_err = float('inf')
    newton_won = False

    current_p_block = p_block

    # Initialisation pour le tout premier pas
    current_err = float('inf')

    while picard_count < max_picard_iters:
        
        # --- PICARD ---
        for _ in range(current_p_block):
            if picard_count >= max_picard_iters: break
            picard_count += 1
            
            nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
            current_err = 0.0
            
            for blade, ind in zip(blades, nearWakeInducedVelocities):
                old_g = blade.gammaBound.copy()
                f_g = blade.compute_f_Gamma(uInfty, ind)
                new_g = blade.apply_Picard_relaxation(f_g, custom_relax=current_relax)
                current_err = max(current_err, np.max(np.abs(f_g - old_g)))
                blade.updateSheds(new_g)
                blade.updateTrails(new_g)
            
            best_algo_overall = "Picard"
            best_p_count = picard_count
            best_n_count = 0

        # Si Picard atteint la cible tout seul
        if current_err <= tol:
            final_err = current_err
            break

        # Mode Picard Pur
        if n_block == 0:
            continue

        # --- SAUVEGARDE DU N-ème PICARD ---
        picard_final_err = current_err
        picard_final_gammas = [b.gammaBound.copy() for b in blades]

        # --- L'ÉCLAIREUR NEWTON (Avec Argmin Local) ---
        current_n_count = 0
        best_newton_err = float('inf')
        best_newton_gammas = None
        best_newton_iter = 0

        for _ in range(n_block):
            current_n_count += 1
            nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
            f_g_list, current_g_list = [], []
            
            for blade, ind in zip(blades, nearWakeInducedVelocities):
                f_g_list.append(blade.compute_f_Gamma(uInfty, ind))
                current_g_list.append(blade.gammaBound.copy())
            
            F_G = np.concatenate(f_g_list)
            Gamma = np.concatenate(current_g_list)
            err_vector = F_G - Gamma
            newton_err = np.max(np.abs(err_vector))

            # ARGMIN LOCAL À NEWTON
            if newton_err < best_newton_err:
                best_newton_err = newton_err
                best_newton_gammas = [b.gammaBound.copy() for b in blades]
                best_newton_iter = current_n_count

            if newton_err <= tol:
                newton_won = True
                final_err = newton_err
                best_algo_overall = "Newton"
                best_p_count = picard_count
                best_n_count = current_n_count
                break

            J = analyzer.compute_jacobian(blades, deltaFlts)
            last_eta_opt = analyzer.compute_optimal_eta(J)
            eta_evals += 1
            if last_eta_opt > 0: valid_eta_count += 1
            
            A = np.eye(total_n) - J
            try:
                new_Gamma = Gamma + np.linalg.solve(A, err_vector)
            except np.linalg.LinAlgError:
                new_Gamma = F_G 
            
            idx = 0
            for blade in blades:
                n_sec = len(blade.centers)
                ng = new_Gamma[idx : idx+n_sec]
                blade.gammaBound = ng
                blade.newGammaBound = ng.copy()
                blade.updateSheds(ng)
                blade.updateTrails(ng)
                idx += n_sec

        # --- ARBITRAGE DU POINT DE DÉPART POUR LE CYCLE SUIVANT ---
        if newton_won:
            break
        else:
            # Newton a échoué. On compare le meilleur Newton avec le dernier Picard.
            if best_newton_err < picard_final_err:
                # Le saut de Newton était meilleur : on part de là
                for b, g in zip(blades, best_newton_gammas):
                    b.gammaBound = g.copy()
                    b.newGammaBound = g.copy()
                    b.updateSheds(g)
                    b.updateTrails(g)
                current_err = best_newton_err
                best_algo_overall = "Newton (Argmin)"
                best_n_count = best_newton_iter
            else:
                # Newton a tout empiré : on repart du N-ème Picard
                for b, g in zip(blades, picard_final_gammas):
                    b.gammaBound = g.copy()
                    b.newGammaBound = g.copy()
                    b.updateSheds(g)
                    b.updateTrails(g)
                current_err = picard_final_err
                best_algo_overall = "Picard"
                best_n_count = 0
            
            # Expansion géométrique de Picard pour le prochain cycle
            current_p_block = min(current_p_block * 2, 800)

            # Mode Newton Pur
            if p_block == 0:
                break

    if not newton_won:
        # Les pales sont déjà restaurées sur la meilleure configuration grâce à l'arbitrage
        final_err = current_err

    # =========================================================================
    # ANALYSE DE STABILITÉ DU POINT FIXE
    # =========================================================================
    J_final = analyzer.compute_jacobian(blades, deltaFlts)
    final_eta_opt = analyzer.compute_optimal_eta(J_final)
    eta_evals += 1
    if final_eta_opt > 0:
        valid_eta_count += 1
    # =========================================================================

    solver_time = time.time() - t_solver_start
    for iB, blade in enumerate(blades):
        blade.storeOldGammaBound([b.gammaBound for b in blades][iB])

    if blades[0].nearWakeLength > 2:
        wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, "wake")
    bladeInductionsOnWake(blades, deltaFlts)
    
    for blade in blades:
        blade.advectFilaments(uInfty, timeStep)
        blade.spliceNearWake()
        blade.updateFilamentCirulations()

    iterationVect.append([time.time() - iterationTime, time.time() - startTime])

    return final_err, solver_time, best_p_count, best_n_count, best_algo_overall, final_eta_opt, eta_evals, valid_eta_count