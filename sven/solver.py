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
    # BOUCLE "TRAIN PICARD" AVEC EXPANSION GÉOMÉTRIQUE
    # =========================================================================
    total_n = sum(len(b.centers) for b in blades)
    
    picard_count = 0
    eta_evals = 0
    valid_eta_count = 0
    last_eta_opt = 0.0
    
    absolute_best_err = float('inf')
    absolute_best_gammas = [b.gammaBound.copy() for b in blades]
    best_algo_overall = "Picard"
    best_p_count = 0
    best_n_count = 0

    final_err = float('inf')
    newton_won = False

    # --- DYNAMIQUE DES BLOCS ---
    # current_p_block commence à 5 et doublera en cas d'échec de Newton
    current_p_block = p_block

    while picard_count < max_picard_iters:
        
        # --- LE TRAIN PICARD (Avance sur current_p_block) ---
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
            
            # Mise à jour de la mémoire absolue
            if current_err < absolute_best_err:
                absolute_best_err = current_err
                absolute_best_gammas = [b.gammaBound.copy() for b in blades]
                best_algo_overall = "Picard"
                best_p_count = picard_count
                best_n_count = 0

        # Si Picard atteint la cible tout seul, on s'arrête
        if current_err <= tol:
            final_err = current_err
            break

        if n_block == 0: # cas particlier où on ne veut pas faire de Newton du tout
            continue

        # --- SAUVEGARDE AVANT DIGRESSION ---
        end_picard_gammas = [b.gammaBound.copy() for b in blades]

        # --- L'ÉCLAIREUR NEWTON (Digression) ---
        current_n_count = 0
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

            # Mise à jour de la mémoire absolue
            if newton_err < absolute_best_err:
                absolute_best_err = newton_err
                absolute_best_gammas = [b.gammaBound.copy() for b in blades]
                best_algo_overall = "Newton"
                best_p_count = picard_count
                best_n_count = current_n_count

            if newton_err <= tol:
                newton_won = True
                final_err = newton_err
                break

            # Full Newton
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

        # --- ANALYSE DE LA DIGRESSION ---
        if newton_won:
            break
        else:
            # Newton a échoué. On restaure le train Picard pour continuer la progression.
            for b, g in zip(blades, end_picard_gammas):
                b.gammaBound = g.copy()
                b.newGammaBound = g.copy()
                b.updateSheds(g)
                b.updateTrails(g)
            
            # --- EXPANSION GÉOMÉTRIQUE ---
            # Newton a raté, on double le nombre d'itérations de Picard
            # On le plafonne à 40 pour éviter qu'un bloc ne dévore tout le budget restant d'un coup.
            current_p_block = min(current_p_block * 2, 40)

            if p_block == 0: # cas particulier où on ne fait jamais de Picard du tout
                stop_reason = "Newton Diverged"
                break

    # --- SÉCURITÉ DE FIN DE BOUCLE (Restauration de l'Argmin) ---
    if not newton_won and current_err > tol:
        for b, g in zip(blades, absolute_best_gammas):
            b.gammaBound = g.copy()
            b.newGammaBound = g.copy()
            b.updateSheds(g)
            b.updateTrails(g)
        final_err = absolute_best_err

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

    return final_err, solver_time, best_p_count, best_n_count, best_algo_overall, last_eta_opt, eta_evals, valid_eta_count