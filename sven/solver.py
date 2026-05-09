import time
import numpy as np
from sven.inductions import *
from sven.math_analyzer import MathAnalyzer

# Instance globale de l'analyseur pour les calculs de Jacobienne
analyzer = MathAnalyzer()

def update(
    blades, uInfty, timeStep, timeSimulation, max_total_iters, 
    deltaFlts, startTime, iterationVect, algo_type="hybrid", tol=1e-6, 
    p_block=5, n_block=3, current_relax=0.3):
    """
    Met à jour l'état de la turbine en utilisant une stratégie de cycles adaptatifs.
    Alterne entre des blocs de Picard (stabilité) et de Newton (précision).
    """
    iterationTime = time.time()
    t_solver_start = time.time()
    
    # 1. Initialisation des inductions et du sillage
    for blade in blades:
        blade.inductionsFromWake[:, :] = 0.
        blade.inductionsAtNodes[:, :] = 0.
        blade.wakeNodesInductions[:, :, :] = 0.
        blade.updateFirstWakeRow()

    nearWakeLength = blades[0].nearWakeLength

    # 3. Calcul des inductions sur les pales
    if nearWakeLength > 2:
        wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, "blade")

    # 4. Préparation de la boucle de convergence
    for blade in blades:
        blade.updateSheds(blade.gammaBound)
        blade.updateTrails(blade.gammaBound)
   
    # =========================================================================
    # BOUCLE ADAPTATIVE PAR CYCLES (nP + mN)
    # =========================================================================
    total_n = sum(len(b.centers) for b in blades)
    best_err = float('inf')
    best_gammas = [b.gammaBound.copy() for b in blades]
    
    total_iters = 0
    eta_evals = 0
    valid_eta_count = 0
    last_eta_opt = 0.0
    best_it = 0
    
    prev_cycle_err = float('inf')
    stop_reason = "MaxIters"

    # La boucle continue tant que le budget d'itérations n'est pas épuisé
    # et que la tolérance n'est pas atteinte.
    while total_iters < max_total_iters and best_err > tol:
        
        # --- SOUS-CYCLE PICARD (Recherche de stabilité) ---
        for _ in range(p_block):
            if total_iters >= max_total_iters or best_err <= tol:
                break
            total_iters += 1
            
            nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
            max_err = 0.0
            for blade, ind in zip(blades, nearWakeInducedVelocities):
                old_g = blade.gammaBound.copy()
                f_g = blade.compute_f_Gamma(uInfty, ind)
                new_g = blade.apply_Picard_relaxation(f_g, custom_relax=current_relax)
                max_err = max(max_err, np.max(np.abs(f_g - old_g)))
                blade.updateSheds(new_g)
                blade.updateTrails(new_g)
            
            if max_err < best_err:
                best_err, best_it = max_err, total_iters
                best_gammas = [b.gammaBound.copy() for b in blades]

        # --- SOUS-CYCLE NEWTON (Accélération de la précision) ---
        if best_err > tol:
            for _ in range(n_block):
                if total_iters >= max_total_iters or best_err <= tol:
                    break
                total_iters += 1
                
                nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
                f_g_list, current_g_list = [], []
                
                for blade, ind in zip(blades, nearWakeInducedVelocities):
                    f_g_list.append(blade.compute_f_Gamma(uInfty, ind))
                    current_g_list.append(blade.gammaBound.copy())
                
                F_G = np.concatenate(f_g_list)
                Gamma = np.concatenate(current_g_list)
                err_vector = F_G - Gamma
                max_err = np.max(np.abs(err_vector))

                if max_err < best_err:
                    best_err, best_it = max_err, total_iters
                    best_gammas = [b.gammaBound.copy() for b in blades]

                # Full Newton : Jacobienne et Eta calculés à chaque itération du bloc
                J = analyzer.compute_jacobian(blades, deltaFlts)
                last_eta_opt = analyzer.compute_optimal_eta(J)
                eta_evals += 1
                if last_eta_opt > 0:
                    valid_eta_count += 1
                
                # Résolution du système linéaire
                A = np.eye(total_n) - J
                try:
                    dGamma = np.linalg.solve(A, err_vector)
                    new_Gamma = Gamma + dGamma
                except np.linalg.LinAlgError:
                    new_Gamma = F_G 
                
                # Mise à jour des circulations sur les pales
                idx = 0
                for blade in blades:
                    n_sec = len(blade.centers)
                    ng = new_Gamma[idx : idx+n_sec]
                    blade.gammaBound = ng
                    blade.newGammaBound = ng.copy()
                    blade.updateSheds(ng)
                    blade.updateTrails(ng)
                    idx += n_sec

        # --- TEST DE STAGNATION (Fin du cycle combiné) ---
        # Si l'amélioration de l'erreur sur le cycle est inférieure à 2%, on arrête.
        if best_err >= prev_cycle_err * 0.98:
            stop_reason = "Stagnation"
            break
        prev_cycle_err = best_err

    if best_err <= tol:
        stop_reason = "Converged"

    # Diagnostic de "Rebond" : l'erreur minimale a été trouvée avant la fin
    early_argmin = (best_it < total_iters) and (best_err > tol)

    # Restauration finale de la meilleure solution rencontrée (Argmin)
    for b, g in zip(blades, best_gammas):
        b.gammaBound = g.copy()
        b.newGammaBound = g.copy()
        b.updateSheds(g)
        b.updateTrails(g)

    # 6. Mise à jour finale du sillage et advection
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

    # Enregistrement du temps CPU
    iterationVect.append([time.time() - iterationTime, time.time() - startTime])

    # Détermination de l'algorithme vainqueur pour les logs
    # Si Newton a été lancé au moins une fois, on le considère comme l'algo actif
    win_algo = "Newton" if total_iters > p_block else "Picard"

    return best_err, solver_time, total_iters, win_algo, last_eta_opt, early_argmin, best_it, eta_evals, valid_eta_count, stop_reason