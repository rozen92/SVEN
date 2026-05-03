import numpy as np
from sven.inductions import nearWakeInduction

class MathAnalyzer:
    def __init__(self):
        self.active = False 
        self.reset()

    def reset(self):
        """Réinitialise les historiques pour relancer une nouvelle simulation"""
        self.active = False
        self.eta_primes_history = []
        
        # Rayons spectraux et normes
        self.spectral_radii_K = []
        self.K_infinity_norms = []
        self.spectral_radii_K_prime_init = [] 
        self.spectral_radii_K_prime_sol = []  
        
        # Constantes de Lipschitz empiriques
        self.lipschitz_empirical = []
        self.lipschitz_empirical_relax = []   
        
        # Résidus
        self.picard_residuals = []
        self.picard_relax_residuals = []
        self.newton_residuals = []
        
        # Valeurs propres et conditionnements
        self.jacobian_eigenvalues_init = []   
        self.jacobian_eigenvalues_sol = []    
        self.jacobian_condition_numbers_init = [] # Renommé pour clarté
        self.jacobian_condition_numbers_sol = []  # NOUVEAU
        
        # Erreur de validation de la Jacobienne par Différences Finies
        self.fd_verification_errors = []

    def gather_gamma(self, blades):
        return np.concatenate([b.gammaBound for b in blades])

    def set_gamma(self, blades, gamma_array):
        n_sec = len(blades[0].centers)
        for i, b in enumerate(blades):
            gammas = gamma_array[i * n_sec : (i + 1) * n_sec]
            b.gammaBound = gammas.copy()
            b.newGammaBound = gammas.copy() 

    def extract_eta_prime(self, blades):
        if not self.active: return
        max_eta = np.max([np.max(b.eta_prime) for b in blades])
        self.eta_primes_history.append(max_eta)

    def get_F_matrix(self, blades, deltaFlts):
        n_blades = len(blades)
        n_sec = len(blades[0].centers)
        total_n = n_blades * n_sec
        F_matrix = np.zeros((total_n, total_n, 3))
        
        orig_states = [(np.copy(b.newGammaBound), np.copy(b.gammaShed), np.copy(b.gammaTrail)) for b in blades]
        
        for l in range(total_n):
            b_idx, s_idx = divmod(l, n_sec)
            for b in blades:
                b.newGammaBound[:] = 0.
                b.gammaShed[:] = 0.
                b.gammaTrail[:] = 0.
                
            blades[b_idx].newGammaBound[s_idx] = 1.0
            blades[b_idx].gammaShed[s_idx] = -1.0
            
            ghost = np.zeros(n_sec + 2)
            ghost[1:-1] = blades[b_idx].newGammaBound
            blades[b_idx].gammaTrail[:] = -(ghost[1:] - ghost[:-1])
            
            inductions = nearWakeInduction(blades, deltaFlts)
            flat_ind = np.concatenate(inductions, axis=0)
            
            F_matrix[:, l, :] = flat_ind
            
        for i, b in enumerate(blades):
            b.newGammaBound[:], b.gammaShed[:], b.gammaTrail[:] = orig_states[i]
            
        return F_matrix

    def compute_jacobian_and_K(self, blades, deltaFlts):
        F_glob = self.get_F_matrix(blades, deltaFlts)
        n_blades = len(blades)
        n_sec = len(blades[0].centers)
        total_n = n_blades * n_sec
        
        J = np.zeros((total_n, total_n))
        K_matrix = np.zeros((total_n, total_n))
        K_prime_matrix = np.zeros((total_n, total_n)) 
        
        c = np.zeros(total_n)
        u_loc = np.zeros((total_n, 3))
        CL = np.zeros(total_n)
        dCL_dalpha = np.zeros(total_n)
        
        C_k = np.zeros(total_n)
        C_k_prime = np.zeros(total_n)
        C_k_local = np.zeros(total_n)       
        C_k_prime_local = np.zeros(total_n) 
        
        for b_idx, blade in enumerate(blades):
            for s_idx in range(n_sec):
                k = b_idx * n_sec + s_idx
                c[k] = blade.centerChords[s_idx]
                V = blade.effectiveVelocity[s_idx]
                alpha = blade.attackAngle[s_idx]
                
                u_loc[k, 0] = V * np.cos(alpha)
                u_loc[k, 1] = 0.0
                u_loc[k, 2] = V * np.sin(alpha)
                
                airfoil = blade.airfoils[s_idx]
                CL[k] = airfoil.getLift(alpha)
                dCL_dalpha[k] = airfoil.getLiftDerivative(alpha)
                
                max_CL = np.max(np.abs(airfoil.Lifts))
                max_dCL = np.max(np.abs([airfoil.getLiftDerivative(a) for a in airfoil.AOAs]))
                C_k[k] = 0.5 * c[k] * max_CL
                C_k_prime[k] = 0.5 * c[k] * max_dCL
                
                C_k_local[k] = 0.5 * c[k] * np.abs(CL[k])
                C_k_prime_local[k] = 0.5 * c[k] * np.abs(dCL_dalpha[k])

        for k in range(total_n):
            b_idx, s_idx = divmod(k, n_sec)
            R_k = blades[b_idx].centersOrientationMatrix[s_idx]
            norm_u = np.linalg.norm(u_loc[k])
            
            for l in range(total_n):
                F_l = R_k.T @ F_glob[k, l, :]
                F_l[1] = 0.0
                
                if norm_u > 1e-12:
                    term1 = CL[k] * np.dot(F_l, u_loc[k])
                    term2 = dCL_dalpha[k] * (F_l[2] * u_loc[k, 0] - F_l[0] * u_loc[k, 2])
                    J[k, l] = (c[k] / (2 * norm_u)) * (term1 + term2)
                
                norm_F = np.linalg.norm(F_l)
                K_matrix[k, l] = (C_k[k] + C_k_prime[k]) * norm_F
                K_prime_matrix[k, l] = (C_k_local[k] + C_k_prime_local[k]) * norm_F
                
        return J, K_matrix, K_prime_matrix

    def verify_jacobian_fd(self, evaluate_func, current_gamma, J_analytic, epsilon=1e-3):
        """Vérifie la Jacobienne via Différences Finies sur une direction aléatoire (adapté float32)"""
        d = np.random.randn(len(current_gamma))
        d = d / np.linalg.norm(d) 
        
        f_plus = evaluate_func(current_gamma + epsilon * d)
        f_minus = evaluate_func(current_gamma - epsilon * d)
        
        fd_derivative = (f_plus - f_minus) / (2.0 * epsilon)
        analytic_derivative = J_analytic @ d
        
        error = np.linalg.norm(fd_derivative - analytic_derivative) / (np.linalg.norm(analytic_derivative) + 1e-16)
        return error

    def run_shadow_convergence(self, blades, uInfty, deltaFlts, max_iter=20):
        if not self.active: return
        
        orig_gamma = self.gather_gamma(blades)
        total_n = len(orig_gamma)
        relax_factor = blades[0].relax
        
        def evaluate_f(current_gamma):
            self.set_gamma(blades, current_gamma)
            for b in blades:
                b.updateSheds(b.gammaBound)
                b.updateTrails(b.gammaBound)
            
            ind = nearWakeInduction(blades, deltaFlts)
            f_g = []
            for i, b in enumerate(blades):
                b.estimateGammaBound(uInfty, ind[i])
                f_g.append(b.f_Gamma)
            return np.concatenate(f_g)

        # 1. Picard Pur
        gamma_pic = orig_gamma.copy()
        pic_res = []
        L_empirics = []
        for _ in range(max_iter):
            f_g = evaluate_f(gamma_pic)
            res = np.linalg.norm(f_g - gamma_pic)
            pic_res.append(res)
            if len(pic_res) > 1 and pic_res[-2] > 1e-12:
                L_empirics.append(pic_res[-1] / pic_res[-2])
            gamma_pic = f_g.copy()
            
        self.picard_residuals.append(pic_res)
        self.lipschitz_empirical.append(np.max(L_empirics) if L_empirics else 0.0)

        # 2. Picard avec Relaxation
        gamma_relax = orig_gamma.copy()
        relax_res = []
        L_empirics_relax = []
        for _ in range(max_iter):
            f_g = evaluate_f(gamma_relax)
            next_gamma = gamma_relax + relax_factor * (f_g - gamma_relax)
            res = np.linalg.norm(next_gamma - gamma_relax)
            relax_res.append(res)
            if len(relax_res) > 1 and relax_res[-2] > 1e-12:
                L_empirics_relax.append(relax_res[-1] / relax_res[-2])
            gamma_relax = next_gamma.copy()
            
        self.picard_relax_residuals.append(relax_res)
        self.lipschitz_empirical_relax.append(np.max(L_empirics_relax) if L_empirics_relax else 0.0)

        # 3. Newton (Tolérance modifiée à 1e-10)
        gamma_newton = orig_gamma.copy()
        newton_res = []
        for _ in range(max_iter):
            f_g = evaluate_f(gamma_newton)
            res = np.linalg.norm(f_g - gamma_newton)
            newton_res.append(res)
            
            if res < 1e-10 or np.isnan(res) or np.isinf(res):
                break
                
            J, _, _ = self.compute_jacobian_and_K(blades, deltaFlts)
            A = np.eye(total_n) - J
            B = f_g - gamma_newton
            try:
                dGamma = np.linalg.solve(A, B)
                gamma_newton = gamma_newton + dGamma
            except np.linalg.LinAlgError:
                break
        self.newton_residuals.append(newton_res)
        
        # 4. Évaluations des Matrices au point INITIAL (Gamma_init)
        _ = evaluate_f(orig_gamma) 
        J_init, K_init, K_prime_init = self.compute_jacobian_and_K(blades, deltaFlts)
        
        self.spectral_radii_K.append(np.max(np.abs(np.linalg.eigvals(K_init))))
        self.K_infinity_norms.append(np.linalg.norm(K_init, ord=np.inf))
        self.spectral_radii_K_prime_init.append(np.max(np.abs(np.linalg.eigvals(K_prime_init))))
        self.jacobian_eigenvalues_init.append(np.linalg.eigvals(J_init))
        self.jacobian_condition_numbers_init.append(np.linalg.cond(np.eye(total_n) - J_init))
        
        # Validation par différences finies (sur Gamma_init avec la vraie Jacobienne J_init)
        fd_err = self.verify_jacobian_fd(evaluate_f, orig_gamma, J_init)
        self.fd_verification_errors.append(fd_err)
        
        # 5. Évaluations des Matrices au point FINAL / SOLUTION (Gamma_sol issu de Picard Relaxé)
        _ = evaluate_f(gamma_relax) 
        J_sol, _, K_prime_sol = self.compute_jacobian_and_K(blades, deltaFlts)
        
        self.spectral_radii_K_prime_sol.append(np.max(np.abs(np.linalg.eigvals(K_prime_sol))))
        self.jacobian_eigenvalues_sol.append(np.linalg.eigvals(J_sol))
        self.jacobian_condition_numbers_sol.append(np.linalg.cond(np.eye(total_n) - J_sol)) # NOUVEAU

        # Restauration finale avant de rendre la main au solveur
        self.set_gamma(blades, orig_gamma)
        for b in blades:
            b.updateSheds(b.gammaBound)
            b.updateTrails(b.gammaBound)