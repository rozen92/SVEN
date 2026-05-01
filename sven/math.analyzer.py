import numpy as np
from sven.inductions import nearWakeInduction

class MathAnalyzer:
    def __init__(self):
        # Ratio de vitesse transverse
        self.eta_primes_history = []
        
        # Métriques de la matrice K
        self.spectral_radii_K = []
        self.K_infinity_norms = []
        
        # Constante de Lipschitz empirique
        self.lipschitz_empirical = []
        
        # Historique des résidus
        self.picard_residuals = []
        self.picard_relax_residuals = []
        self.newton_residuals = []

    def gather_gamma(self, blades):
        """Concatène les circulations Gamma en un seul vecteur de taille 3n"""
        return np.concatenate([b.gammaBound for b in blades])

    def set_gamma(self, blades, gamma_array):
        """Dispatche un vecteur global de taille 3n vers les pales respectives"""
        n_sec = len(blades[0].centers)
        for i, b in enumerate(blades):
            b.gammaBound = gamma_array[i * n_sec : (i + 1) * n_sec]

    def extract_eta_prime(self, blades):
        """Enregistre le max du ratio eta' sur toutes les pales"""
        max_eta = np.max([np.max(b.eta_prime) for b in blades])
        self.eta_primes_history.append(max_eta)

    def get_F_matrix(self, blades, deltaFlts):
        """
        Calcule la matrice géométrique F_l(P_k)
        Retourne un tenseur de dimension (3n, 3n, 3).
        """
        n_blades = len(blades)
        n_sec = len(blades[0].centers)
        total_n = n_blades * n_sec
        F_matrix = np.zeros((total_n, total_n, 3))
        
        # 1. Sauvegarde de l'état actuel des circulations (y compris newGammaBound)
        orig_states = [(np.copy(b.newGammaBound), np.copy(b.gammaShed), np.copy(b.gammaTrail)) for b in blades]
        
        for l in range(total_n):
            b_idx, s_idx = divmod(l, n_sec)
            
            # Mise à zéro de tout le domaine
            for b in blades:
                b.newGammaBound[:] = 0.
                b.gammaShed[:] = 0.
                b.gammaTrail[:] = 0.
                
            # (Gamma_l = 1)
            blades[b_idx].newGammaBound[s_idx] = 1.0
            blades[b_idx].gammaShed[s_idx] = -1.0
            
            ghost = np.zeros(n_sec + 2)
            ghost[1:-1] = blades[b_idx].newGammaBound
            blades[b_idx].gammaTrail[:] = -(ghost[1:] - ghost[:-1])
            
            # Calcul des inductions (Biot-Savart linéaire)
            inductions = nearWakeInduction(blades, deltaFlts)
            flat_ind = np.concatenate(inductions, axis=0) # Taille (3n, 3)
            
            # La colonne l de la matrice F correspond à ces inductions
            F_matrix[:, l, :] = flat_ind
            
        # 2. Restauration de l'état
        for i, b in enumerate(blades):
            b.newGammaBound[:], b.gammaShed[:], b.gammaTrail[:] = orig_states[i]
            
        return F_matrix

    def compute_jacobian_and_K(self, blades, deltaFlts):
        """
        Calcule la Jacobienne de f et la matrice K(0).
        """
        F_glob = self.get_F_matrix(blades, deltaFlts)
        
        n_blades = len(blades)
        n_sec = len(blades[0].centers)
        total_n = n_blades * n_sec
        
        J = np.zeros((total_n, total_n))
        K_matrix = np.zeros((total_n, total_n))
        
        # Vecteurs 1D
        c = np.zeros(total_n)
        u_loc = np.zeros((total_n, 3))
        CL = np.zeros(total_n)
        dCL_dalpha = np.zeros(total_n)
        C_k = np.zeros(total_n)
        C_k_prime = np.zeros(total_n)
        
        # Construction des grandeurs locales
        for b_idx, blade in enumerate(blades):
            for s_idx in range(n_sec):
                k = b_idx * n_sec + s_idx
                c[k] = blade.centerChords[s_idx]
                
                V = blade.effectiveVelocity[s_idx]
                alpha = blade.attackAngle[s_idx]
                
                # Vitesse 2D projetée (hypothèse SVEN)
                u_loc[k, 0] = V * np.cos(alpha)
                u_loc[k, 1] = 0.0
                u_loc[k, 2] = V * np.sin(alpha)
                
                airfoil = blade.airfoils[s_idx]
                CL[k] = airfoil.getLift(alpha)
                dCL_dalpha[k] = airfoil.getLiftDerivative(alpha)
                
                # Coefficients maximaux pour la matrice K
                max_CL = np.max(np.abs(airfoil.Lifts))
                max_dCL = np.max(np.abs([airfoil.getLiftDerivative(a) for a in airfoil.AOAs]))
                
                C_k[k] = 0.5 * c[k] * max_CL
                C_k_prime[k] = 0.5 * c[k] * max_dCL

        # Remplissage des matrices J et K
        for k in range(total_n):
            b_idx, s_idx = divmod(k, n_sec)
            R_k = blades[b_idx].centersOrientationMatrix[s_idx] # Passage Global -> Local
            norm_u = np.linalg.norm(u_loc[k])
            
            for l in range(total_n):
                # Projection de F_glob dans le repère local
                F_l = R_k.T @ F_glob[k, l, :]
                F_l[1] = 0.0 # Hypothèse 2D
                
                if norm_u > 1e-12:
                    term1 = CL[k] * np.dot(F_l, u_loc[k])
                    term2 = dCL_dalpha[k] * (F_l[0] * u_loc[k, 2] - F_l[2] * u_loc[k, 0]) # F_l x u_loc projeté sur Y
                    J[k, l] = (c[k] / (2 * norm_u)) * (term1 + term2)
                
                norm_F = np.linalg.norm(F_l)
                K_matrix[k, l] = (C_k[k] + C_k_prime[k]) * norm_F
                
        return J, K_matrix

    def run_shadow_convergence(self, blades, uInfty, deltaFlts, max_iter=20):
        """
        Lance les résolutions Picard pur et Newton en arrière-plan
        pour enregistrer les résidus sans perturber le solveur SVEN.
        """
        orig_gamma = self.gather_gamma(blades)
        total_n = len(orig_gamma)
        
        # --- Fonction utilitaire pour évaluer f(Gamma) ---
        def evaluate_f(current_gamma):
            self.set_gamma(blades, current_gamma)
            for b in blades:
                b.updateSheds(b.gammaBound) # Utilise oldGammaBound intact !
                b.updateTrails(b.gammaBound)
            
            ind = nearWakeInduction(blades, deltaFlts)
            f_g = []
            for i, b in enumerate(blades):
                b.estimateGammaBound(uInfty, ind[i])
                f_g.append(b.f_Gamma) # f_Gamma brut (sans relaxation)
            return np.concatenate(f_g)

        # ==========================================
        # 1. Méthode de Picard pure (sans relaxation)
        # ==========================================
        gamma_pic = orig_gamma.copy()
        pic_res = []
        L_empirics = []
        
        for _ in range(max_iter):
            f_g = evaluate_f(gamma_pic)
            res = np.linalg.norm(f_g - gamma_pic)
            pic_res.append(res)
            
            # Constante de Lipschitz empirique
            if len(pic_res) > 1 and pic_res[-2] > 1e-12:
                L_empirics.append(pic_res[-1] / pic_res[-2])
                
            gamma_pic = f_g.copy()
            
        self.picard_residuals.append(pic_res)
        if L_empirics:
            self.lipschitz_empirical.append(np.max(L_empirics))
        else:
            self.lipschitz_empirical.append(0.0)

        # ==========================================
        # 2. Méthode de Newton
        # ==========================================
        gamma_newton = orig_gamma.copy()
        newton_res = []
        
        for _ in range(max_iter):
            f_g = evaluate_f(gamma_newton)
            res = np.linalg.norm(f_g - gamma_newton)
            newton_res.append(res)
            
            if res < 1e-6:
                break
                
            # Calcul de la Jacobienne au point courant
            J, K = self.compute_jacobian_and_K(blades, deltaFlts)
            
            # Résolution (I - J) * dGamma = f_g - gamma
            A = np.eye(total_n) - J
            B = f_g - gamma_newton
            try:
                dGamma = np.linalg.solve(A, B)
                gamma_newton = gamma_newton + dGamma
            except np.linalg.LinAlgError:
                break # Matrice singulière
                
        self.newton_residuals.append(newton_res)
        
        # Enregistrement des propriétés de K(0) au point initial
        _, K_initial = self.compute_jacobian_and_K(blades, deltaFlts)
        self.spectral_radii_K.append(np.max(np.abs(np.linalg.eigvals(K_initial))))
        self.K_infinity_norms.append(np.linalg.norm(K_initial, ord=np.inf))

        # --- Restauration de l'état SVEN ---
        self.set_gamma(blades, orig_gamma)
        for b in blades:
            b.updateSheds(b.gammaBound)
            b.updateTrails(b.gammaBound)