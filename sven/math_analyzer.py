import numpy as np
from sven.inductions import nearWakeInduction
from sven.kernels import influenceMatrixNumba

class MathAnalyzer:
    def __init__(self):
        self.active_eta_opt = False 
        self.reset()

    def reset(self):
        self.eta_relax_init_history = []
        self.eta_relax_sol_history = []

    def get_F_matrix(self, blades, deltaFlts):
        """ Calcule la matrice d'influence géométrique """
        n_blades = len(blades)
        n_sec = len(blades[0].centers)
        n_nodes = len(blades[0].bladeNodes)
        total_centers = n_blades * n_sec
        
        centers = np.concatenate([b.centers for b in blades])
        
        leftNodes = np.zeros((0, 3))
        rightNodes = np.zeros((0, 3))
        for blade in blades:
            b_left, b_right, _ = blade.getNodesAndCirculations(True) 
            leftNodes = np.concatenate((leftNodes, b_left))
            rightNodes = np.concatenate((rightNodes, b_right))
            
        infMatrix = influenceMatrixNumba(centers, leftNodes, rightNodes, deltaFlts)
        F_matrix = np.zeros((total_centers, total_centers, 3))
        filaments_per_blade = n_nodes + n_sec + n_sec 
        
        for l in range(total_centers):
            b_idx, s_idx = divmod(l, n_sec)
            offset = b_idx * filaments_per_blade
            
            idx_trail_left  = offset + s_idx
            idx_trail_right = offset + s_idx + 1
            idx_shed        = offset + n_nodes + s_idx
            idx_bound       = offset + n_nodes + n_sec + s_idx
            
            F_matrix[:, l, :] = (infMatrix[:, idx_bound, :] 
                               - infMatrix[:, idx_shed, :] 
                               - infMatrix[:, idx_trail_left, :] 
                               + infMatrix[:, idx_trail_right, :])
        return F_matrix

    def compute_jacobian(self, blades, deltaFlts):
        """ Calcule et renvoie la Jacobienne analytique J = d_Gamma f """
        F_glob = self.get_F_matrix(blades, deltaFlts)
        n_blades = len(blades)
        n_sec = len(blades[0].centers)
        total_n = n_blades * n_sec
        
        c = np.zeros(total_n)
        u_loc = np.zeros((total_n, 3))
        CL = np.zeros(total_n)
        dCL_dalpha = np.zeros(total_n)
        R_matrices = np.zeros((total_n, 3, 3))
        
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
                R_matrices[k] = blade.centersOrientationMatrix[s_idx]
                
        R_T = np.transpose(R_matrices, axes=(0, 2, 1))
        F_local = np.einsum('kij, klj -> kli', R_T, F_glob)
        
        F_x = F_local[:, :, 0] 
        F_z = F_local[:, :, 2] 
        u_x = u_loc[:, 0][:, np.newaxis] 
        u_z = u_loc[:, 2][:, np.newaxis] 
        
        dot_product = (F_x * u_x) + (F_z * u_z)
        cross_product_y = (F_z * u_x) - (F_x * u_z)
        
        term1 = CL[:, np.newaxis] * dot_product
        term2 = dCL_dalpha[:, np.newaxis] * cross_product_y
        
        norm_u = np.linalg.norm(u_loc, axis=1)
        multiplier = (0.5 * c / np.maximum(norm_u, 1e-12))[:, np.newaxis]
        
        J = multiplier * (term1 + term2)
        return J

    def compute_optimal_eta(self, J):
        """ Calcule le meilleur taux de relaxation théorique basé sur le spectre de J """
        eigenvalues = np.linalg.eigvals(J)
        re_lambda = np.real(eigenvalues)
        

        denom = np.abs(eigenvalues - 1.0)**2
        denom = np.where(denom < 1e-14, 1e-14, denom)
        kappas = 2.0 * (1.0 - re_lambda) / denom
        
        if np.all(re_lambda > 1.0):
            return float(np.max(kappas))
        elif np.all(re_lambda < 1.0):
            return float(np.min(kappas))
        else:
            return 0.0

    def evaluate_eta_opt(self, blades, deltaFlts, is_init=True):
        """ Enregistre la valeur optimale de eta si la fonctionnalité est activée """
        if not self.active_eta_opt:
            return
        
        J = self.compute_jacobian(blades, deltaFlts)
        eta_opt = self.compute_optimal_eta(J)
        
        if is_init:
            self.eta_relax_init_history.append(eta_opt)
        else:
            self.eta_relax_sol_history.append(eta_opt)