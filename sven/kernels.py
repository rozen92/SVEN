from numba import jit, njit, prange
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

@jit(nopython=True, fastmath=True)
def biotSavartFilaments(evaluationPoints, leftNodes, rightNodes, circulations, 
                        deltaFlts):
    """
    Computes induced velocities using the Biot-Savart law on CPU
    """
    inducedVelocities = np.zeros((len(evaluationPoints),3))
    fd = np.zeros(len(circulations))
    for k in range(len(circulations)):
        fd[k] = np.linalg.norm(rightNodes[k] - leftNodes[k])
    fd = fd * deltaFlts
    for i in range(len(evaluationPoints)):
        evaluationPoint = evaluationPoints[i]
        curr_px = evaluationPoint[0];
        curr_py = evaluationPoint[1];
        curr_pz = evaluationPoint[2];
        numFilaments = len(circulations)
        for k in range(numFilaments):
            curr_fp1x = leftNodes[k][0]
            curr_fp1y = leftNodes[k][1]
            curr_fp1z = leftNodes[k][2]
            curr_fp2x = rightNodes[k][0]
            curr_fp2y = rightNodes[k][1]
            curr_fp2z = rightNodes[k][2]
            pxx1 = curr_px - curr_fp1x;
            pyy1 = curr_py - curr_fp1y;
            pzz1 = curr_pz - curr_fp1z;
            pxx2 = curr_px - curr_fp2x;
            pyy2 = curr_py - curr_fp2y;
            pzz2 = curr_pz - curr_fp2z;
            r1 = np.sqrt((pxx1 * pxx1 + pyy1 * pyy1 + pzz1 * pzz1));
            r2 = np.sqrt((pxx2 * pxx2 + pyy2 * pyy2 + pzz2 * pzz2));
            r1dr2 = pxx1 * pxx2 + pyy1 * pyy2 + pzz1 * pzz2;
            r1tr2 = r1 * r2;
            den = (r1tr2 * (r1tr2 + r1dr2) + fd[k] * fd[k]);
            ubar = circulations[k] * (r1 + r2) / den;
            inducedVelocities[i,0] += ubar * (pyy1 * pzz2 - pzz1 * pyy2);
            inducedVelocities[i,1] += ubar * (pzz1 * pxx2 - pxx1 * pzz2);
            inducedVelocities[i,2] += ubar * (pxx1 * pyy2 - pyy1 * pxx2);
    return inducedVelocities / (4. * np.pi)

# This kernel computes induced velocities on GPU
modFlts = SourceModule("""
__global__ void inducedVelocityKernel(float *destUx, float *destUy, 
                      float *destUz, float *ptclePosX, float *ptclePosY, 
                      float *ptclePosZ, float *fltsLeftX, float *fltsLeftY, 
                      float *fltsLeftZ, float *fltsRightX, float *fltsRightY, 
                      float *fltsRightZ, float *fltsCirculations, 
                      float *lengthRegul, int numParticles, int numFilaments, 
                      float deltaFlts)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx<numParticles){
    float curr_px  , curr_py  , curr_pz;
    curr_px = ptclePosX[idx];
    curr_py = ptclePosY[idx];
    curr_pz = ptclePosZ[idx];
    for(int k=0; k < numFilaments; k++)
    {
        float  pxx1  = curr_px  - fltsLeftX[k];
        float  pyy1  = curr_py  - fltsLeftY[k];
        float  pzz1  = curr_pz  - fltsLeftZ[k];
        float  pxx2  = curr_px - fltsRightX[k];
        float  pyy2  = curr_py - fltsRightY[k];
        float  pzz2  = curr_pz - fltsRightZ[k];
        float    r1  = sqrt((pxx1*pxx1 + pyy1*pyy1 + pzz1*pzz1));
        float    r2  = sqrt((pxx2*pxx2 + pyy2*pyy2 + pzz2*pzz2));
        float   den  = r1*r2*(r1*r2 + pxx1*pxx2 + pyy1*pyy2 + pzz1*pzz2) + lengthRegul[k];
        float ubar = 0.;
        if(abs(den) > 1e-32){
          ubar  = fltsCirculations[k] *(r1 + r2) / den;
        }
        destUx[idx] += ubar * (pyy1*pzz2-pzz1*pyy2);
        destUy[idx] += ubar * (pzz1*pxx2-pxx1*pzz2);
        destUz[idx] += ubar * (pxx1*pyy2-pyy1*pxx2);
    }
  }
}
""")

# =========================================================================
# NOYAU NUMBA CPU (POUR LA JACOBIENNE NEWTON)
# =========================================================================
@njit(parallel=True, fastmath=True)
def influenceMatrixNumba(evaluationPoints, leftNodes, rightNodes, deltaFlts):
    """
    Calcule la matrice d'influence géométrique brute (Gamma=1) 
    entre N points et M filaments.
    Retourne une matrice de taille (N, M, 3).
    """
    numPoints = len(evaluationPoints)
    numFilaments = len(leftNodes)
    
    infMatrix = np.zeros((numPoints, numFilaments, 3))
    
    fd = np.zeros(numFilaments)
    for j in range(numFilaments):
        vec = rightNodes[j] - leftNodes[j]
        fd[j] = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2) * deltaFlts
        
    for i in prange(numPoints):
        curr_px = evaluationPoints[i, 0]
        curr_py = evaluationPoints[i, 1]
        curr_pz = evaluationPoints[i, 2]
        
        for j in range(numFilaments):
            pxx1 = curr_px - leftNodes[j, 0]
            pyy1 = curr_py - leftNodes[j, 1]
            pzz1 = curr_pz - leftNodes[j, 2]
            
            pxx2 = curr_px - rightNodes[j, 0]
            pyy2 = curr_py - rightNodes[j, 1]
            pzz2 = curr_pz - rightNodes[j, 2]
            
            r1 = np.sqrt(pxx1*pxx1 + pyy1*pyy1 + pzz1*pzz1)
            r2 = np.sqrt(pxx2*pxx2 + pyy2*pyy2 + pzz2*pzz2)
            
            r1dr2 = pxx1*pxx2 + pyy1*pyy2 + pzz1*pzz2
            r1tr2 = r1 * r2
            
            den = (r1tr2 * (r1tr2 + r1dr2) + fd[j] * fd[j])
            
            ubar = 0.0
            if abs(den) > 1e-32:
                ubar = (r1 + r2) / den
                
            infMatrix[i, j, 0] = ubar * (pyy1*pzz2 - pzz1*pyy2) / (4.0 * np.pi)
            infMatrix[i, j, 1] = ubar * (pzz1*pxx2 - pxx1*pzz2) / (4.0 * np.pi)
            infMatrix[i, j, 2] = ubar * (pxx1*pyy2 - pyy1*pxx2) / (4.0 * np.pi)
            
    return infMatrix