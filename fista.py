import scipy.ndimage as ndimage
import scipy.signal as signal
from skimage.util import random_noise
from skimage import data, img_as_float
#for convolution on numpy arrays
import numpy as np
import time
#import #we will use scipy.ndimage.gaussian_filter for gaussian blurring
from scipy.ndimage import gaussian_filter
#import kernal
from scipy.signal import convolve2d
#gaussian kernal
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm
import cv2
import bm3d
from skimage.metrics import peak_signal_noise_ratio as psnr
from NLM import * 
from utils import * 




def fista_pnp(img_obs,img_ref,scale,gamma,n_iter):

    """
    Effectue la restauration d'image en utilisant l'algorithme FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) 
    avec NLM.

    Args:
        img_obs (ndarray) : L'image observée .
        img_ref (ndarray) : L'image de référence .
        scale (float) : Le facteur d'echantillonnage.
        gamma (float) : pas
        n_iter (int) : Le nombre d'itérations .

    Retourne:
        ndarray : L'image restaurée 

    """
    
    x  = cv2.resize(img_obs, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    x0 =  x.copy()
    y = x.copy()
    t = 1
    error_values = []
    for i in tqdm(range(n_iter)):
        sigma_est = np.mean(estimate_sigma(x, channel_axis=None))
        x1 = DSG_NLM((y - gamma * SB_adjoint(SB(y,scale)-img_obs,scale)),x0,3,10,sigma_est)
        t1 = 1/2 * (1 + np.sqrt(1+4*t**2))
        y = x1 + (t -1)/(t1)*(x1 - x)
        x = x1
        t = t1
        current_error = np.linalg.norm((img_ref - x)**2,2)
        error_values.append(current_error)
        
    plt.plot(range(n_iter), error_values)
    plt.xlabel('Iterations')
    plt.ylabel('error')
    plt.title('error vs interations (FISTA PNP )')
    plt.show()
    return x





def fista_TV(y, n_iter,img_ref, scale, gamma, tv_weight=0.0003):
    """
    Effectue la restauration d'image FISTA  avec une régularisation TV

    Args:
        y (ndarray) : L'image observée
        n_iter (int) : Le nombre d'itérations 
        img_ref (ndarray) : L'image de référence 
        scale (float) : Le facteur d'echantillonnage
        gamma (float) : pas 
        tv_weight (float, optionnel) : Poids de la régularisaiton

    Retourne:
        ndarray : L'image restaurée 

    """
  
    n, m = y.shape
    x = cv2.resize(y, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    z = x.copy()
    t = 1 
    print('FISTA_TV:')
    error_values = []
    for i in tqdm(range(n_iter)):
      
        grad = SB_adjoint(SB(z, scale) - y, scale)
        
    
        x_new = prox_tv(z - gamma * grad, tv_weight)
        
        
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + (t - 1) / t_new * (x_new - x)
        
        
        x = x_new
        t = t_new
        current_error = np.linalg.norm((img_ref - x)**2,2)
        error_values.append(current_error)
        
    plt.plot(range(n_iter), error_values)
    plt.xlabel('Iterations')
    plt.ylabel('error')
    plt.title('error vs interations (TV)')
    plt.show()
    
    return x




def tikhonov(img_obs,img_ref,n_iter,scale,gamma,lamb):

    """
    Effectue une descente de gradient avec régularisation tikhonov

    Args:
        img_obs (ndarray) : L'image observée
        img_ref (ndarray) : L'image de référence 
        n_iter (int) : Le nombre d'itérations 
        scale (float) : Le facteur d'echantillonnage
        gamma (float) : pas 
        lamb (float) : poids régularisaiton
        

    Retourne:
        ndarray : L'image restaurée 

    """


    n,m = img_obs.shape
    x = np.zeros((scale*n,scale*m))
    
    z = x.copy()

    print('TIKHONOV:')
    error_values = []
    for i in tqdm(range(n_iter)):
        grad = SB_adjoint(SB(z,scale)-img_obs,scale) - 2*lamb * laplacian(z)
        z = z -  gamma*grad
    
        current_error = np.linalg.norm((img_ref - x)**2,2)
        error_values.append(current_error)
        
    plt.plot(range(n_iter), error_values)
    plt.xlabel('Iterations')
    plt.ylabel('error')
    plt.title('error vs interations (TIKHONOV)')
    plt.show()
    
        
    return(z)
    
    


