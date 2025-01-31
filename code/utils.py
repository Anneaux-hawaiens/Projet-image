import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage
import cv2
from tqdm import tqdm
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_nl_means, estimate_sigma
import bm3d

from utils import*
from NLM import * 

import SimpleITK as sitk






def gradient(u):
    m, n = u.shape
    grad_u = np.zeros((2, m, n))
    
    grad_u[0, :-1, :] = u[1:] - u[:-1]
    
    grad_u[1, :, :-1] = u[:, 1:] - u[:, :-1]
    
    return grad_u



def div(p):
    m, n = p.shape[1:]
    
    div_1 = np.zeros((m, n))
    div_1[:-1, :] = p[0, :-1, :]
    div_1[1:, :] -= p[0, :-1, :]
    
    div_2 = np.zeros((m, n))
    div_2[:, :-1] = p[1, :, :-1]
    div_2[:, 1:] -= p[1, :, :-1]
    
    return div_1 + div_2


def laplacian(u):
    return (div(gradient(u)))



def prox_tv(x, weight):
    
    return denoise_tv_chambolle(x,weight=weight)



def S_down(img,scale):
    return img[::scale,::scale]





def gaussian_kernel(size, sigma):
    """Create a Gaussian kernel with the given size and sigma."""
    size = int(size) // 2
    x, y = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1))
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) * normal


    g = g / np.sum(g)
    
    return g


def B(input_image,sigma  = 1.5):
    output = gaussian_filter(input = input_image, sigma=sigma, mode='mirror')
    return output



def uniform_blurring(input_image, kernal_size ):

    """
    Applique un flou uniforme 

    Args:
        input_image (ndarray) : Image d'entrée à flouter.
        kernal_size (int) : Taille du noyau 

    Retourne:
        ndarray : Image floutée.

    """
   
    normalization_factor = (kernal_size)**2

   
    kernal = np.ones((kernal_size,kernal_size))/normalization_factor

    output = signal.convolve2d(input_image, kernal, mode='same',\
                                boundary='symm')

    return output


def S_up(img, scale):
    """
    Effectue un suréchantillonnage d'une image 

    Args:
        img (ndarray) : Image d'entrée à suréchantillonner.
        scale (int) : Facteur d'échantillonage

    Retourne:
        ndarray : Image suréchantillonnée

    """
    n,m = img.shape
    result = np.zeros((scale*n,scale*m))
    result[::scale,::scale] = img
    return result


def SB(img,scale):
    """
    Applique l'operateur de dégradation SB

    Args:
        img (ndarray) : Image d'entrée 
        scale (int) : Facteur  d'échantillonnagee.

    Retourne:
        ndarray : Image transformée

    """
    result = B(img)
    result = S_down(result,scale)
    return result



def SB_adjoint(img,scale):
    """
    Applique l'operateur de dégradation SB adjoint

    Args:
        img (ndarray) : Image d'entrée 
        scale (int) : Facteur  d'échantillonnagee.

    Retourne:
        ndarray : Image transformée

    """
    
    result = S_up(img,scale)
    result = B(result)
    return result



def BM3D(x):
    " Débruiteur BM3D"
    sigma_est = np.mean(estimate_sigma(x, channel_axis=None))
    return bm3d.bm3d(x, sigma_psd=sigma_est, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)



def gradient_descent(z,x,img_observed,scale,gamma,rho,n_iter):

    for i in range(n_iter):
        z = x - gamma * SB_adjoint(SB(x,scale)-img_observed,scale) + rho*(z - x)
        return z


