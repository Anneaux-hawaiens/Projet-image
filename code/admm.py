from utils import * 

from NLM import * 
from scipy.ndimage.filters import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma

from functools import partial
from NLM import * 


def admm(img_obs,img_ref,scale,n_iter,gamma,rho):

    """
    Effectue la restauration ADMM avec denoiser (NLM).

    Args:
        img_obs (ndarray) : Image observée
        img_ref (ndarray) : Image de référence
        scale (float) : Facteur d'echantillonnage
        n_iter (int) : Nombre d'itérations 
        gamma (float) : pas
    
        
    Retourne:
        ndarray : Image restaurée après l'application de l'algorithme ADMM.

    """

    

    x = cv2.resize(img_obs, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    z = x.copy()
    x0 = x.copy()

    u = np.zeros(x.shape)
    psnr_values = [np.linalg.norm((img_ref - x)**2,2)]
    for i in tqdm(range(n_iter)):
        sigma_est = np.mean(estimate_sigma(x, channel_axis=None))
        z = gradient_descent(z,x - u,img_obs,2,gamma,rho,500)
        x = DSG_NLM(z+u,x0,3,10,sigma_est)
        u = u +(z - x)
        current_psnr = np.linalg.norm((img_ref - x)**2,2)
        psnr_values.append(current_psnr)
    
    plt.plot(range(n_iter+1), psnr_values)
    plt.xlabel('Iterations')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR en fonction des itérations (ADMM PNP)')
    plt.show()

    return(x)
   

