
import os
import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from torchvision.models.feature_extraction import create_feature_extractor
from functools import partial


from fista import *
from admm import * 
from utils import * 



# Initialiser LPIPS (Perceptual loss)
loss_fn_vgg = lpips.LPIPS(net='vgg')

# Charger le modèle InceptionV3 pré-entraîné pour le calcul du FID
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.eval()

# Créer un extracteur de caractéristiques pour obtenir les sorties de la couche 'avgpool'
return_nodes = {'avgpool': 'avgpool'}
feature_extractor = create_feature_extractor(inception_model, return_nodes=return_nodes)

def calculate_fid(image1, image2):
    """
    Calcule une approximation du FID entre deux images individuelles.


    Paramètres:
    - image1: Première image (numpy array)
    - image2: Deuxième image (numpy array)

    Retourne:
    - fid_value: Valeur approximative du FID
    """
    # Prétraitement pour InceptionV3
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Moyennes normalisées pour InceptionV3
                             [0.229, 0.224, 0.225])  # Écarts-types normalisés pour InceptionV3
    ])

    # Transformer les images en tensors
    img1 = preprocess(image1).unsqueeze(0)
    img2 = preprocess(image2).unsqueeze(0)

    # Extraire les caractéristiques
    with torch.no_grad():
        # Obtenir les caractéristiques de la couche 'avgpool'
        features1 = feature_extractor(img1)['avgpool'].squeeze().cpu().numpy()
        features2 = feature_extractor(img2)['avgpool'].squeeze().cpu().numpy()

    # Calculer la moyenne et la variance (pour une seule image, la moyenne est le vecteur lui-même)
    mu1 = features1
    mu2 = features2
    sigma1 = np.var(features1)
    sigma2 = np.var(features2)

    # Calculer le FID approximatif
    diff = mu1 - mu2
    diff_squared = diff.dot(diff)

    # Calcul simplifié avec variances
    fid_value = diff_squared + sigma1 + sigma2 - 2 * np.sqrt(sigma1 * sigma2)

    return fid_value

def process_image(img_input, sampling_factor=2, display=True, slice_index=65, crop_coords=(46,160,3,-3)):
    """
    Traite une image en effectuant le sous-échantillonnage, l'interpolation et calcule les métriques.

    Paramètres:
    - img_input: Chemin de l'image ou tableau numpy de l'image.
    - sampling_factor: Facteur de sous-échantillonnage (par exemple, 2 pour réduire la résolution par 2).
    - display: Booléen pour afficher ou non les images et les différences.
    - slice_index: Indice de la tranche à extraire si l'image est 3D.
    - crop_coords: Coordonnées pour rogner l'image sous la forme (x_start, x_end, y_start, y_end).

    Retourne:
    - metrics: Dictionnaire contenant les métriques PSNR, SSIM, LPIPS et FID pour chaque méthode d'interpolation.
    """
 
    if isinstance(img_input, str):
       
        img = sitk.ReadImage(img_input)
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = img_arr / np.max(img_arr)
       
    else:
        
        img_arr = img_input
        img_arr = img_arr / np.max(img_arr)

  
    if slice_index is not None:
        img = img_arr[:, slice_index, :]
    else:
        img = img_arr

    if crop_coords is not None:
        x_start, x_end, y_start, y_end = crop_coords
        # Ajuster les indices négatifs
        if x_end < 0:
            x_end = img.shape[0] + x_end
        if y_end < 0:
            y_end = img.shape[1] + y_end
        img = img[x_start:x_end, y_start:y_end]

    
    img = np.flipud(img)
    img = img[2:114,2:]


    img = img.astype(np.float32)
    
    


   

  
  


   
    

 
    up_factor = sampling_factor
    img_sampled = SB(img,up_factor)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
 

    img_resized_nn = cv2.resize(img_sampled, None, fx=up_factor, fy=up_factor, interpolation=cv2.INTER_NEAREST)
    img_resized_lin = cv2.resize(img_sampled, None, fx=up_factor, fy=up_factor, interpolation=cv2.INTER_LINEAR)
    img_resized_cub = cv2.resize(img_sampled, None, fx=up_factor, fy=up_factor, interpolation=cv2.INTER_CUBIC)
    img_resized_spline = zoom(img_sampled, up_factor, order=3)
    img_resized_admm = admm(img_sampled,img, up_factor,n_iter = 500,gamma = 0.1,rho = 1).astype(np.float32)

    img_resized_fista = fista_pnp(img_sampled,img, scale=up_factor, gamma= 0.1,n_iter = 300).astype(np.float32)
    img_resized_fista_TV = fista_TV(img_sampled, n_iter = 3000,img_ref = img, scale = up_factor, gamma = 0.1, tv_weight=0.000001).astype(np.float32)
    img_resized_tikhonov = tikhonov(img_sampled,img_ref = img,n_iter = 2000,scale = up_factor,gamma = 0.01,lamb = 0.01).astype(np.float32)
   
   
   
   
    img_resized_nn_rgb = cv2.cvtColor(img_resized_nn, cv2.COLOR_GRAY2RGB)
    img_resized_lin_rgb = cv2.cvtColor(img_resized_lin, cv2.COLOR_GRAY2RGB)
    img_resized_cub_rgb = cv2.cvtColor(img_resized_cub, cv2.COLOR_GRAY2RGB)
    img_resized_spline_rgb = cv2.cvtColor(img_resized_spline, cv2.COLOR_GRAY2RGB)
    img_resized_admm_rgb = cv2.cvtColor(img_resized_admm, cv2.COLOR_GRAY2RGB)
    img_resized_fista_rgb = cv2.cvtColor(img_resized_fista, cv2.COLOR_GRAY2RGB)
    img_resized_fista_TV_rgb = cv2.cvtColor(img_resized_fista_TV, cv2.COLOR_GRAY2RGB)
    img_resized_tikhonov_rgb = cv2.cvtColor(img_resized_tikhonov, cv2.COLOR_GRAY2RGB)

    # Transformation pour LPIPS
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertir l'image en Tensor
        transforms.Lambda(lambda x: 2 * x - 1)  # Normaliser entre [-1, 1]
    ])

    # Calcul des métriques
    metrics = {}

    # Nearest Neighbor
    psnr_nn = cv2.PSNR(img_resized_nn * 255, img * 255)
    ssim_nn = ssim(img_resized_nn, img, data_range=img.max() - img.min())
    lpips_nn = loss_fn_vgg(transform(img_resized_nn_rgb), transform(img_rgb)).item()
    fid_nn = calculate_fid((img_resized_nn_rgb * 255).astype(np.uint8), (img_rgb * 255).astype(np.uint8))
    metrics['Nearest Neighbor'] = {'PSNR': psnr_nn, 'SSIM': ssim_nn, 'LPIPS': lpips_nn, 'FID': fid_nn}

    # Bilinéaire
    psnr_lin = cv2.PSNR(img_resized_lin * 255, img * 255)
    ssim_lin = ssim(img_resized_lin, img, data_range=img.max() - img.min())
    lpips_lin = loss_fn_vgg(transform(img_resized_lin_rgb), transform(img_rgb)).item()
    fid_lin = calculate_fid((img_resized_lin_rgb * 255).astype(np.uint8), (img_rgb * 255).astype(np.uint8))
    metrics['Bilinear'] = {'PSNR': psnr_lin, 'SSIM': ssim_lin, 'LPIPS': lpips_lin, 'FID': fid_lin}

    # Bicubique
    psnr_cub = cv2.PSNR(img_resized_cub * 255, img * 255)
    ssim_cub = ssim(img_resized_cub, img, data_range=img.max() - img.min())
    lpips_cub = loss_fn_vgg(transform(img_resized_cub_rgb), transform(img_rgb)).item()
    fid_cub = calculate_fid((img_resized_cub_rgb * 255).astype(np.uint8), (img_rgb * 255).astype(np.uint8))
    metrics['Bicubic'] = {'PSNR': psnr_cub, 'SSIM': ssim_cub, 'LPIPS': lpips_cub, 'FID': fid_cub}


    #ADMM
    psnr_ADMM = cv2.PSNR(img_resized_admm * 255, img * 255)
    ssim_ADMM = ssim(img_resized_admm, img, data_range=img.max() - img.min())
    lpips_ADMM = loss_fn_vgg(transform(img_resized_admm_rgb), transform(img_rgb)).item()
    fid_ADMM = calculate_fid((img_resized_admm_rgb * 255).astype(np.uint8), (img_rgb * 255).astype(np.uint8))
    metrics['ADMM'] = {'PSNR': psnr_ADMM, 'SSIM': ssim_ADMM, 'LPIPS': lpips_ADMM, 'FID': fid_ADMM}


    #FISTA
    psnr_fista = cv2.PSNR(img_resized_fista* 255, img * 255)
    ssim_fista = ssim(img_resized_fista, img, data_range=img.max() - img.min())
    lpips_fista = loss_fn_vgg(transform(img_resized_fista_rgb), transform(img_rgb)).item()
    fid_fista = calculate_fid((img_resized_fista_rgb * 255).astype(np.uint8), (img_rgb * 255).astype(np.uint8))
    metrics['FISTA'] = {'PSNR': psnr_fista, 'SSIM': ssim_fista, 'LPIPS': lpips_fista, 'FID': fid_fista}

    #FISTA_TV
    psnr_fista_TV = cv2.PSNR(img_resized_fista_TV* 255, img * 255)
    ssim_fista_TV = ssim(img_resized_fista_TV, img, data_range=img.max() - img.min())
    lpips_fista_TV = loss_fn_vgg(transform(img_resized_fista_TV_rgb), transform(img_rgb)).item()
    fid_fista_TV = calculate_fid((img_resized_fista_TV_rgb * 255).astype(np.uint8), (img_rgb * 255).astype(np.uint8))
    metrics['FISTA_TV'] = {'PSNR': psnr_fista_TV, 'SSIM': ssim_fista_TV, 'LPIPS': lpips_fista_TV, 'FID': fid_fista_TV}
    
    #TIKHONOV
    psnr_tikhonov = cv2.PSNR(img_resized_tikhonov* 255, img * 255)
    ssim_tikhonov = ssim(img_resized_tikhonov, img, data_range=img.max() - img.min())
    lpips_tikhonov = loss_fn_vgg(transform(img_resized_tikhonov_rgb), transform(img_rgb)).item()
    fid_tikhonov  = calculate_fid((img_resized_tikhonov_rgb * 255).astype(np.uint8), (img_rgb * 255).astype(np.uint8))
    metrics['FISTA_TIKHONOV'] = {'PSNR': psnr_tikhonov, 'SSIM': ssim_tikhonov, 'LPIPS': lpips_tikhonov, 'FID': fid_tikhonov}


    # Affichage des images si demandé
    if display:
        methods = [('Nearest Neighbor', img_resized_nn), ('Bilinear', img_resized_lin), ('Bicubic', img_resized_cub), ('ADMM', img_resized_admm),("FISTA",img_resized_fista),("FISTA_TV",img_resized_fista_TV),("TIKHONOV",img_resized_tikhonov)]
        num_methods = len(methods)
    
    # Définir la figure globale
        plt.figure(figsize=(30, 5 * num_methods))
    
        for i, (method_name, img_resized) in enumerate(methods):
        # Index de base pour chaque méthode
            base_idx = i * 4
        
        # Image HR
            plt.subplot(num_methods, 4, base_idx + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Image HR {img.shape}")
            plt.axis('off')

        # Reconstruction
            plt.subplot(num_methods, 4, base_idx + 2)
            plt.imshow(img_resized, cmap='gray')
            plt.title(f"Reconstruction {method_name}")
            plt.axis('off')

        # Image LR
            plt.subplot(num_methods, 4, base_idx + 3)
            plt.imshow(img_sampled, cmap='gray')
            plt.title(f"Image LR {img_sampled.shape}")
            plt.axis('off')

        # Différence
            plt.subplot(num_methods, 4, base_idx + 4)
            plt.imshow(img - img_resized, cmap='gray', vmin=-0.1, vmax=0.1)
            plt.title(f"Différence {method_name}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()
    return metrics




cwd = os.getcwd()
img_paths = [cwd + os.sep + 'AbdomenMRCT_0001_0000.nii.gz', cwd + os.sep + 'AbdomenMRCT_0001_0001.nii.gz']

for path in img_paths:
    k = 0
    metrics = process_image(path, sampling_factor=2, display=True)
    if k == 0:
        print(' MRI metrics:')
    if k == 1: 
        print(' CT metrics:')
       

    print(type(metrics))
    for method, values in metrics.items():
        print(f"--- {method} ---")
        print(f"PSNR: {values['PSNR']}")
        print(f"SSIM: {values['SSIM']}")
        print(f"LPIPS: {values['LPIPS']}")
        print(f"FID: {values['FID']}\n")
    k += 1
