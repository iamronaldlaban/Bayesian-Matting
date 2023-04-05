import numpy as np
from skimage.metrics import structural_similarity as SSIM

def quality_metrics(image, image_reference):

    mse = np.sum(np.square(image - image_reference))/np.size(image)
    
    sad = np.sum(np.abs(image - image_reference))
    
    psnr = 20 * np.log10( 255 / np.sqrt(mse))

    ssim = SSIM(image, image_reference, data_range=1)

    print("MSE is ",mse)
    print("SAD is ",sad)
    print("PSNR is ",psnr)
    print("SSIM is ", ssim)







