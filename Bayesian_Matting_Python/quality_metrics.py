import numpy as np
from skimage.io import imread

def quality_metrics(image, image_reference):

    mse = np.sum(np.square(image - image_reference))/np.size(image)
    
    sad = np.sum(np.abs(image - image_reference))
    
    psnr = 20 * np.log10( 255 / np.sqrt(mse))

    print("MSE is ",mse)
    print("SAD is ",sad)
    print("PSNR is ",psnr)



# image = imread('alpha.png')
# image_reference = imread('gt_training_lowres\GT04.png')[:,:,1]
# quality_metrics(image, image_reference)



