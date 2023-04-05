import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
from matting_functions import compositing
from Bayesian_matte_OB import Bayesian_Matte
from quality_metrics import quality_metrics
from timeit import default_timer as timer
import datetime
import unittest
from orchard_bouman_clust import calc_mean, calc_total_weight
from matting_functions import matlab_style_gauss2d



# root = tk.Tk()
# root.withdraw() # Hide the main window

# image = filedialog.askopenfilename()
# image_trimap = filedialog.askopenfilename()

# Read the image, trimap and ground truth
filename = 'GT10'
image = np.array(Image.open(f'input_training_lowres\{filename}.png'))
image_trimap = np.array(ImageOps.grayscale(Image.open(f'trimap_training_lowres\Trimap2\{filename}.png')))
ground_truth = np.array(ImageOps.grayscale(Image.open(f'gt_training_lowres/{filename}.png')))
# Window Size
N = 105

start = timer() # Start the timer

# Run the Bayesian Matting algorithm
alpha_OB= Bayesian_Matte(image, image_trimap, N)
end = timer() # End the timer
alpha_OB = alpha_OB*255 # Convert to 8-bit image

# Calculate the quality metrics
quality_metrics(alpha_OB, ground_truth)
print('Time taken: ', datetime.timedelta(seconds = end - start))

# Read the background image
background = cv2.imread('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/background.jpg')
background2 = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

# Compositing
comp_OB = compositing(image, alpha_OB, background2)

# Create a subplot of input image, trimap, alpha matte, and composite image
fig, ax = plt.subplots(1, 4, figsize=(20,5))
ax[0].imshow(image)
ax[0].set_title('Input Image')
ax[0].axis('off')
ax[1].imshow(image_trimap, cmap='gray')
ax[1].set_title('Trimap')
ax[1].axis('off')
ax[2].imshow(alpha_OB, cmap='gray')
ax[2].set_title('Alpha Matte')
ax[2].axis('off')
ax[3].imshow(comp_OB)
ax[3].set_title('Composite Image')
ax[3].axis('off')

plt.show()

# unit test

class TestBayesianMatte(unittest.TestCase):
    def test_image_and_alpha_size(self):
              
        # Load the images
        try:
            input_image = image
            image_alpha = alpha_OB
        except IOError:
            self.fail("Failed to load images")

        # Check if the images have the same dimensions
        try:
            self.assertEqual(input_image.shape[0:2], image_alpha.shape)
        except AssertionError:
            self.fail("Image and alpha have different dimensions")
        
        # If everything is proper, assert that the test passes successfully
        self.assertTrue(True, "The images have the same dimensions")

    def test_window_size_is_odd(self):
        self.assertNotEqual(N % 2, 0, "N is not odd")

    def test_images(self):
        # Load example images
        input_image = image
        trimap_image = image_trimap
        ground_truth_image = ground_truth
        # Check if images are not None
        assert input_image is not None, "Input image is not loaded"
        assert trimap_image is not None, "Trimap image is not loaded"
        assert ground_truth_image is not None, "Ground truth image is not loaded"
        # Check if images are valid
        assert input_image.shape[:2] == trimap_image.shape[:2] == ground_truth_image.shape[:2], "Images have different sizes"

    def setUp(self):
        # Initialize test data
        self.weights = np.array([1.0, 2.0, 3.0])
        self.data = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])
        self.total_weight = np.sum(self.weights)

    def test_calc_total_weight(self):
        # Test calc_total_weight function
        self.assertEqual(calc_total_weight(self.weights), 6.0)

    def test_calc_mean(self):
        # Test calc_mean function
        expected_mean = np.array([5.0, 6.0, 7.0])
        self.assertTrue(np.allclose(calc_mean(self.data, self.weights, self.total_weight), expected_mean))

    def test_output_shape(self):
        sigma = 0.5
        shape = (5, 5)
        filter = matlab_style_gauss2d(shape=shape, sigma=sigma)
        self.assertEqual(filter.shape, shape)

    def test_sum_is_one(self):
        sigma = 1.0
        shape = (7, 7)
        filter = matlab_style_gauss2d(shape=shape, sigma=sigma)
        self.assertAlmostEqual(np.sum(filter), 1.0, places=6)

    def test_symmetry(self):
        sigma = 2.0
        shape = (11, 11)
        filter = matlab_style_gauss2d(shape=shape, sigma=sigma)
        flipped_filter = np.flip(filter)
        self.assertTrue(np.allclose(filter, flipped_filter))

   
if __name__ == '__main__':
    unittest.main()


# image_alpha = np.array(ImageOps.grayscale(Image.open('C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/gt_training_lowres/GT10.png')))
# write_path = 'C:/Users/labanr/Desktop/Matting/Bayesian-Matting/Bayesian_Matting_Python/alpha.png'
# cv2.imwrite(write_path, alpha_OB)
#image = np.array(Image.open('High_Resolution/input_training_highres/GT04.png'))
#image_trimap = np.array(ImageOps.grayscale(Image.open('High_Resolution/trimap_training_highres/Trimap1/GT04.png')))
# image_alpha = np.array(ImageOps.grayscale(Image.open('High_Resolution/gt_training_highres/GT04.png')))