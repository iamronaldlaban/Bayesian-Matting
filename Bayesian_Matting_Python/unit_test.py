

from orchard_bouman_clust import calc_mean, calc_total_weight
from matting_functions import matlab_style_gauss2d


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

        
