import unittest
import numpy as np
from Bayesian_matte import Bayesian_Matte

class TestBayesianMatte(unittest.TestCase):
    
    def test_output_shape(self):
        img = np.ones((100, 100, 3))
        trimap = np.zeros((100, 100))
        alpha = Bayesian_Matte(img, trimap)
        print(alpha.shape)
        self.assertEqual(alpha.shape, (100, 100), 'Output shape is not correct')
    
    def test_output_range(self):
        img = np.ones((100, 100, 3))
        trimap = np.zeros((100, 100))
        alpha = Bayesian_Matte(img, trimap)
        self.assertTrue(np.all(alpha >= 0) and np.all(alpha <= 1))
      
    def test_input_dtype(self):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        trimap = np.zeros((100, 100), dtype=np.uint8)
        alpha = Bayesian_Matte(img, trimap)
        self.assertEqual(img.dtype, np.float)
        self.assertEqual(trimap.dtype, np.float)
        
    def test_minNeighbours(self):
        img = np.ones((100, 100, 3))
        trimap = np.zeros((100, 100))
        alpha = Bayesian_Matte(img, trimap, minNeighbours=0)
        self.assertTrue(np.all(np.isnan(alpha)))
        
    def test_N_odd(self):
        img = np.ones((100, 100, 3))
        trimap = np.zeros((100, 100))
        with self.assertRaises(ValueError):
            alpha = Bayesian_Matte(img, trimap, N=24)
            
    def test_N_greater_than_1(self):
        img = np.ones((100, 100, 3))
        trimap = np.zeros((100, 100))
        with self.assertRaises(ValueError):
            alpha = Bayesian_Matte(img, trimap, N=1)
        
if __name__ == '__main__':
    unittest.main()
