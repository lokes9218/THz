"""
Unit tests for SVR and MLP models and metric evaluation.
"""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.svr_model import SVRModel
from src.models.mlp_model import MLPModel
from src.evaluation.metrics import MetricsCalculator

class TestModels(unittest.TestCase):
    
    def test_svr_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = 100.0 + 5.0 * X[:, 0] + np.random.normal(0, 1.0, 100)
        
        model = SVRModel(C=10.0)
        model.fit(X, y)
        preds = model.predict(X[:10])
        
        self.assertEqual(len(preds), 10)
        self.assertTrue(np.all(preds > 0))

    def test_mlp_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = 100.0 + 5.0 * X[:, 0] + np.random.normal(0, 1.0, 100)
        
        model = MLPModel(max_iter=200)
        model.fit(X, y)
        preds = model.predict(X[:10])
        
        self.assertEqual(len(preds), 10)
        self.assertTrue(np.all(preds > 0))

    def test_metrics_calculator(self):
        y_real = np.array([100.0, 120.0, 140.0])
        y_pred = np.array([102.0, 119.0, 138.0])
        
        res = MetricsCalculator.calculate_all_metrics(y_real, y_pred)
        self.assertIn('mae', res)
        self.assertIn('rmse', res)
        self.assertIn('r2', res)
        self.assertIn('mape', res)
        self.assertGreater(res['r2'], 0.9)

if __name__ == '__main__':
    unittest.main()
