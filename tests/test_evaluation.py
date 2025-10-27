import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    calculate_metrics,
    get_top_n_features,
    calculate_cumulative_importance,
    features_for_threshold
)


class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        self.y_true = np.array([10, 20, 30, 40, 50])
        self.y_pred = np.array([12, 18, 32, 38, 52])
        
        self.importance_df = pd.DataFrame({
            'feature': ['feat_A', 'feat_B', 'feat_C', 'feat_D', 'feat_E'],
            'importance': [0.50, 0.30, 0.10, 0.07, 0.03]
        })
    
    def test_calculate_metrics(self):
        metrics = calculate_metrics(self.y_true, self.y_pred)
        
        self.assertTrue('rmse' in metrics)
        self.assertTrue('mae' in metrics)
        self.assertTrue('r2' in metrics)
        
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertLessEqual(metrics['r2'], 1.0)
    
    def test_calculate_metrics_perfect_prediction(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertEqual(metrics['rmse'], 0)
        self.assertEqual(metrics['mae'], 0)
        self.assertEqual(metrics['r2'], 1.0)
    
    def test_get_top_n_features(self):
        top_3 = get_top_n_features(self.importance_df, n=3)
        
        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3[0], 'feat_A')
        self.assertEqual(top_3[1], 'feat_B')
        self.assertEqual(top_3[2], 'feat_C')
    
    def test_calculate_cumulative_importance(self):
        cumulative_df = calculate_cumulative_importance(self.importance_df)
        
        self.assertTrue('cumulative_importance' in cumulative_df.columns)
        
        self.assertAlmostEqual(cumulative_df.loc[0, 'cumulative_importance'], 0.50)
        self.assertAlmostEqual(cumulative_df.loc[1, 'cumulative_importance'], 0.80)
        self.assertAlmostEqual(cumulative_df.loc[4, 'cumulative_importance'], 1.00)
    
    def test_features_for_threshold(self):
        n_features_90 = features_for_threshold(self.importance_df, threshold=0.90)
        
        self.assertEqual(n_features_90, 3)
        
        n_features_50 = features_for_threshold(self.importance_df, threshold=0.50)
        
        self.assertEqual(n_features_50, 1)
    
    def test_metrics_with_negative_values(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([50, 60, 70])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertLess(metrics['r2'], 0)
    
    def test_metrics_shapes_match(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        
        with self.assertRaises(ValueError):
            calculate_metrics(y_true, y_pred)


class TestFeatureImportance(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        
        self.feature_names = [f'feature_{i}' for i in range(10)]
        self.importances = np.random.random(10)
        self.importances = self.importances / self.importances.sum()
        
        self.importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    def test_importance_sums_to_one(self):
        total_importance = self.importance_df['importance'].sum()
        self.assertAlmostEqual(total_importance, 1.0, places=5)
    
    def test_top_features_ordering(self):
        top_5 = get_top_n_features(self.importance_df, n=5)
        
        self.assertEqual(len(top_5), 5)
        
        importances = [self.importance_df[self.importance_df['feature'] == f]['importance'].values[0] 
                      for f in top_5]
        
        self.assertEqual(importances, sorted(importances, reverse=True))
    
    def test_cumulative_increases(self):
        cumulative_df = calculate_cumulative_importance(self.importance_df)
        
        cumulative_values = cumulative_df['cumulative_importance'].values
        
        for i in range(1, len(cumulative_values)):
            self.assertGreaterEqual(cumulative_values[i], cumulative_values[i-1])


if __name__ == '__main__':
    unittest.main()
