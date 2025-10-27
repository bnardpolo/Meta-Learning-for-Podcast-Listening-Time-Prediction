import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline import train_baseline_model, evaluate_baseline_model
from src.models.meta_learning import PrototypicalNetwork
from src.models.improved_prototypical import ImprovedPrototypicalNetwork


class TestBaselineModel(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        
        self.train_tasks = []
        for i in range(3):
            n_samples = 100
            X = np.random.randn(n_samples, 10)
            y = np.random.randn(n_samples) * 10 + 50
            
            self.train_tasks.append({
                'podcast_name': f'Podcast_{i}',
                'X': X,
                'y': y,
                'n_episodes': n_samples
            })
        
        self.test_tasks = []
        for i in range(2):
            n_samples = 50
            X = np.random.randn(n_samples, 10)
            y = np.random.randn(n_samples) * 10 + 50
            
            self.test_tasks.append({
                'podcast_name': f'TestPodcast_{i}',
                'X': X,
                'y': y,
                'n_episodes': n_samples
            })
    
    def test_train_baseline_model(self):
        model = train_baseline_model(self.train_tasks)
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
    
    def test_evaluate_baseline_model(self):
        model = train_baseline_model(self.train_tasks)
        results = evaluate_baseline_model(model, self.test_tasks)
        
        self.assertTrue('rmse' in results)
        self.assertTrue('mae' in results)
        self.assertTrue('r2' in results)
        
        self.assertGreater(results['rmse'], 0)
        self.assertGreater(results['mae'], 0)
    
    def test_baseline_predictions_shape(self):
        model = train_baseline_model(self.train_tasks)
        results = evaluate_baseline_model(model, self.test_tasks)
        
        expected_predictions = sum(task['n_episodes'] for task in self.test_tasks)
        actual_predictions = len(results['predictions'])
        
        self.assertEqual(actual_predictions, expected_predictions)


class TestPrototypicalNetwork(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        
        self.feature_cols = [f'feature_{i}' for i in range(10)]
        self.top_features = self.feature_cols[:5]
        
        self.train_tasks = []
        for i in range(3):
            n_samples = 100
            X = np.random.randn(n_samples, 10)
            y = np.random.randn(n_samples) * 10 + 50
            
            self.train_tasks.append({
                'podcast_name': f'Podcast_{i}',
                'X': X,
                'y': y,
                'n_episodes': n_samples
            })
        
        self.test_task = {
            'podcast_name': 'TestPodcast',
            'X': np.random.randn(50, 10),
            'y': np.random.randn(50) * 10 + 50,
            'n_episodes': 50
        }
    
    def test_init(self):
        proto_net = PrototypicalNetwork(feature_names=self.top_features)
        
        self.assertIsNotNone(proto_net)
        self.assertEqual(proto_net.feature_names, self.top_features)
    
    def test_fit(self):
        proto_net = PrototypicalNetwork(feature_names=self.top_features)
        proto_net.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        self.assertIsNotNone(proto_net.scaler)
        self.assertTrue(len(proto_net.prototypes) > 0)
    
    def test_predict(self):
        proto_net = PrototypicalNetwork(feature_names=self.top_features)
        proto_net.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        predictions = proto_net.predict(self.test_task, self.feature_cols, k_shot=5)
        
        self.assertEqual(len(predictions), self.test_task['n_episodes'] - 5)
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_predict_different_k_shots(self):
        proto_net = PrototypicalNetwork(feature_names=self.top_features)
        proto_net.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        pred_1shot = proto_net.predict(self.test_task, self.feature_cols, k_shot=1)
        pred_10shot = proto_net.predict(self.test_task, self.feature_cols, k_shot=10)
        
        self.assertEqual(len(pred_1shot), self.test_task['n_episodes'] - 1)
        self.assertEqual(len(pred_10shot), self.test_task['n_episodes'] - 10)


class TestImprovedPrototypicalNetwork(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        
        self.feature_cols = [f'feature_{i}' for i in range(10)]
        self.top_features = self.feature_cols[:5]
        
        self.train_tasks = []
        for i in range(3):
            n_samples = 100
            X = np.random.randn(n_samples, 10)
            y = np.random.randn(n_samples) * 10 + 50
            
            self.train_tasks.append({
                'podcast_name': f'Podcast_{i}',
                'X': X,
                'y': y,
                'n_episodes': n_samples
            })
        
        self.test_task = {
            'podcast_name': 'TestPodcast',
            'X': np.random.randn(50, 10),
            'y': np.random.randn(50) * 10 + 50,
            'n_episodes': 50
        }
    
    def test_init(self):
        improved_proto = ImprovedPrototypicalNetwork(feature_names=self.top_features)
        
        self.assertIsNotNone(improved_proto)
        self.assertEqual(improved_proto.feature_names, self.top_features)
        self.assertIsNotNone(improved_proto.global_model)
    
    def test_fit(self):
        improved_proto = ImprovedPrototypicalNetwork(feature_names=self.top_features)
        improved_proto.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        self.assertIsNotNone(improved_proto.scaler)
        self.assertIsNotNone(improved_proto.global_model)
    
    def test_predict(self):
        improved_proto = ImprovedPrototypicalNetwork(feature_names=self.top_features)
        improved_proto.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        predictions = improved_proto.predict(self.test_task, self.feature_cols, k_shot=5)
        
        self.assertEqual(len(predictions), self.test_task['n_episodes'] - 5)
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_predict_adaptation(self):
        improved_proto = ImprovedPrototypicalNetwork(feature_names=self.top_features)
        improved_proto.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        predictions = improved_proto.predict(self.test_task, self.feature_cols, k_shot=5)
        
        self.assertTrue(np.all(predictions >= 0))
    
    def test_predict_different_k_shots(self):
        improved_proto = ImprovedPrototypicalNetwork(feature_names=self.top_features)
        improved_proto.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        pred_3shot = improved_proto.predict(self.test_task, self.feature_cols, k_shot=3)
        pred_10shot = improved_proto.predict(self.test_task, self.feature_cols, k_shot=10)
        
        self.assertEqual(len(pred_3shot), self.test_task['n_episodes'] - 3)
        self.assertEqual(len(pred_10shot), self.test_task['n_episodes'] - 10)


class TestModelComparison(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        
        self.feature_cols = [f'feature_{i}' for i in range(10)]
        self.top_features = self.feature_cols[:5]
        
        self.train_tasks = []
        for i in range(3):
            n_samples = 100
            X = np.random.randn(n_samples, 10)
            y = np.random.randn(n_samples) * 10 + 50
            
            self.train_tasks.append({
                'podcast_name': f'Podcast_{i}',
                'X': X,
                'y': y,
                'n_episodes': n_samples
            })
        
        self.test_tasks = []
        for i in range(2):
            n_samples = 50
            X = np.random.randn(n_samples, 10)
            y = np.random.randn(n_samples) * 10 + 50
            
            self.test_tasks.append({
                'podcast_name': f'TestPodcast_{i}',
                'X': X,
                'y': y,
                'n_episodes': n_samples
            })
    
    def test_baseline_vs_prototypical(self):
        baseline_model = train_baseline_model(self.train_tasks)
        baseline_results = evaluate_baseline_model(baseline_model, self.test_tasks)
        
        proto_net = PrototypicalNetwork(feature_names=self.top_features)
        proto_net.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        self.assertIsNotNone(baseline_results)
        self.assertIsNotNone(proto_net)
    
    def test_prototypical_vs_improved(self):
        proto_net = PrototypicalNetwork(feature_names=self.top_features)
        proto_net.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        improved_proto = ImprovedPrototypicalNetwork(feature_names=self.top_features)
        improved_proto.fit(self.train_tasks, self.feature_cols, k_shot=5)
        
        test_task = self.test_tasks[0]
        
        pred_basic = proto_net.predict(test_task, self.feature_cols, k_shot=5)
        pred_improved = improved_proto.predict(test_task, self.feature_cols, k_shot=5)
        
        self.assertEqual(len(pred_basic), len(pred_improved))


if __name__ == '__main__':
    unittest.main()
