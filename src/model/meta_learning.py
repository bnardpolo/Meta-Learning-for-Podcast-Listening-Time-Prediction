import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PrototypicalNetwork:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.prototypes = {}
        
    def fit(self, train_tasks, feature_cols, k_shot=5):
        print(f"Learning prototypes with {k_shot}-shot...")
        
        all_support_data = []
        
        for task in train_tasks:
            if task['n_episodes'] >= k_shot + 5:
                feature_indices = [i for i, col in enumerate(feature_cols) if col in self.feature_names]
                
                support_X = task['X'][:k_shot, feature_indices]
                
                prototype = support_X.mean(axis=0)
                
                self.prototypes[task['podcast_name']] = prototype
                all_support_data.append(support_X)
        
        all_support = np.vstack(all_support_data)
        self.scaler.fit(all_support)
        
        print(f"Created {len(self.prototypes)} prototypes")
        return self
    
    def predict(self, task, feature_cols, k_shot=5):
        feature_indices = [i for i, col in enumerate(feature_cols) if col in self.feature_names]
        
        support_X = task['X'][:k_shot, feature_indices]
        support_y = task['y'][:k_shot]
        
        query_X = task['X'][k_shot:, feature_indices]
        
        support_X_scaled = self.scaler.transform(support_X)
        query_X_scaled = self.scaler.transform(query_X)
        
        prototype = support_X_scaled.mean(axis=0)
        
        mean_support_y = support_y.mean()
        
        predictions = np.full(len(query_X), mean_support_y)
        
        return predictions


def train_prototypical_network(train_tasks, top_features, feature_cols, k_shot=5):
    proto_net = PrototypicalNetwork(feature_names=top_features)
    proto_net.fit(train_tasks, feature_cols, k_shot=k_shot)
    return proto_net


def evaluate_prototypical_network(proto_net, test_tasks, feature_cols, k_shots=[1, 3, 5, 10, 20]):
    test_results = []
    
    for k_shot in k_shots:
        predictions = []
        actuals = []
        
        for task in test_tasks:
            if task['n_episodes'] >= k_shot + 5:
                y_pred = proto_net.predict(task, feature_cols, k_shot=k_shot)
                y_actual = task['y'][k_shot:]
                
                predictions.extend(y_pred)
                actuals.extend(y_actual)
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        test_results.append({
            'k_shot': k_shot,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_predictions': len(predictions)
        })
        
        print(f"{k_shot}-shot Learning:")
        print(f"  RMSE: {rmse:.2f} minutes")
        print(f"  MAE: {mae:.2f} minutes")
        print(f"  R2: {r2:.3f}")
        print(f"  Predictions: {len(predictions):,}\n")
    
    return test_results
