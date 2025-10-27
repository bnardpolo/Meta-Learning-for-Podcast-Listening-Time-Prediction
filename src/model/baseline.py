import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_baseline_model(train_tasks):
    X_train_baseline = np.vstack([task['X'] for task in train_tasks])
    y_train_baseline = np.concatenate([task['y'] for task in train_tasks])
    
    print(f"Training samples: {len(X_train_baseline):,}")
    print(f"Features: {X_train_baseline.shape[1]}")
    
    baseline_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    baseline_model.fit(X_train_baseline, y_train_baseline)
    
    print("Baseline model trained")
    
    return baseline_model


def evaluate_baseline_model(baseline_model, test_tasks):
    baseline_predictions = []
    baseline_actuals = []
    
    for task in test_tasks:
        y_pred = baseline_model.predict(task['X'])
        y_pred = np.clip(y_pred, 0, 120)
        baseline_predictions.extend(y_pred)
        baseline_actuals.extend(task['y'])
    
    baseline_rmse = np.sqrt(mean_squared_error(baseline_actuals, baseline_predictions))
    baseline_mae = mean_absolute_error(baseline_actuals, baseline_predictions)
    baseline_r2 = r2_score(baseline_actuals, baseline_predictions)
    
    print("\nBASELINE PERFORMANCE:")
    print(f"RMSE: {baseline_rmse:.2f} minutes")
    print(f"MAE: {baseline_mae:.2f} minutes")
    print(f"R2: {baseline_r2:.3f}")
    print(f"Predictions: {len(baseline_predictions):,}")
    
    return {
        'rmse': baseline_rmse,
        'mae': baseline_mae,
        'r2': baseline_r2,
        'predictions': baseline_predictions,
        'actuals': baseline_actuals
    }
