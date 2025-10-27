import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def print_metrics(metrics, model_name="Model"):
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {metrics['rmse']:.2f} minutes")
    print(f"  MAE: {metrics['mae']:.2f} minutes")
    print(f"  R2: {metrics['r2']:.3f}")


def compare_models(baseline_metrics, meta_metrics, k_shot=5):
    print("\nModel Comparison:")
    print(f"\nBaseline (all data):")
    print(f"  RMSE: {baseline_metrics['rmse']:.2f} minutes")
    print(f"  MAE: {baseline_metrics['mae']:.2f} minutes")
    print(f"  R2: {baseline_metrics['r2']:.3f}")
    
    print(f"\nMeta-Learning ({k_shot}-shot):")
    print(f"  RMSE: {meta_metrics['rmse']:.2f} minutes")
    print(f"  MAE: {meta_metrics['mae']:.2f} minutes")
    print(f"  R2: {meta_metrics['r2']:.3f}")
    
    rmse_diff = baseline_metrics['rmse'] - meta_metrics['rmse']
    rmse_pct = (rmse_diff / baseline_metrics['rmse']) * 100
    
    print(f"\nDifference:")
    print(f"  RMSE: {rmse_diff:+.2f} minutes ({rmse_pct:+.1f}%)")
    
    if rmse_diff > 0:
        print(f"  Meta-learning is BETTER")
    else:
        print(f"  Meta-learning uses {k_shot} episodes vs baseline 50+")


def get_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


def print_top_features(importance_df, n=20):
    print(f"\nTOP {n} MOST IMPORTANT FEATURES:")
    print("=" * 70)
    
    for i, row in importance_df.head(n).iterrows():
        bar_length = int(row['importance'] * 100)
        bar = '█' * bar_length
        print(f"{row['feature']:35s} | {bar} {row['importance']:.4f}")


def get_top_n_features(importance_df, n=10):
    return importance_df.head(n)['feature'].tolist()


def calculate_cumulative_importance(importance_df):
    importance_df = importance_df.copy()
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    return importance_df


def features_for_threshold(importance_df, threshold=0.90):
    cumulative_df = calculate_cumulative_importance(importance_df)
    n_features = (cumulative_df['cumulative_importance'] <= threshold).sum()
    return n_features


def print_results_summary(results_list):
    print("\nResults Summary:")
    print("=" * 70)
    
    for result in results_list:
        k_shot = result.get('k_shot', 'N/A')
        rmse = result.get('rmse', 0)
        mae = result.get('mae', 0)
        r2 = result.get('r2', 0)
        
        print(f"\n{k_shot}-shot:")
        print(f"  RMSE: {rmse:.2f} minutes")
        print(f"  MAE: {mae:.2f} minutes")
        print(f"  R2: {r2:.3f}")
