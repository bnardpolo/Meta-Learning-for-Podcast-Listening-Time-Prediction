import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def create_meta_tasks(df_encoded, X, y):
    tasks = []
    
    for podcast_name in df_encoded['Podcast_Name'].unique():
        mask = df_encoded['Podcast_Name'] == podcast_name
        podcast_data = df_encoded[mask].copy()
        
        podcast_X = X[mask].values
        podcast_y = y[mask].values
        
        if len(podcast_data) >= 10:
            tasks.append({
                'podcast_name': podcast_name,
                'X': podcast_X,
                'y': podcast_y,
                'n_episodes': len(podcast_data)
            })
    
    print(f"Created {len(tasks)} podcast tasks")
    print(f"Total episodes across tasks: {sum(t['n_episodes'] for t in tasks):,}")
    
    return tasks


def split_tasks(tasks, test_size=0.2, random_state=42):
    train_tasks, test_tasks = train_test_split(tasks, test_size=test_size, random_state=random_state)
    
    print(f"Train tasks (podcasts): {len(train_tasks)}")
    print(f"Test tasks (podcasts): {len(test_tasks)}")
    print(f"Train episodes: {sum(t['n_episodes'] for t in train_tasks):,}")
    print(f"Test episodes: {sum(t['n_episodes'] for t in test_tasks):,}")
    
    return train_tasks, test_tasks


def verify_data_quality(X):
    nan_count = np.isnan(X.values).sum()
    
    print(f"NaN count in X: {nan_count:,}")
    
    if nan_count > 0:
        print("Found NaN - needs fixing")
        return False
    else:
        print("Feature matrix is clean")
        return True


def fix_nan_values(X, strategy='median'):
    if np.isnan(X.values).sum() > 0:
        print("Fixing NaN values...")
        imputer = SimpleImputer(strategy=strategy)
        X_imputed = imputer.fit_transform(X)
        X_fixed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        print(f"Fixed! NaN count: {np.isnan(X_fixed.values).sum()}")
        return X_fixed
    return X


def verify_tasks(tasks):
    tasks_with_nan = sum(1 for t in tasks if np.isnan(t['X']).any())
    
    if tasks_with_nan > 0:
        print(f"Found {tasks_with_nan} tasks with NaN")
        return False
    else:
        print("All tasks are clean")
        return True


def fix_tasks(tasks, strategy='median'):
    imputer = SimpleImputer(strategy=strategy)
    
    for task in tasks:
        if np.isnan(task['X']).any():
            task['X'] = imputer.fit_transform(task['X'])
    
    print("All tasks fixed")
    return tasks


def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def save_results(results, filepath):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    with open(filepath, 'r') as f:
        results = json.load(f)
    print(f"Results loaded from {filepath}")
    return results


def print_task_info(task):
    print(f"\nTask Information:")
    print(f"  Podcast: {task['podcast_name']}")
    print(f"  Episodes: {task['n_episodes']}")
    print(f"  Features shape: {task['X'].shape}")
    print(f"  Target shape: {task['y'].shape}")


def get_task_statistics(tasks):
    n_tasks = len(tasks)
    total_episodes = sum(t['n_episodes'] for t in tasks)
    episode_counts = [t['n_episodes'] for t in tasks]
    
    return {
        'n_tasks': n_tasks,
        'total_episodes': total_episodes,
        'mean_episodes': np.mean(episode_counts),
        'median_episodes': np.median(episode_counts),
        'min_episodes': np.min(episode_counts),
        'max_episodes': np.max(episode_counts)
    }


def print_dataset_summary(train_tasks, test_tasks):
    train_stats = get_task_statistics(train_tasks)
    test_stats = get_task_statistics(test_tasks)
    
    print("\nDataset Summary:")
    print(f"  Train tasks: {train_stats['n_tasks']}")
    print(f"  Train episodes: {train_stats['total_episodes']:,}")
    print(f"  Test tasks: {test_stats['n_tasks']}")
    print(f"  Test episodes: {test_stats['total_episodes']:,}")
