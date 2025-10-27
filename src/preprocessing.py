import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class DataPreprocessor:
    
    def __init__(self):
        self.genre_medians = {}
        self.number_of_ads_median = None
        
    def load_data(self, train_path: str, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path else None
        return train_df, test_df
    
    def handle_missing_values(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        if is_train:
            self.number_of_ads_median = df['Number_of_Ads'].median()
            self.genre_medians = df.groupby('Genre')['Episode_Length_minutes'].median().to_dict()
        
        df['Number_of_Ads'].fillna(self.number_of_ads_median, inplace=True)
        
        for genre in df['Genre'].unique():
            mask = (df['Genre'] == genre) & (df['Episode_Length_minutes'].isna())
            if genre in self.genre_medians:
                df.loc[mask, 'Episode_Length_minutes'] = self.genre_medians[genre]
        
        df['Guest_Popularity_percentage'].fillna(0, inplace=True)
        
        return df
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        missing_summary = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        
        return missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    def get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats = {
            'shape': df.shape,
            'unique_podcasts': df['Podcast_Name'].nunique(),
            'unique_genres': df['Genre'].nunique(),
            'target_mean': df['Listening_Time_minutes'].mean(),
            'target_median': df['Listening_Time_minutes'].median(),
            'target_std': df['Listening_Time_minutes'].std(),
            'target_min': df['Listening_Time_minutes'].min(),
            'target_max': df['Listening_Time_minutes'].max(),
            'genre_distribution': df['Genre'].value_counts()
        }
        return stats
    
    def get_episodes_per_podcast(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('Podcast_Name').size().sort_values(ascending=False)
    
    def preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_processed = self.handle_missing_values(train_df, is_train=True)
        
        if test_df is not None:
            test_processed = self.handle_missing_values(test_df, is_train=False)
            return train_processed, test_processed
        
        return train_processed, None
    
    def validate_preprocessing(self, df: pd.DataFrame) -> bool:
        missing_count = df.isnull().sum().sum()
        return missing_count == 0


def load_and_preprocess_data(train_path: str, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.load_data(train_path, test_path)
    train_processed, test_processed = preprocessor.preprocess(train_df, test_df)
    
    return train_processed, test_processed


def get_data_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Unique Podcasts: {df['Podcast_Name'].nunique()}")
    print(f"Unique Genres: {df['Genre'].nunique()}")
    
    print("\nTarget Variable Statistics:")
    print(f"  Mean: {df['Listening_Time_minutes'].mean():.2f} minutes")
    print(f"  Median: {df['Listening_Time_minutes'].median():.2f} minutes")
    print(f"  Std: {df['Listening_Time_minutes'].std():.2f} minutes")
    print(f"  Min: {df['Listening_Time_minutes'].min():.2f} minutes")
    print(f"  Max: {df['Listening_Time_minutes'].max():.2f} minutes")
    
    print("\nTop 5 Genres:")
    genre_counts = df['Genre'].value_counts().head(5)
    for genre, count in genre_counts.items():
        pct = count / len(df) * 100
        print(f"  {genre:15s}: {count:7,} ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)


def print_missing_values_report(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("MISSING VALUES REPORT")
    print("=" * 70)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_summary = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_summary) > 0:
        print("\n", missing_summary)
    else:
        print("\nNo missing values found.")
    
    print("\n" + "=" * 70)
