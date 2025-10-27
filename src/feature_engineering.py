import numpy as np
import pandas as pd
from typing import Tuple


class FeatureEngineer:
    
    def __init__(self):
        self.feature_count = {
            'host': 10,
            'guest': 10,
            'content': 8,
            'publication': 10,
            'ads': 5,
            'temporal': 8,
            'interaction': 10
        }
    
    def create_host_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['Host_Popularity_Binned'] = pd.cut(
            df['Host_Popularity_percentage'],
            bins=[0, 30, 60, 80, 100],
            labels=['Low', 'Medium', 'High', 'Star']
        )
        
        df['Host_Popularity_Quintile'] = pd.qcut(
            df['Host_Popularity_percentage'],
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        )
        
        genre_host_mean = df.groupby('Genre')['Host_Popularity_percentage'].transform('mean')
        df['Host_vs_Genre_Mean'] = df['Host_Popularity_percentage'] - genre_host_mean
        
        genre_host_median = df.groupby('Genre')['Host_Popularity_percentage'].transform('median')
        df['Host_vs_Genre_Median'] = df['Host_Popularity_percentage'] - genre_host_median
        
        df['Host_Percentile'] = df['Host_Popularity_percentage'].rank(pct=True) * 100
        
        df['Host_Is_Rising_Star'] = (
            (df['Host_Popularity_percentage'] > 70) & 
            (df['Host_vs_Genre_Mean'] > 10)
        ).astype(int)
        
        df['Host_Is_Established'] = (
            (df['Host_Popularity_percentage'] > 60) &
            (df['Host_vs_Genre_Mean'] <= 10)
        ).astype(int)
        
        df['Host_Popularity_Squared'] = df['Host_Popularity_percentage'] ** 2
        
        def categorize_host(pop):
            if pop < 20: return 'Emerging'
            if pop < 50: return 'Known'
            if pop < 75: return 'Famous'
            return 'Celebrity'
        
        df['Host_Category'] = df['Host_Popularity_percentage'].apply(categorize_host)
        
        df['Host_Popularity_Log'] = np.log1p(df['Host_Popularity_percentage'])
        
        return df
    
    def create_guest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['Has_Guest'] = (df['Guest_Popularity_percentage'] > 0).astype(int)
        
        df['Guest_Popularity_Binned'] = pd.cut(
            df['Guest_Popularity_percentage'],
            bins=[-0.1, 0, 30, 60, 100],
            labels=['No_Guest', 'Minor', 'Notable', 'Star']
        )
        
        df['Guest_vs_Host_Diff'] = (df['Guest_Popularity_percentage'] - 
                                     df['Host_Popularity_percentage'])
        
        df['Is_Mega_Guest'] = (df['Guest_Popularity_percentage'] > 90).astype(int)
        
        df['Guest_Host_Synergy'] = (df['Guest_Popularity_percentage'] * 
                                     df['Host_Popularity_percentage'])
        
        df['Guest_Host_Mismatch'] = (
            np.abs(df['Guest_vs_Host_Diff']) > 30
        ).astype(int)
        
        def categorize_guest(pop, has_guest):
            if has_guest == 0: return 'No_Guest'
            if pop < 30: return 'Minor_Guest'
            if pop < 70: return 'Notable_Guest'
            return 'Star_Guest'
        
        df['Guest_Category'] = df.apply(
            lambda x: categorize_guest(x['Guest_Popularity_percentage'], x['Has_Guest']), axis=1
        )
        
        df['Guest_Boost_Potential'] = np.maximum(0, df['Guest_vs_Host_Diff'])
        
        df['Total_Star_Power'] = (df['Host_Popularity_percentage'] + 
                                  df['Guest_Popularity_percentage'])
        
        df['Guest_Percentile'] = df['Guest_Popularity_percentage'].rank(pct=True) * 100
        
        return df
    
    def create_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['Episode_Length_Binned'] = pd.cut(
            df['Episode_Length_minutes'],
            bins=[0, 30, 60, 90, 1000],
            labels=['Short', 'Medium', 'Long', 'VeryLong']
        )
        
        genre_length_mean = df.groupby('Genre')['Episode_Length_minutes'].transform('mean')
        df['Length_vs_Genre_Mean'] = df['Episode_Length_minutes'] - genre_length_mean
        
        genre_length_40 = df.groupby('Genre')['Episode_Length_minutes'].transform(lambda x: x.quantile(0.4))
        genre_length_60 = df.groupby('Genre')['Episode_Length_minutes'].transform(lambda x: x.quantile(0.6))
        df['Is_Optimal_Length'] = (
            (df['Episode_Length_minutes'] >= genre_length_40) &
            (df['Episode_Length_minutes'] <= genre_length_60)
        ).astype(int)
        
        sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        df['Sentiment_Encoded'] = df['Episode_Sentiment'].map(sentiment_map)
        
        df['Is_Positive_Sentiment'] = (df['Episode_Sentiment'] == 'Positive').astype(int)
        
        sentiment_weight = df['Sentiment_Encoded'].fillna(0) + 2
        df['Content_Quality_Score'] = df['Total_Star_Power'] * sentiment_weight
        
        df['Length_Sentiment_Interaction'] = (
            df['Episode_Length_minutes'] * df['Sentiment_Encoded'].fillna(0)
        )
        
        df['Episode_Complexity'] = (
            df['Episode_Length_minutes'] / 60 * (1 + df['Has_Guest'])
        )
        
        return df
    
    def create_publication_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        day_mapping = {
            'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
            'Friday': 5, 'Saturday': 6, 'Sunday': 7
        }
        df['Publication_Day_Encoded'] = df['Publication_Day'].map(day_mapping)
        
        df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
        
        df['Is_Prime_Day'] = df['Publication_Day'].isin(['Thursday', 'Friday']).astype(int)
        
        time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
        df['Publication_Time_Encoded'] = df['Publication_Time'].map(time_mapping)
        
        df['Is_Prime_Time'] = df['Publication_Time'].isin(['Evening', 'Night']).astype(int)
        
        df['Is_Morning_Commute'] = (
            (df['Publication_Time'] == 'Morning') & 
            (df['Is_Weekend'] == 0)
        ).astype(int)
        
        timing_scores = {
            ('Friday', 'Evening'): 5, ('Friday', 'Night'): 5,
            ('Thursday', 'Evening'): 5, ('Thursday', 'Night'): 5,
            ('Monday', 'Morning'): 4, ('Tuesday', 'Morning'): 4,
            ('Wednesday', 'Morning'): 4, ('Thursday', 'Morning'): 4,
        }
        
        def calculate_timing_score(row):
            key = (row['Publication_Day'], row['Publication_Time'])
            return timing_scores.get(key, 3)
        
        df['Publication_Timing_Score'] = df.apply(calculate_timing_score, axis=1)
        
        df['Day_Time_Combo'] = df['Publication_Day'] + '_' + df['Publication_Time']
        
        def categorize_release_strategy(row):
            if row['Is_Prime_Day'] and row['Is_Prime_Time']:
                return 'Optimal'
            elif row['Is_Prime_Day'] or row['Is_Prime_Time']:
                return 'Good'
            elif row['Is_Weekend']:
                return 'Suboptimal'
            else:
                return 'Poor'
        
        df['Release_Strategy_Category'] = df.apply(categorize_release_strategy, axis=1)
        
        df['Is_Weekend_Prime'] = (
            (df['Is_Weekend'] == 1) & 
            (df['Is_Prime_Time'] == 1)
        ).astype(int)
        
        return df
    
    def create_ad_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['Ads_Binned'] = pd.cut(
            df['Number_of_Ads'],
            bins=[-0.1, 0, 2, 4, 100],
            labels=['None', 'Few', 'Many', 'Heavy']
        )
        
        df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
        df['Ad_Density'] = df['Ad_Density'].fillna(0)
        
        df['Has_Ads'] = (df['Number_of_Ads'] > 0).astype(int)
        
        def categorize_ad_load(ads, density):
            if ads == 0:
                return 'Ad_Free'
            elif density < 0.05:
                return 'Light_Ads'
            elif density < 0.1:
                return 'Moderate_Ads'
            else:
                return 'Heavy_Ads'
        
        df['Ad_Load_Category'] = df.apply(
            lambda x: categorize_ad_load(x['Number_of_Ads'], x['Ad_Density']), axis=1
        )
        
        df['Ad_Intensity'] = df['Number_of_Ads'] / df['Number_of_Ads'].max()
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df = df.sort_values(['Podcast_Name', 'Episode_Title']).reset_index(drop=True)
        
        df['Episode_Number'] = df.groupby('Podcast_Name').cumcount() + 1
        
        df['Episode_Number_Log'] = np.log1p(df['Episode_Number'])
        
        df['Is_Pilot_Episode'] = (df['Episode_Number'] == 1).astype(int)
        
        df['Is_Early_Episode'] = (df['Episode_Number'] <= 10).astype(int)
        
        df['Is_Established_Episode'] = (df['Episode_Number'] > 50).astype(int)
        
        podcast_episode_counts = df.groupby('Podcast_Name')['Episode_Number'].transform('max')
        df['Podcast_Total_Episodes'] = podcast_episode_counts
        
        def categorize_maturity(total_episodes):
            if total_episodes < 1000:
                return 'Launch'
            elif total_episodes < 10000:
                return 'Growth'
            else:
                return 'Mature'
        
        df['Podcast_Maturity_Stage'] = df['Podcast_Total_Episodes'].apply(categorize_maturity)
        
        df['Episode_Progress'] = df['Episode_Number'] / df['Podcast_Total_Episodes']
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['Guest_Sentiment_Score'] = (
            df['Guest_Popularity_percentage'] * 
            (df['Sentiment_Encoded'].fillna(0) + 2)
        )
        
        df['Ads_Weekend_Effect'] = df['Number_of_Ads'] * df['Is_Weekend']
        
        df['Prime_Quality_Score'] = df['Is_Prime_Time'] * df['Content_Quality_Score']
        
        df['Host_Sentiment_Score'] = (
            df['Host_Popularity_percentage'] * 
            (df['Sentiment_Encoded'].fillna(0) + 2)
        )
        
        df['Length_Timing_Score'] = (
            df['Episode_Length_minutes'] * df['Publication_Timing_Score']
        )
        
        df['Ads_Length_Ratio'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] / 10)
        
        df['Guest_Host_Ratio'] = np.where(
            df['Host_Popularity_percentage'] > 0,
            df['Guest_Popularity_percentage'] / df['Host_Popularity_percentage'],
            0
        )
        
        df['Weekend_Prime_Score'] = df['Is_Weekend'] * df['Is_Prime_Time']
        
        strategy_mapping = {'Optimal': 4, 'Good': 3, 'Suboptimal': 2, 'Poor': 1}
        df['Star_Strategy_Score'] = (
            df['Total_Star_Power'] * 
            df['Release_Strategy_Category'].map(strategy_mapping)
        )
        
        df['Maturity_Quality_Score'] = df['Episode_Progress'] * df['Content_Quality_Score']
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df = self.create_host_features(df)
        df = self.create_guest_features(df)
        df = self.create_content_features(df)
        df = self.create_publication_features(df)
        df = self.create_ad_features(df)
        df = self.create_temporal_features(df)
        df = self.create_interaction_features(df)
        
        return df
    
    def get_feature_summary(self) -> dict:
        total = sum(self.feature_count.values())
        return {
            'total_features': total,
            'breakdown': self.feature_count
        }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineer = FeatureEngineer()
    return engineer.engineer_all_features(df)


def get_categorical_columns() -> list:
    return [
        'Genre',
        'Publication_Day',
        'Publication_Time',
        'Episode_Sentiment',
        'Host_Popularity_Binned',
        'Host_Popularity_Quintile',
        'Host_Category',
        'Guest_Popularity_Binned',
        'Guest_Category',
        'Episode_Length_Binned',
        'Day_Time_Combo',
        'Release_Strategy_Category',
        'Ads_Binned',
        'Ad_Load_Category',
        'Podcast_Maturity_Stage'
    ]


def encode_categorical_features(df: pd.DataFrame, categorical_columns: list = None) -> pd.DataFrame:
    df = df.copy()
    
    if categorical_columns is None:
        categorical_columns = get_categorical_columns()
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    
    return df
