import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'Podcast_Name': ['Podcast A', 'Podcast A', 'Podcast B', 'Podcast B', 'Podcast C'],
            'Episode_Title': ['Ep1', 'Ep2', 'Ep3', 'Ep4', 'Ep5'],
            'Episode_Length_minutes': [30.0, np.nan, 45.0, 60.0, np.nan],
            'Genre': ['Comedy', 'Comedy', 'News', 'News', 'Tech'],
            'Host_Popularity_percentage': [50.0, 60.0, 70.0, 80.0, 90.0],
            'Publication_Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'Publication_Time': ['Morning', 'Afternoon', 'Evening', 'Night', 'Morning'],
            'Guest_Popularity_percentage': [np.nan, 50.0, np.nan, 70.0, 80.0],
            'Number_of_Ads': [1.0, 2.0, np.nan, 3.0, 0.0],
            'Episode_Sentiment': ['Positive', 'Neutral', 'Negative', 'Positive', 'Neutral'],
            'Listening_Time_minutes': [25.0, 45.0, 35.0, 50.0, 40.0]
        })
    
    def test_handle_missing_values(self):
        df = self.sample_data.copy()
        df_processed = self.preprocessor.handle_missing_values(df, is_train=True)
        
        missing_count = df_processed.isnull().sum().sum()
        self.assertEqual(missing_count, 0)
    
    def test_guest_popularity_filled_with_zero(self):
        df = self.sample_data.copy()
        df_processed = self.preprocessor.handle_missing_values(df, is_train=True)
        
        self.assertEqual(df_processed.loc[0, 'Guest_Popularity_percentage'], 0)
        self.assertEqual(df_processed.loc[2, 'Guest_Popularity_percentage'], 0)
    
    def test_number_of_ads_filled(self):
        df = self.sample_data.copy()
        df_processed = self.preprocessor.handle_missing_values(df, is_train=True)
        
        self.assertFalse(df_processed['Number_of_Ads'].isnull().any())
    
    def test_episode_length_filled_by_genre(self):
        df = self.sample_data.copy()
        df_processed = self.preprocessor.handle_missing_values(df, is_train=True)
        
        self.assertFalse(df_processed['Episode_Length_minutes'].isnull().any())
    
    def test_check_missing_values(self):
        df = self.sample_data.copy()
        missing_summary = self.preprocessor.check_missing_values(df)
        
        self.assertTrue(len(missing_summary) > 0)
    
    def test_validate_preprocessing(self):
        df = self.sample_data.copy()
        df_processed = self.preprocessor.handle_missing_values(df, is_train=True)
        
        is_valid = self.preprocessor.validate_preprocessing(df_processed)
        self.assertTrue(is_valid)
    
    def test_get_basic_statistics(self):
        df = self.sample_data.copy()
        stats = self.preprocessor.get_basic_statistics(df)
        
        self.assertEqual(stats['shape'], (5, 12))
        self.assertTrue('target_mean' in stats)
        self.assertTrue('unique_podcasts' in stats)
    
    def test_preprocess(self):
        train_df = self.sample_data.copy()
        test_df = self.sample_data.copy()
        
        train_processed, test_processed = self.preprocessor.preprocess(train_df, test_df)
        
        self.assertEqual(train_processed.isnull().sum().sum(), 0)
        self.assertEqual(test_processed.isnull().sum().sum(), 0)


if __name__ == '__main__':
    unittest.main()
