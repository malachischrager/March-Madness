import pandas as pd
import os
from typing import Dict, List
import numpy as np

class DataProcessor:
    def __init__(self):
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.raw_path = os.path.join(self.base_path, 'raw')
        self.processed_path = os.path.join(self.base_path, 'processed')

    def process_team_stats(self, year: int = 2024) -> pd.DataFrame:
        """Process team statistics"""
        filepath = os.path.join(self.raw_path, f'team_stats_{year}.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No team stats file found for {year}")

        df = pd.read_csv(filepath)
        
        # Clean up column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Convert numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                continue

        # Calculate additional metrics
        if 'points' in df.columns and 'points_against' in df.columns:
            df['point_differential'] = df['points'] - df['points_against']
        
        # Save processed data
        processed_filepath = os.path.join(self.processed_path, f'processed_team_stats_{year}.csv')
        df.to_csv(processed_filepath, index=False)
        return df

    def process_tournament_data(self, start_year: int = 2019) -> pd.DataFrame:
        """Process historical tournament data"""
        all_data = []
        current_year = 2024

        for year in range(start_year, current_year + 1):
            filepath = os.path.join(self.raw_path, f'tournament_results_{year}.csv')
            if not os.path.exists(filepath):
                continue

            df = pd.read_csv(filepath)
            
            # Convert scores to numeric
            df['score1'] = pd.to_numeric(df['score1'], errors='coerce')
            df['score2'] = pd.to_numeric(df['score2'], errors='coerce')
            
            # Calculate margin of victory
            df['margin'] = df['score1'] - df['score2']
            
            # Add winner/loser columns
            df['winner'] = np.where(df['score1'] > df['score2'], df['team1'], df['team2'])
            df['loser'] = np.where(df['score1'] > df['score2'], df['team2'], df['team1'])
            
            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        # Combine all years
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save processed data
        processed_filepath = os.path.join(self.processed_path, 'processed_tournament_history.csv')
        combined_df.to_csv(processed_filepath, index=False)
        return combined_df

    def process_player_stats(self, year: int = 2024) -> pd.DataFrame:
        """Process player statistics"""
        filepath = os.path.join(self.raw_path, f'player_stats_{year}.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No player stats file found for {year}")

        df = pd.read_csv(filepath)
        
        # Clean up column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Convert numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                continue

        # Save processed data
        processed_filepath = os.path.join(self.processed_path, f'processed_player_stats_{year}.csv')
        df.to_csv(processed_filepath, index=False)
        return df

def main():
    processor = DataProcessor()
    
    # Process current season data
    try:
        team_stats = processor.process_team_stats()
        print("Processed team stats")
    except Exception as e:
        print(f"Error processing team stats: {e}")

    try:
        player_stats = processor.process_player_stats()
        print("Processed player stats")
    except Exception as e:
        print(f"Error processing player stats: {e}")

    try:
        tournament_data = processor.process_tournament_data()
        print("Processed tournament data")
    except Exception as e:
        print(f"Error processing tournament data: {e}")

if __name__ == "__main__":
    main()
