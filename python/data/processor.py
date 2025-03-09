import pandas as pd
import os
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        # Use absolute paths to top-level data directory
        # Get the absolute path to the March Madness project root
        self.base_path = '/Users/malachischrager/Desktop/Github/March Madness'
        
        # Use the root data directory consistently
        self.data_path = os.path.join(self.base_path, 'data')  # Root data directory
        self.raw_path = os.path.join(self.data_path, 'raw')
        self.processed_path = os.path.join(self.data_path, 'processed')
        
        # Define paths to key data files
        self.team_stats_path = os.path.join(self.raw_path, 'ncaa_team_stats.csv')
        
        self.scaler = StandardScaler()
        
        # Ensure raw and processed directories exist
        for path in [self.raw_path, self.processed_path]:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created directory: {path}")
            
        # Print the paths for debugging
        print(f"Project path: {self.base_path}")
        print(f"Data directory: {self.data_path}")
        print(f"Raw data: {self.raw_path}")
        print(f"Processed data: {self.processed_path}")
        
        # Check if key files exist
        if os.path.exists(self.team_stats_path):
            print(f"✓ NCAA team stats file found")
        else:
            print(f"✗ NCAA team stats file NOT found at: {self.team_stats_path}")
        
    def calculate_conference_strength(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate conference strength based on team performance"""
        # Extract conference from team name if needed
        if 'conference' not in team_stats.columns or team_stats['conference'].isna().all():
            team_stats['conference'] = team_stats['Team'].str.extract(r'\((.*?)\)$')
            
        # If conference is still missing for all teams, add a default
        if team_stats['conference'].isna().all() or team_stats['conference'].isnull().all():
            print("Warning: No conference information found, using default conference")
            team_stats['conference'] = 'Unknown'
        
        # Group by conference and calculate average metrics
        conf_metrics = team_stats.groupby('conference').agg({
            'W-L%': 'mean',  # Win-Loss percentage
            'SRS': 'mean',    # Simple Rating System
            'SOS': 'mean',    # Strength of Schedule
            'Tm.': 'mean',    # Points scored
            'Opp.': 'mean'    # Points against
        }).reset_index()
        
        # Calculate composite conference strength score
        numeric_cols = ['W-L%', 'SRS', 'SOS', 'Tm.', 'Opp.']
        # Handle any missing values
        conf_metrics[numeric_cols] = conf_metrics[numeric_cols].fillna(0)
        
        try:
            # Only try to scale if we have enough data
            if len(conf_metrics) > 1:
                scaled_data = self.scaler.fit_transform(conf_metrics[numeric_cols])
                conf_metrics['conference_strength'] = np.mean(scaled_data, axis=1)
            else:
                # Not enough data to scale properly, use unscaled average instead
                print("Not enough conference data to scale, using unscaled metrics")
                # Normalize manually instead
                for col in numeric_cols:
                    col_min = conf_metrics[col].min()
                    col_max = conf_metrics[col].max()
                    # Avoid division by zero
                    if col_max > col_min:
                        conf_metrics[f"{col}_norm"] = (conf_metrics[col] - col_min) / (col_max - col_min)
                    else:
                        conf_metrics[f"{col}_norm"] = 0.5  # Default value if no variation
                
                # For offense (Tm.), higher is better; for defense (Opp.), lower is better
                # Reverse the Opp. normalization
                if 'Opp._norm' in conf_metrics.columns:
                    conf_metrics['Opp._norm'] = 1 - conf_metrics['Opp._norm']
                
                # Calculate average of normalized metrics
                norm_cols = [f"{col}_norm" for col in numeric_cols if f"{col}_norm" in conf_metrics.columns]
                if norm_cols:
                    conf_metrics['conference_strength'] = conf_metrics[norm_cols].mean(axis=1)
                else:
                    # Fallback if normalization failed
                    conf_metrics['conference_strength'] = 0.5
        except Exception as e:
            print(f"Error scaling conference metrics: {e}")
            # Provide a default conference strength
            conf_metrics['conference_strength'] = 0.5
        
        return conf_metrics
    
    def calculate_team_experience(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate team experience metrics based on player data"""
        # Group by team and calculate experience metrics
        team_exp = player_stats.groupby('team').agg({
            'class': lambda x: np.mean([{'FR': 1, 'SO': 2, 'JR': 3, 'SR': 4}.get(c, 0) for c in x]),
            'games': 'mean',
            'minutes_played': 'sum'
        }).reset_index()
        
        team_exp.columns = ['team', 'avg_class_year', 'avg_games_played', 'total_minutes_played']
        return team_exp
    
    def calculate_momentum(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate team momentum metrics"""
        # Calculate win streak and point differential metrics
        team_momentum = pd.DataFrame()
        team_momentum['Team'] = team_stats['Team']
        
        # Calculate point differential
        team_stats['point_differential'] = team_stats['Tm.'] - team_stats['Opp.']
        
        # Use win percentage as a proxy for recent performance
        team_momentum['win_pct'] = team_stats['W-L%']
        team_momentum['point_diff'] = team_stats['point_differential']
        
        return team_momentum

    def process_team_stats(self, year: int = 2023) -> pd.DataFrame:
        """Process team statistics with advanced metrics"""
        # For 2024, use the dedicated file
        if year == 2024:
            year_specific_path = os.path.join(self.raw_path, f'team_stats_{year}.csv')
            if os.path.exists(year_specific_path):
                print(f"Using year-specific team stats file for {year}: {year_specific_path}")
                filepath = year_specific_path
            else:
                filepath = self.team_stats_path
        else:
            # Use the NCAA team stats file for all other years
            filepath = self.team_stats_path
        
        if not os.path.exists(filepath):
            print(f"Team stats file not found at {filepath}")
            print("Using mock data for testing purposes")
            df = self._create_mock_team_stats(year)
        else:
            print(f"Loading data from {filepath}")
            try:
                df = pd.read_csv(filepath)
                if 'Year' in df.columns:
                    print(f"Years available in data: {sorted(df['Year'].unique())}")
                    
                    # Filter by year if not using a year-specific file
                    if year != 2024 or filepath == self.team_stats_path:
                        df['Year'] = df['Year'].astype(int)
                        df = df[df['Year'] == int(year)].copy()
                else:
                    # If no Year column, assume the file is for the specific year
                    print(f"Assuming data is for year {year}")
                    df['Year'] = year
                
                # Ensure all required columns exist or add them with default values
                required_columns = ['Team', 'W-L%', 'SRS', 'SOS', 'conference']
                
                # For 2024 data which may have different column names
                if year == 2024 and filepath != self.team_stats_path:
                    # Rename columns if they exist with different names
                    column_mapping = {}
                    
                    # Check for alternate column names
                    if 'TEAM' in df.columns:
                        column_mapping['TEAM'] = 'Team'
                    if 'PPG' in df.columns and 'Tm.' not in df.columns:
                        column_mapping['PPG'] = 'Tm.'
                    if 'OPP_PPG' in df.columns and 'Opp.' not in df.columns:
                        column_mapping['OPP_PPG'] = 'Opp.'
                    if 'WIN_PCT' in df.columns and 'W-L%' not in df.columns:
                        column_mapping['WIN_PCT'] = 'W-L%'
                    if 'CONF' in df.columns and 'conference' not in df.columns:
                        column_mapping['CONF'] = 'conference'
                        
                    # Apply column renaming
                    if column_mapping:
                        df = df.rename(columns=column_mapping)
                        print(f"Renamed columns: {column_mapping}")
                
                # Add missing columns with default values
                for col in required_columns:
                    if col not in df.columns:
                        print(f"Adding missing column '{col}' with default values")
                        if col == 'Team':
                            df[col] = [f"Team {i+1}" for i in range(len(df))]
                        elif col == 'conference':
                            df[col] = 'Unknown'
                        else:
                            df[col] = 0.5  # Default value for numeric columns
                
                # Ensure scoring columns exist
                if 'Tm.' not in df.columns:
                    if 'Points For' in df.columns:
                        df['Tm.'] = df['Points For']
                    else:
                        df['Tm.'] = 70.0  # Reasonable default
                    print("Added 'Tm.' column with available or default values")
                if 'Opp.' not in df.columns:
                    if 'Points Against' in df.columns:
                        df['Opp.'] = df['Points Against']
                    else:
                        df['Opp.'] = 65.0  # Reasonable default
                    print("Added 'Opp.' column with available or default values")
            except Exception as e:
                print(f"Error reading team stats file: {e}")
                print(f"Exception details: {type(e).__name__}: {str(e)}")
                df = self._create_mock_team_stats(year)
        
        if len(df) == 0:
            print(f"No team stats found for {year}, using mock data")
            df = self._create_mock_team_stats(year)
        
        if len(df) == 0:
            raise FileNotFoundError(f"No team stats found for {year}")
            
        print(f"Found {len(df)} teams for {year}")
        
        # Drop unnamed columns and clean up column names
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Convert numeric columns, handling percentages
        numeric_cols = ['G', 'W', 'L', 'W-L%', 'SRS', 'SOS', 'Tm.', 'Opp.',
                       'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
                       'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    print(f"Could not convert {col} to numeric")
                    continue

        # Calculate point differential
        df['point_differential'] = df['Tm.'] - df['Opp.']
            
        # Add conference strength
        try:
            conf_strength = self.calculate_conference_strength(df)
            df = df.merge(conf_strength[['conference', 'conference_strength']], on='conference', how='left')
        except Exception as e:
            print(f"Error calculating conference strength: {e}")
            df['conference_strength'] = 0.0  # Default value if calculation fails
        
        # Add team experience if player stats available
        try:
            player_stats = self.process_player_stats(year)
            if len(player_stats) > 0:
                team_exp = self.calculate_team_experience(player_stats)
                df = df.merge(team_exp, on='Team', how='left')
            else:
                print(f"No player stats data for {year}")
        except FileNotFoundError:
            print(f"No player stats available for {year}, skipping experience metrics")
        except Exception as e:
            print(f"Error processing player stats: {e}")
        
        # Add momentum metrics directly from team stats
        try:
            momentum = self.calculate_momentum(df)
            df = df.merge(momentum[['Team', 'win_pct', 'point_diff']], on='Team', how='left')
        except Exception as e:
            print(f"Error calculating momentum: {e}")
            df['win_pct'] = 0.5  # Default values if calculation fails
            df['point_diff'] = 0.0
        
        # Save processed data
        processed_filepath = os.path.join(self.processed_path, f'processed_team_stats_{year}.csv')
        print(f"Saving processed data to {processed_filepath}")
        df.to_csv(processed_filepath, index=False)
        return df
        
    def _create_mock_team_stats(self, year: int) -> pd.DataFrame:
        """Create mock team stats for testing when the file doesn't exist"""
        print(f"Creating mock team stats data for {year}")
        # Create a basic dataset with key columns
        mock_teams = [
            f"Team {i} (Conference {(i % 10) + 1})" for i in range(1, 69)
        ]
        
        mock_data = {
            'Rk': list(range(1, 69)),
            'Team': mock_teams,
            'G': [32 + (i % 5) for i in range(68)],
            'W': [20 + (i % 12) for i in range(68)],
            'L': [5 + (i % 10) for i in range(68)],
            'W-L%': [0.65 + (i % 30) / 100 for i in range(68)],
            'SRS': [10 + (i % 15) for i in range(68)],
            'SOS': [5 + (i % 10) for i in range(68)],
            'Tm.': [70 + (i % 20) for i in range(68)],
            'Opp.': [65 + (i % 15) for i in range(68)],
            'Year': [year] * 68
        }
        
        df = pd.DataFrame(mock_data)
        # Extract conference from team name
        df['conference'] = df['Team'].str.extract(r'\((.*?)\)$')
        return df

    def process_tournament_data(self, start_year: int = 2019) -> pd.DataFrame:
        """Process historical tournament data"""
        all_data = []
        current_year = 2024

        for year in range(start_year, current_year + 1):
            filepath = os.path.join(self.raw_path, f'tournament_results_{year}.csv')
            if not os.path.exists(filepath):
                print(f"No tournament data found for {year}")
                continue

            try:
                df = pd.read_csv(filepath)
                if df.empty:
                    print(f"Tournament data file for {year} is empty")
                    # Create a minimal DataFrame with the required structure
                    df = pd.DataFrame({
                        'team1': [], 'team2': [], 'score1': [], 'score2': [], 'round': [], 'year': []
                    })
                    df['year'] = year
                elif 'year' not in df.columns:
                    # Add year column if missing
                    df['year'] = year
                
                # Convert scores to numeric
                if 'score1' in df.columns and 'score2' in df.columns:
                    df['score1'] = pd.to_numeric(df['score1'], errors='coerce')
                    df['score2'] = pd.to_numeric(df['score2'], errors='coerce')
                    
                    # Calculate margin of victory
                    df['margin'] = df['score1'] - df['score2']
                    
                    # Add winner/loser columns
                    if 'team1' in df.columns and 'team2' in df.columns:
                        df['winner'] = np.where(df['score1'] > df['score2'], df['team1'], df['team2'])
                        df['loser'] = np.where(df['score1'] > df['score2'], df['team2'], df['team1'])
                
                all_data.append(df)
            except Exception as e:
                print(f"Error processing tournament data for {year}: {e}")

        if not all_data:
            print("No tournament data available, returning empty DataFrame")
            return pd.DataFrame({
                'team1': [], 'team2': [], 'score1': [], 'score2': [], 'round': [], 
                'year': [], 'margin': [], 'winner': [], 'loser': []
            })

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

class Backtester:
    def __init__(self, processor: DataProcessor):
        self.processor = processor
        
    def evaluate_signals(self, year: int) -> Dict[str, float]:
        """Evaluate the predictive power of signals for a given year"""
        # Get processed data
        team_stats = self.processor.process_team_stats(year)
        tournament_data = self.processor.process_tournament_data()
        
        # Filter tournament data for the current year
        if 'year' in tournament_data.columns:
            tournament_year = tournament_data[tournament_data['year'] == year]
            print(f"Found {len(tournament_year)} tournament games for {year}")
        else:
            print(f"No year column in tournament data")
            tournament_year = pd.DataFrame()
        
        # Initialize signal performance metrics
        signal_performance = {
            'conference_strength_accuracy': 0.0,
            'team_experience_accuracy': 0.0,
            'momentum_accuracy': 0.0
        }
        
        # Evaluate each game
        total_games = len(tournament_year)
        if total_games == 0:
            return signal_performance
            
        correct_predictions = {
            'conference_strength': 0,
            'team_experience': 0,
            'momentum': 0
        }
        
        for _, game in tournament_year.iterrows():
            # Clean team names to match between datasets
            team1_name = game['team1'].split('(')[0].strip()
            team2_name = game['team2'].split('(')[0].strip()
            actual_winner = game['winner'].split('(')[0].strip()
            
            team1_stats = team_stats[team_stats['Team'].str.contains(team1_name, case=False, na=False)].iloc[0]
            team2_stats = team_stats[team_stats['Team'].str.contains(team2_name, case=False, na=False)].iloc[0]
            
            # Conference strength prediction
            if team1_stats['conference_strength'] > team2_stats['conference_strength']:
                predicted_winner = game['team1']
            else:
                predicted_winner = game['team2']
            if predicted_winner == actual_winner:
                correct_predictions['conference_strength'] += 1
            
            # Team experience prediction
            if 'avg_class_year' in team1_stats:
                if team1_stats['avg_class_year'] > team2_stats['avg_class_year']:
                    predicted_winner = game['team1']
                else:
                    predicted_winner = game['team2']
                if predicted_winner == actual_winner:
                    correct_predictions['team_experience'] += 1
            
            # Momentum prediction
            if 'recent_win_pct' in team1_stats:
                if team1_stats['recent_win_pct'] > team2_stats['recent_win_pct']:
                    predicted_winner = game['team1']
                else:
                    predicted_winner = game['team2']
                if predicted_winner == actual_winner:
                    correct_predictions['momentum'] += 1
        
        # Calculate accuracies
        signal_performance['conference_strength_accuracy'] = correct_predictions['conference_strength'] / total_games
        if 'avg_class_year' in team_stats.columns:
            signal_performance['team_experience_accuracy'] = correct_predictions['team_experience'] / total_games
        if 'recent_win_pct' in team_stats.columns:
            signal_performance['momentum_accuracy'] = correct_predictions['momentum'] / total_games
        
        return signal_performance

def main():
    processor = DataProcessor()
    
    print("\nAnalyzing NCAA team statistics...")
    
    # Process data for all available years plus current year
    available_years = [2019, 2020, 2021, 2022, 2023, 2024]  # Focus on recent years with tournament data
    
    # Store results for each year
    all_results = []
    
    # Process each year's data
    for year in available_years:
        try:
            print(f"\nProcessing team statistics for {year}...")
            processed_stats = processor.process_team_stats(year)
            
            # Analyze key metrics
            print(f"\nKey metrics for {year}:")
            print(f"Number of teams: {len(processed_stats)}")
            print(f"Average conference strength: {processed_stats['conference_strength'].mean():.4f}")
            print(f"Average point differential: {processed_stats['point_differential'].mean():.4f}")
            
            # Find top teams
            if len(processed_stats) > 0:
                try:
                    print(f"Highest scoring team: {processed_stats.loc[processed_stats['Tm.'].idxmax()]['Team']}")
                    print(f"Best defensive team: {processed_stats.loc[processed_stats['Opp.'].idxmin()]['Team']}")
                    print(f"Highest SRS team: {processed_stats.loc[processed_stats['SRS'].idxmax()]['Team']}")
                except Exception as e:
                    print(f"Could not determine top teams: {e}")
            
            # Process tournament data if available
            try:
                tournament_path = os.path.join(processor.raw_path, f'tournament_results_{year}.csv')
                if os.path.exists(tournament_path):
                    print(f"\nProcessing tournament data for {year}...")
                    # Check if file is empty
                    if os.path.getsize(tournament_path) > 0:
                        try:
                            tournament_data = pd.read_csv(tournament_path)
                            print(f"Tournament games: {len(tournament_data)}")
                            print(f"Tournament columns: {tournament_data.columns.tolist()}")
                        except pd.errors.EmptyDataError:
                            print(f"Tournament file exists but is empty")
                        except Exception as e:
                            print(f"Error reading tournament data: {type(e).__name__}: {str(e)}")
                    else:
                        print(f"Tournament file exists but is empty")
            except Exception as e:
                print(f"Error processing tournament data: {type(e).__name__}: {str(e)}")
            
            # Add summary metrics to results
            year_result = {
                'year': year,
                'num_teams': len(processed_stats),
                'avg_conference_strength': processed_stats['conference_strength'].mean() if 'conference_strength' in processed_stats.columns else None,
                'avg_point_differential': processed_stats['point_differential'].mean() if 'point_differential' in processed_stats.columns else None,
                'avg_win_pct': processed_stats['W-L%'].mean() if 'W-L%' in processed_stats.columns else None,
                'highest_srs': processed_stats['SRS'].max() if 'SRS' in processed_stats.columns else None
            }
            all_results.append(year_result)
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
    
    # Save summary results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_filepath = os.path.join(processor.processed_path, 'metrics_by_year.csv')
        results_df.to_csv(results_filepath, index=False)
        print(f"\nYear-by-year summary metrics saved to {results_filepath}")

if __name__ == "__main__":
    main()
