import pandas as pd
import os
import sys
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add the project root to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from the project modules
# We don't actually need DataProcessor for this standalone backtester

class Backtester:
    def __init__(self, data_dir='data'):
        """
        Initialize the backtester with paths to data directories
        
        Args:
            data_dir: Path to the data directory (can be 'data' or 'data/sample')
        """
        # Use absolute paths to top-level data directory
        self.base_path = '/Users/malachischrager/Desktop/Github/March Madness'
        self.data_path = os.path.join(self.base_path, data_dir)
        
        # Set up paths based on the provided data directory
        if data_dir == 'data':
            self.raw_path = os.path.join(self.data_path, 'raw')
        else:
            # Use the provided directory directly (e.g., data/sample)
            self.raw_path = self.data_path
            
        self.processed_path = os.path.join(self.base_path, 'data', 'processed')
        
        # Ensure processed directory exists
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)
            
        # Print paths for debugging
        print(f"Base path: {self.base_path}")
        print(f"Data path: {self.data_path}")
        print(f"Raw path: {self.raw_path}")
        print(f"Processed path: {self.processed_path}")
        
    def load_team_stats(self, year):
        """Load team statistics for a specific year"""
        filepath = os.path.join(self.raw_path, f'team_stats_{year}.csv')
        
        if not os.path.exists(filepath):
            print(f"Team stats file not found at {filepath}")
            return None
            
        print(f"Loading team stats from {filepath}")
        df = pd.read_csv(filepath)
        
        # Handle different column formats
        # Map common column names to expected format
        column_mapping = {
            'team': 'Team',
            'win_pct': 'W-L%',
            'srs': 'SRS',
            'sos': 'SOS'
        }
        
        # Rename columns if needed
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Add Year column if missing
        if 'Year' not in df.columns:
            df['Year'] = year
            
        # Ensure required columns exist
        required_columns = ['Team', 'W-L%', 'SRS', 'SOS', 'conference', 'conference_strength']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns in team stats: {missing_columns}")
            # Add missing columns with default values
            for col in missing_columns:
                if col == 'conference':
                    if 'conference' not in df.columns:
                        df[col] = 'Unknown'
                elif col == 'W-L%':
                    # Try to calculate from wins/losses if available
                    if 'wins' in df.columns and 'losses' in df.columns:
                        df['W-L%'] = df['wins'] / (df['wins'] + df['losses'])
                    else:
                        df['W-L%'] = 0.5  # Default value
                elif col == 'conference_strength':
                    # Calculate conference strength if missing
                    self._calculate_conference_strength(df)
                else:
                    df[col] = 0.5  # Default value
        
        # Add other commonly used columns if missing
        if 'avg_class_year' not in df.columns:
            df['avg_class_year'] = 2.5  # Default value (sophomore/junior average)
            
        if 'recent_win_pct' not in df.columns:
            if 'W-L%' in df.columns:
                df['recent_win_pct'] = df['W-L%'] * 1.05  # Simple approximation
                df['recent_win_pct'] = df['recent_win_pct'].clip(upper=1.0)  # Cap at 1.0
            else:
                df['recent_win_pct'] = 0.6  # Default value
                
        # Add Tm. and Opp. if missing (offensive and defensive stats)
        if 'Tm.' not in df.columns:
            df['Tm.'] = 75.0  # Default value
        if 'Opp.' not in df.columns:
            df['Opp.'] = 70.0  # Default value
                    
        return df
        
    def load_tournament_data(self, year):
        """Load tournament data for a specific year"""
        filepath = os.path.join(self.raw_path, f'tournament_results_{year}.csv')
        
        if not os.path.exists(filepath):
            print(f"Tournament results file not found at {filepath}")
            return None
            
        print(f"Loading tournament data from {filepath}")
        df = pd.read_csv(filepath)
        
        if df.empty:
            print(f"Tournament data is empty for {year}")
            return None
            
        # Ensure required columns exist
        required_columns = ['team1', 'team2', 'winner', 'round']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns in tournament data: {missing_columns}")
            return None
            
        return df
    
    def _calculate_conference_strength(self, team_stats):
        """Calculate conference strength if it's missing from the data"""
        if 'conference' not in team_stats.columns:
            print("Cannot calculate conference strength: 'conference' column is missing")
            team_stats['conference_strength'] = 0.5
            return
            
        # Group by conference and calculate average metrics
        metrics = ['W-L%', 'SRS', 'SOS']
        available_metrics = [m for m in metrics if m in team_stats.columns]
        
        if not available_metrics:
            print("Cannot calculate conference strength: no metrics available")
            team_stats['conference_strength'] = 0.5
            return
            
        conf_metrics = team_stats.groupby('conference')[available_metrics].mean().reset_index()
        
        # Normalize metrics
        for col in available_metrics:
            col_min = conf_metrics[col].min()
            col_max = conf_metrics[col].max()
            if col_max > col_min:
                conf_metrics[f"{col}_norm"] = (conf_metrics[col] - col_min) / (col_max - col_min)
            else:
                conf_metrics[f"{col}_norm"] = 0.5
                
        # Calculate average of normalized metrics
        norm_cols = [f"{col}_norm" for col in available_metrics]
        conf_metrics['conference_strength'] = conf_metrics[norm_cols].mean(axis=1)
        
        # Merge back to team_stats
        team_stats_with_strength = pd.merge(
            team_stats, 
            conf_metrics[['conference', 'conference_strength']], 
            on='conference', 
            how='left'
        )
        
        # Update the original dataframe
        team_stats['conference_strength'] = team_stats_with_strength['conference_strength']
        
        # Fill any missing values
        team_stats['conference_strength'] = team_stats['conference_strength'].fillna(0.5)
        
    def evaluate_signals(self, year, verbose=True):
        """
        Evaluate the predictive power of different signals for tournament games
        
        Args:
            year: The year to evaluate
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary of signal performance metrics
        """
        # Load data
        team_stats = self.load_team_stats(year)
        tournament_data = self.load_tournament_data(year)
        
        if team_stats is None or tournament_data is None:
            print(f"Cannot evaluate signals for {year}: missing data")
            return {}
            
        if verbose:
            print(f"\nEvaluating signals for {year} tournament")
            print(f"Team stats shape: {team_stats.shape}")
            print(f"Tournament games: {len(tournament_data)}")
            
        # Initialize signal performance metrics
        signal_performance = {
            'conference_strength_accuracy': 0.0,
            'team_experience_accuracy': 0.0,
            'momentum_accuracy': 0.0,
            'win_pct_accuracy': 0.0,
            'srs_accuracy': 0.0
        }
        
        # Initialize prediction counters
        correct_predictions = {
            'conference_strength': 0,
            'team_experience': 0,
            'momentum': 0,
            'win_pct': 0,
            'srs': 0
        }
        
        # Track games that couldn't be evaluated
        skipped_games = 0
        
        # Evaluate each game
        for _, game in tournament_data.iterrows():
            # Clean team names to match between datasets
            team1_name = game['team1'].split('(')[0].strip()
            team2_name = game['team2'].split('(')[0].strip()
            
            # Handle the winner format
            if 'winner' in game:
                actual_winner = game['winner'].split('(')[0].strip()
            else:
                # If no winner column, determine from scores
                if 'score1' in game and 'score2' in game and pd.notna(game['score1']) and pd.notna(game['score2']):
                    actual_winner = team1_name if int(game['score1']) > int(game['score2']) else team2_name
                else:
                    if verbose:
                        print(f"Skipping game {team1_name} vs {team2_name}: no winner information")
                    skipped_games += 1
                    continue
            
            # Find team stats - improved matching logic
            # First try exact match
            team1_stats = team_stats[team_stats['Team'] == team1_name]
            # If no exact match, try case-insensitive contains
            if len(team1_stats) == 0:
                team1_stats = team_stats[team_stats['Team'].str.contains(team1_name, case=False, na=False, regex=False)]
                # If still no match, try with first word only (e.g., "North Carolina" -> "North")
                if len(team1_stats) == 0 and ' ' in team1_name:
                    first_word = team1_name.split(' ')[0]
                    team1_stats = team_stats[team_stats['Team'].str.contains(first_word, case=False, na=False, regex=False)]
            
            # Same for team2
            team2_stats = team_stats[team_stats['Team'] == team2_name]
            if len(team2_stats) == 0:
                team2_stats = team_stats[team_stats['Team'].str.contains(team2_name, case=False, na=False, regex=False)]
                if len(team2_stats) == 0 and ' ' in team2_name:
                    first_word = team2_name.split(' ')[0]
                    team2_stats = team_stats[team_stats['Team'].str.contains(first_word, case=False, na=False, regex=False)]
            
            # Skip if we can't find the teams
            if len(team1_stats) == 0 or len(team2_stats) == 0:
                if verbose:
                    if len(team1_stats) == 0:
                        print(f"Could not find stats for {team1_name}")
                    if len(team2_stats) == 0:
                        print(f"Could not find stats for {team2_name}")
                skipped_games += 1
                continue
                
            # Get the first match for each team
            team1_stats = team1_stats.iloc[0]
            team2_stats = team2_stats.iloc[0]
            
            # Conference strength prediction
            if 'conference_strength' in team1_stats and 'conference_strength' in team2_stats:
                if team1_stats['conference_strength'] > team2_stats['conference_strength']:
                    predicted_winner = team1_name
                else:
                    predicted_winner = team2_name
                if predicted_winner == actual_winner:
                    correct_predictions['conference_strength'] += 1
            
            # Team experience prediction
            if 'avg_class_year' in team1_stats and 'avg_class_year' in team2_stats:
                if team1_stats['avg_class_year'] > team2_stats['avg_class_year']:
                    predicted_winner = team1_name
                else:
                    predicted_winner = team2_name
                if predicted_winner == actual_winner:
                    correct_predictions['team_experience'] += 1
            
            # Momentum prediction
            if 'recent_win_pct' in team1_stats and 'recent_win_pct' in team2_stats:
                if team1_stats['recent_win_pct'] > team2_stats['recent_win_pct']:
                    predicted_winner = team1_name
                else:
                    predicted_winner = team2_name
                if predicted_winner == actual_winner:
                    correct_predictions['momentum'] += 1
            
            # Win percentage prediction
            if 'W-L%' in team1_stats and 'W-L%' in team2_stats:
                if team1_stats['W-L%'] > team2_stats['W-L%']:
                    predicted_winner = team1_name
                else:
                    predicted_winner = team2_name
                if predicted_winner == actual_winner:
                    correct_predictions['win_pct'] += 1
            
            # SRS prediction
            if 'SRS' in team1_stats and 'SRS' in team2_stats:
                if team1_stats['SRS'] > team2_stats['SRS']:
                    predicted_winner = team1_name
                else:
                    predicted_winner = team2_name
                if predicted_winner == actual_winner:
                    correct_predictions['srs'] += 1
        
        # Calculate accuracies
        total_evaluated_games = len(tournament_data) - skipped_games
        
        if total_evaluated_games > 0:
            if 'conference_strength' in team_stats.columns:
                signal_performance['conference_strength_accuracy'] = correct_predictions['conference_strength'] / total_evaluated_games
                
            if 'avg_class_year' in team_stats.columns:
                signal_performance['team_experience_accuracy'] = correct_predictions['team_experience'] / total_evaluated_games
                
            if 'recent_win_pct' in team_stats.columns:
                signal_performance['momentum_accuracy'] = correct_predictions['momentum'] / total_evaluated_games
                
            if 'W-L%' in team_stats.columns:
                signal_performance['win_pct_accuracy'] = correct_predictions['win_pct'] / total_evaluated_games
                
            if 'SRS' in team_stats.columns:
                signal_performance['srs_accuracy'] = correct_predictions['srs'] / total_evaluated_games
        
        # Print results
        if verbose:
            print(f"\nSignal Performance for {year} Tournament:")
            print(f"Total games evaluated: {total_evaluated_games}")
            print(f"Games skipped: {skipped_games}")
            
            for signal, accuracy in signal_performance.items():
                if accuracy > 0:
                    print(f"{signal}: {accuracy:.2%}")
        
        return signal_performance
    
    def evaluate_multiple_years(self, years=None, verbose=True):
        """
        Evaluate signals across multiple years
        
        Args:
            years: List of years to evaluate, or None to use all available years
            verbose: Whether to print detailed results
            
        Returns:
            DataFrame with performance metrics by year
        """
        if years is None:
            # Find all available tournament data files
            tournament_files = [f for f in os.listdir(self.raw_path) if f.startswith('tournament_results_') and f.endswith('.csv')]
            years = [int(f.split('_')[2].split('.')[0]) for f in tournament_files]
            years.sort()
        
        if not years:
            print("No tournament data files found")
            return pd.DataFrame()
            
        # Evaluate each year
        results = []
        for year in years:
            performance = self.evaluate_signals(year, verbose=verbose)
            if performance:
                performance['year'] = year
                results.append(performance)
                
        if not results:
            print("No results to analyze")
            return pd.DataFrame()
            
        # Combine results
        results_df = pd.DataFrame(results)
        
        # Calculate averages
        avg_row = results_df.drop('year', axis=1).mean().to_dict()
        avg_row['year'] = 'Average'
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Print summary
        if verbose:
            print("\nSummary of Signal Performance Across Years:")
            for col in results_df.columns:
                if col != 'year':
                    avg = results_df[col].mean()
                    print(f"Average {col}: {avg:.2%}")
                    
        return results_df
    
    def visualize_results(self, results_df):
        """
        Visualize the backtesting results
        
        Args:
            results_df: DataFrame with performance metrics by year
        """
        if results_df.empty:
            print("No results to visualize")
            return
            
        # Filter out the 'Average' row for the time series plot
        yearly_results = results_df[results_df['year'] != 'Average'].copy()
        yearly_results['year'] = yearly_results['year'].astype(int)
        yearly_results = yearly_results.sort_values('year')
        
        # Prepare data for plotting
        metrics = [col for col in yearly_results.columns if col != 'year']
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Time series plot
        for metric in metrics:
            axes[0].plot(yearly_results['year'], yearly_results[metric], marker='o', label=metric)
            
        axes[0].set_title('Signal Performance by Year')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Average performance bar chart
        avg_results = results_df[results_df['year'] == 'Average'].iloc[0]
        avg_values = [avg_results[metric] for metric in metrics]
        
        axes[1].bar(metrics, avg_values)
        axes[1].set_title('Average Signal Performance')
        axes[1].set_xlabel('Signal')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(avg_values):
            axes[1].text(i, v + 0.01, f'{v:.2%}', ha='center')
            
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)
            print(f"Created directory: {self.processed_path}")
        
        # Save the figure in multiple formats
        # Save as PNG (good for web/GitHub)
        png_path = os.path.join(self.processed_path, 'backtesting_results.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Results visualization saved as PNG: {png_path}")
        
        # Save as PDF (good for presentations and printing)
        pdf_path = os.path.join(self.processed_path, 'backtesting_results.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Results visualization saved as PDF: {pdf_path}")
        
        # Show the plot
        plt.show()

def main():
    # Create backtester using sample data
    backtester = Backtester(data_dir='data/sample')
    
    # Define all years for which we have sample data
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    
    # Evaluate signals for each year
    for year in years:
        backtester.evaluate_signals(year, verbose=True)
    
    # Evaluate across multiple years
    results = backtester.evaluate_multiple_years(years=years, verbose=True)
    
    # Visualize results
    if not results.empty:
        backtester.visualize_results(results)

if __name__ == "__main__":
    main()
