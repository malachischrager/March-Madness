import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from time import sleep
import re
from datetime import datetime
import json

# Headers to make our requests more respectful
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; Python/Research/Educational)',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Define data directories with absolute paths for consistency
BASE_DIR = '/Users/malachischrager/Desktop/Github/March Madness'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Print paths for visibility
print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Raw data directory: {RAW_DATA_DIR}")

def ensure_dir_exists(directory):
    """Ensure the directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def scrape_espn_tournament_data(year):
    """
    Scrape NCAA tournament data from ESPN for a given year
    
    ESPN's bracket URLs follow a pattern like:
    https://www.espn.com/mens-college-basketball/tournament/bracket/_/year/2023
    """
    url = f"https://www.espn.com/mens-college-basketball/tournament/bracket/_/year/{year}"
    
    print(f"Attempting to scrape tournament data for {year} from ESPN...")
    
    max_retries = 3
    base_delay = 2  # Start with 2 seconds delay
    
    for attempt in range(max_retries):
        try:
            # Add exponential backoff delay
            sleep_time = base_delay * (2 ** attempt)
            print(f"Waiting {sleep_time} seconds before requesting {year} tournament data...")
            sleep(sleep_time)
            
            response = requests.get(url, headers=HEADERS)
            
            # If we get rate limited, wait and retry
            if response.status_code == 429:
                if attempt < max_retries - 1:  # If we have more retries left
                    print(f"Rate limited, will retry in {sleep_time * 2} seconds...")
                    continue
                else:
                    print(f"Failed to get data for {year} after {max_retries} attempts")
                    return pd.DataFrame()  # Return empty DataFrame
            
            # If page not found, return empty DataFrame
            if response.status_code == 404:
                print(f"Tournament page for {year} not found at {url} (404)")
                return pd.DataFrame()
                
            # If we get here, we have a valid response
            response.raise_for_status()
            
            # Parse the tournament data
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # ESPN's bracket has a specific structure with regions and matchups
            games_data = []
            
            # Find all regions (South, East, West, Midwest)
            regions = soup.find_all('div', class_='region')
            
            if not regions:
                print(f"No regions found in the bracket for {year}")
                # Try to find the bracket in a different format
                bracket = soup.find('div', class_='bracket')
                if not bracket:
                    print(f"No bracket found for {year}")
                    return pd.DataFrame()
            
            # Process each region
            for region in regions:
                region_name = region.find('h4').text.strip() if region.find('h4') else "Unknown"
                
                # Find all matchups in this region
                matchups = region.find_all('div', class_='matchup')
                
                for matchup in matchups:
                    # Find the teams in this matchup
                    teams = matchup.find_all('dt')
                    seeds = matchup.find_all('span', class_='seed')
                    scores = matchup.find_all('span', class_='score')
                    
                    if len(teams) == 2:
                        team1_name = teams[0].text.strip()
                        team2_name = teams[1].text.strip()
                        
                        # Extract seeds (if available)
                        seed1 = seeds[0].text.strip() if len(seeds) > 0 else ''
                        seed2 = seeds[1].text.strip() if len(seeds) > 1 else ''
                        
                        # Extract scores (if available)
                        score1 = scores[0].text.strip() if len(scores) > 0 else ''
                        score2 = scores[1].text.strip() if len(scores) > 1 else ''
                        
                        # Determine the round based on the matchup's position
                        round_name = "Unknown"
                        if "first-round" in matchup.get('class', []):
                            round_name = "First Round"
                        elif "second-round" in matchup.get('class', []):
                            round_name = "Second Round"
                        elif "sweet-16" in matchup.get('class', []):
                            round_name = "Sweet 16"
                        elif "elite-8" in matchup.get('class', []):
                            round_name = "Elite 8"
                        elif "final-four" in matchup.get('class', []):
                            round_name = "Final Four"
                        elif "championship" in matchup.get('class', []):
                            round_name = "Championship"
                        
                        # Determine winner based on scores or CSS classes
                        winner = ""
                        if score1 and score2 and score1.isdigit() and score2.isdigit():
                            winner = team1_name if int(score1) > int(score2) else team2_name
                        elif 'winner' in teams[0].get('class', []):
                            winner = team1_name
                        elif 'winner' in teams[1].get('class', []):
                            winner = team2_name
                        
                        game_data = {
                            'year': year,
                            'region': region_name,
                            'round': round_name,
                            'team1': team1_name,
                            'seed1': seed1,
                            'score1': score1,
                            'team2': team2_name,
                            'seed2': seed2,
                            'score2': score2,
                            'winner': winner
                        }
                        games_data.append(game_data)
            
            # If we didn't find any games using the region approach, try an alternative approach
            if not games_data:
                print(f"Trying alternative approach for {year}...")
                
                # Look for game containers
                game_containers = soup.find_all('div', class_='game-container')
                
                for container in game_containers:
                    teams = container.find_all('span', class_='team-name')
                    seeds = container.find_all('span', class_='seed')
                    scores = container.find_all('span', class_='score')
                    
                    if len(teams) == 2:
                        team1_name = teams[0].text.strip()
                        team2_name = teams[1].text.strip()
                        
                        seed1 = seeds[0].text.strip() if len(seeds) > 0 else ''
                        seed2 = seeds[1].text.strip() if len(seeds) > 1 else ''
                        
                        score1 = scores[0].text.strip() if len(scores) > 0 else ''
                        score2 = scores[1].text.strip() if len(scores) > 1 else ''
                        
                        # Try to determine the round and region from container classes or parent elements
                        round_name = "Unknown"
                        region_name = "Unknown"
                        
                        # Determine winner
                        winner = ""
                        if score1 and score2 and score1.isdigit() and score2.isdigit():
                            winner = team1_name if int(score1) > int(score2) else team2_name
                        
                        game_data = {
                            'year': year,
                            'region': region_name,
                            'round': round_name,
                            'team1': team1_name,
                            'seed1': seed1,
                            'score1': score1,
                            'team2': team2_name,
                            'seed2': seed2,
                            'score2': score2,
                            'winner': winner
                        }
                        games_data.append(game_data)
            
            # If we still don't have any games, try to use the ESPN API approach
            if not games_data:
                print(f"Trying ESPN API approach for {year}...")
                # This would involve finding the tournament ID and making API calls
                # For now, we'll return an empty DataFrame as this would require more complex logic
            
            # Convert to DataFrame
            if games_data:
                df = pd.DataFrame(games_data)
                print(f"Successfully scraped {len(df)} games for {year}")
                return df
            else:
                print(f"No games found for {year}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error scraping tournament data for {year}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
            else:
                print(f"Failed to scrape tournament data for {year} after {max_retries} attempts")
                return pd.DataFrame()

def main():
    # Ensure raw data directory exists
    ensure_dir_exists(RAW_DATA_DIR)
    
    # Scrape tournament data for recent years
    all_tournament_data = []
    
    # Focus on years 2010-2023
    for year in range(2010, 2024):
        print(f"\nProcessing year {year}...")
        tournament_data = scrape_espn_tournament_data(year)
        
        if tournament_data is not None and not tournament_data.empty:
            # Save tournament data for each year separately
            year_file = os.path.join(RAW_DATA_DIR, f'tournament_results_{year}.csv')
            tournament_data.to_csv(year_file, index=False)
            print(f"Tournament data for {year} saved to: {year_file}")
            print(f"Games for {year}: {len(tournament_data)}")
            all_tournament_data.append(tournament_data)
        else:
            # Create an empty file with headers for consistency
            year_file = os.path.join(RAW_DATA_DIR, f'tournament_results_{year}.csv')
            pd.DataFrame({
                'year': [], 'region': [], 'round': [], 
                'team1': [], 'seed1': [], 'score1': [], 
                'team2': [], 'seed2': [], 'score2': [], 
                'winner': []
            }).to_csv(year_file, index=False)
            print(f"Created empty tournament file for {year}: {year_file}")
    
    if all_tournament_data and any(not df.empty for df in all_tournament_data):
        # Concatenate tournament dataframes
        tournament_df = pd.concat(all_tournament_data, ignore_index=True)
        # Save combined tournament data to CSV
        tournament_file = os.path.join(RAW_DATA_DIR, 'ncaa_tournament_results.csv')
        tournament_df.to_csv(tournament_file, index=False)
        print(f"\nTournament data complete. CSV file saved to: {tournament_file}")
        print(f"Total tournament games collected: {len(tournament_df)}")
    else:
        print("\nNo tournament data was collected. Please check the errors above.")

if __name__ == "__main__":
    main()
