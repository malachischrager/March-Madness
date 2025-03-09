import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from time import sleep

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

def scrape_tournament_data(year):
    """Scrape NCAA tournament data from Sports Reference for a given year"""
    # Try both the modern format and the older format URLs
    urls = [
        f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html",  # Modern format
        f"https://www.sports-reference.com/cbb/postseason/{year}-ncaa-tournament.html"  # Older format
    ]
    
    max_retries = 3
    base_delay = 5  # Start with 5 seconds delay
    
    # Try each URL format
    for url in urls:
        for attempt in range(max_retries):
            try:
                # Add exponential backoff delay
                sleep_time = base_delay * (2 ** attempt)
                print(f"Waiting {sleep_time} seconds before requesting {year} tournament data from {url}...")
                sleep(sleep_time)
                
                response = requests.get(url, headers=HEADERS)
                
                # If we get rate limited, wait and retry
                if response.status_code == 429:
                    if attempt < max_retries - 1:  # If we have more retries left
                        print(f"Rate limited, will retry in {sleep_time * 2} seconds...")
                        continue
                    else:
                        print(f"Failed to get data for {year} after {max_retries} attempts")
                        break  # Try next URL
                
                # If page not found, try next URL
                if response.status_code == 404:
                    print(f"Tournament page for {year} not found at {url} (404)")
                    break  # Try next URL
                    
                # If we get here, we have a valid response
                response.raise_for_status()
                
                # Parse the tournament data
                soup = BeautifulSoup(response.text, 'html.parser')
                games_data = []
                
                # Method 1: Look for bracket structure (modern format)
                brackets = soup.find_all('div', class_='bracket')
                if brackets:
                    print(f"Found bracket format for {year}")
                    for bracket in brackets:
                        region = bracket.find('h2').text.strip() if bracket.find('h2') else 'Unknown'
                        games = bracket.find_all('div', class_='game')
                        
                        for game in games:
                            teams = game.find_all('div', class_='team')
                            if len(teams) == 2:
                                try:
                                    team1_name = teams[0].find('span', class_='name').text.strip()
                                    team2_name = teams[1].find('span', class_='name').text.strip()
                                    
                                    # Get seeds if available
                                    seed1 = teams[0].find('span', class_='seed')
                                    seed1 = seed1.text.strip() if seed1 else ''
                                    
                                    seed2 = teams[1].find('span', class_='seed')
                                    seed2 = seed2.text.strip() if seed2 else ''
                                    
                                    # Get scores if available
                                    score1 = teams[0].find('span', class_='score')
                                    score1 = score1.text.strip() if score1 else ''
                                    
                                    score2 = teams[1].find('span', class_='score')
                                    score2 = score2.text.strip() if score2 else ''
                                    
                                    # Determine round from data attribute or position
                                    round_name = game.get('data-round', 'Unknown')
                                    
                                    # Determine winner based on scores
                                    winner = ''
                                    if score1 and score2 and score1.isdigit() and score2.isdigit():
                                        winner = team1_name if int(score1) > int(score2) else team2_name
                                    
                                    game_data = {
                                        'year': year,
                                        'region': region,
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
                                except (AttributeError, ValueError) as e:
                                    print(f"Error parsing game: {e}")
                                    continue
                
                # Method 2: Look for tournament games table (older format)
                if not games_data:
                    print(f"Trying table format for {year}")
                    # Look for tables with game results
                    tables = soup.find_all('table', class_='teams')
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            # Skip header rows
                            if row.find('th'):
                                continue
                                
                            cells = row.find_all('td')
                            if len(cells) >= 3:  # Need at least winner, score, loser
                                try:
                                    # Format might vary, but typically: Winner, Score, Loser
                                    team1 = cells[0].text.strip()
                                    score_text = cells[1].text.strip()
                                    team2 = cells[2].text.strip()
                                    
                                    # Parse score (format might be like "80-70")
                                    scores = score_text.split('-')
                                    score1 = scores[0].strip() if len(scores) > 0 else ''
                                    score2 = scores[1].strip() if len(scores) > 1 else ''
                                    
                                    # Try to find round information
                                    round_name = 'Unknown'
                                    if len(cells) > 3:
                                        round_name = cells[3].text.strip()
                                    
                                    # Extract seeds if they're in the team names (e.g., "(1) Duke")
                                    import re
                                    seed1_match = re.search(r'\(([0-9]+)\)', team1)
                                    seed1 = seed1_match.group(1) if seed1_match else ''
                                    
                                    seed2_match = re.search(r'\(([0-9]+)\)', team2)
                                    seed2 = seed2_match.group(1) if seed2_match else ''
                                    
                                    # Clean team names to remove seed info
                                    team1 = re.sub(r'\([0-9]+\)', '', team1).strip()
                                    team2 = re.sub(r'\([0-9]+\)', '', team2).strip()
                                    
                                    game_data = {
                                        'year': year,
                                        'region': 'Unknown',  # Region info might not be available
                                        'round': round_name,
                                        'team1': team1,
                                        'seed1': seed1,
                                        'score1': score1,
                                        'team2': team2,
                                        'seed2': seed2,
                                        'score2': score2,
                                        'winner': team1  # Winner is typically listed first
                                    }
                                    games_data.append(game_data)
                                except (IndexError, ValueError) as e:
                                    print(f"Error parsing game row: {e}")
                                    continue
                
                # Method 3: If we still don't have games, try to get tournament teams
                if not games_data:
                    # Try to get a list of tournament teams from the season page
                    season_url = f"https://www.sports-reference.com/cbb/seasons/{year}.html"
                    print(f"Trying to find tournament teams at {season_url}")
                    
                    try:
                        # Wait before making another request
                        sleep(base_delay)
                        season_response = requests.get(season_url, headers=HEADERS)
                        season_response.raise_for_status()
                        
                        season_soup = BeautifulSoup(season_response.text, 'html.parser')
                        
                        # Look for the NCAA Tournament section
                        tournament_section = None
                        for h2 in season_soup.find_all('h2'):
                            if 'NCAA Tournament' in h2.text:
                                tournament_section = h2.find_next('div')
                                break
                        
                        if tournament_section:
                            # Find all teams listed in the tournament section
                            team_links = tournament_section.find_all('a')
                            tournament_teams = []
                            
                            for link in team_links:
                                if '/cbb/schools/' in link.get('href', ''):
                                    team_name = link.text.strip()
                                    tournament_teams.append(team_name)
                            
                            if tournament_teams:
                                print(f"Found {len(tournament_teams)} tournament teams for {year}")
                                
                                # Create a simple dataframe with just the team names
                                teams_df = pd.DataFrame({
                                    'team': tournament_teams,
                                    'year': year
                                })
                                
                                return teams_df
                    except Exception as e:
                        print(f"Error getting tournament teams: {e}")
                
                # If we found games, return them
                if games_data:
                    print(f"Successfully scraped {len(games_data)} tournament games for {year}")
                    return pd.DataFrame(games_data)
                    
                # If we got here with a valid response but no games, try the next URL
                print(f"No tournament games found at {url} for {year}")
                break
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    continue
                else:
                    print(f"Failed to get tournament data for {year} from {url}: {e}")
                    break  # Try next URL
    
    # If we've tried all URLs and still have no data, return an empty DataFrame
    print(f"Could not find tournament data for {year} after trying all sources")
    return pd.DataFrame({
        'year': [year], 'team1': [''], 'team2': [''], 'score1': [''], 'score2': [''], 'round': [''], 'winner': ['']
    })

def scrape_team_stats(year):
    """Scrape detailed team stats from Sports Reference for a given year"""
    url = f"https://www.sports-reference.com/cbb/seasons/{year}-school-stats.html"
    max_retries = 3
    base_delay = 5  # Start with 5 seconds delay
    
    for attempt in range(max_retries):
        try:
            # Add exponential backoff delay
            sleep_time = base_delay * (2 ** attempt)
            print(f"Waiting {sleep_time} seconds before requesting {year} team stats...")
            sleep(sleep_time)
            
            response = requests.get(url, headers=HEADERS)
            
            # If we get rate limited, wait and retry
            if response.status_code == 429:
                if attempt < max_retries - 1:  # If we have more retries left
                    print(f"Rate limited, will retry in {sleep_time * 2} seconds...")
                    continue
                else:
                    print(f"Failed to get data for {year} after {max_retries} attempts")
                    return None
            
            response.raise_for_status()
            break  # If we get here, we got a good response
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Error on attempt {attempt + 1}: {e}")
                continue
            raise  # Re-raise the last exception if we're out of retries
    
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='basic_school_stats')
        
        if table is None:
            print(f"Warning: No data table found for year {year}")
            return None
            
        df = pd.read_html(str(table))[0]
        
        # Clean DataFrame
        df.columns = df.columns.droplevel(0)
        df = df.rename(columns={'School': 'Team'})
        df = df[df['Team'] != 'School']  # Remove repeated headers
        df['Year'] = year
        
        print(f"Successfully scraped data for {year}")
        return df
        
    except Exception as e:
        print(f"Unexpected error processing data for {year}: {e}")
        return None

def scrape_player_stats(year):
    """Scrape individual player statistics from Sports Reference for a given year"""
    url = f"https://www.sports-reference.com/cbb/seasons/{year}-advanced.html"
    max_retries = 3
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            sleep_time = base_delay * (2 ** attempt)
            print(f"Waiting {sleep_time} seconds before requesting {year} player stats...")
            sleep(sleep_time)
            
            response = requests.get(url, headers=HEADERS)
            
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    print(f"Rate limited, will retry in {sleep_time * 2} seconds...")
                    continue
                else:
                    print(f"Failed to get player data for {year} after {max_retries} attempts")
                    return None
            
            response.raise_for_status()
            break
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Error on attempt {attempt + 1}: {e}")
                continue
            raise
    
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='adv_stats')
        
        if table is None:
            print(f"Warning: No player data table found for year {year}")
            return None
            
        df = pd.read_html(str(table))[0]
        
        # Clean DataFrame
        df = df[df['Player'] != 'Player']  # Remove repeated headers
        df['Year'] = year
        
        print(f"Successfully scraped player data for {year}")
        return df
        
    except Exception as e:
        print(f"Unexpected error processing player data for {year}: {e}")
        return None

def main():
    # Ensure raw data directory exists
    ensure_dir_exists(RAW_DATA_DIR)
    
    # Scrape team stats
    print("\nCollecting team statistics...")
    all_team_stats = []
    for year in range(2010, 2024):  # Adjust year range as needed
        yearly_stats = scrape_team_stats(year)
        if yearly_stats is not None:
            all_team_stats.append(yearly_stats)
    
    if all_team_stats:
        # Concatenate team stats dataframes
        team_stats_df = pd.concat(all_team_stats, ignore_index=True)
        # Save team stats to CSV
        team_stats_file = os.path.join(RAW_DATA_DIR, 'ncaa_team_stats.csv')
        team_stats_df.to_csv(team_stats_file, index=False)
        print(f"\nTeam stats complete. CSV file saved to: {team_stats_file}")
        print(f"Total team records collected: {len(team_stats_df)}")
    else:
        print("\nNo team stats were collected. Please check the errors above.")
    
    # Scrape tournament data
    print("\nCollecting tournament data...")
    all_tournament_data = []
    for year in range(2010, 2024):  # Adjust year range as needed
        tournament_data = scrape_tournament_data(year)
        if tournament_data is not None:
            # Add year to the dataset
            if 'year' not in tournament_data.columns:
                tournament_data['year'] = year
                
            # Save tournament data for each year separately
            if not tournament_data.empty:
                year_file = os.path.join(RAW_DATA_DIR, f'tournament_results_{year}.csv')
                tournament_data.to_csv(year_file, index=False)
                print(f"Tournament data for {year} saved to: {year_file}")
                print(f"Games for {year}: {len(tournament_data)}")
            else:
                # Create an empty file with headers for consistency
                year_file = os.path.join(RAW_DATA_DIR, f'tournament_results_{year}.csv')
                pd.DataFrame({
                    'team1': [], 'team2': [], 'score1': [], 'score2': [], 'round': [], 'year': []
                }).to_csv(year_file, index=False)
                print(f"Created empty tournament file for {year}: {year_file}")
                
            all_tournament_data.append(tournament_data)
    
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
    
    # Skipping player stats collection as requested
    print("\nSkipping player statistics collection as requested.")

if __name__ == '__main__':
    main()
