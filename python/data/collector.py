import requests
import pandas as pd
from datetime import datetime
import os
import json
import time
from typing import List, Dict, Any

class BasketballDataCollector:
    def __init__(self):
        # Use absolute paths to top-level data directory
        self.base_path = '/Users/malachischrager/Desktop/Github/March Madness'
        self.data_path = os.path.join(self.base_path, 'data')  # Root data directory
        self.raw_path = os.path.join(self.data_path, 'raw')
        self.processed_path = os.path.join(self.data_path, 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        
        # Print paths for debugging
        print(f"Data directory: {self.data_path}")
        print(f"Raw data path: {self.raw_path}")
        print(f"Processed data path: {self.processed_path}")
        
        # ESPN API endpoints
        self.espn_api_base = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        
        # Headers to mimic browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _make_request(self, url: str) -> Dict:
        """Make a request with proper delay and headers"""
        time.sleep(2)  # Be nice to the server
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def collect_team_stats(self, year: int = 2024) -> pd.DataFrame:
        """
        Collect team statistics from ESPN
        """
        print(f"Collecting team stats for {year}...")
        
        # Get all teams first
        teams_url = f"{self.espn_api_base}/teams"
        teams_data = []
        
        try:
            teams_response = self._make_request(teams_url)
            
            if 'sports' in teams_response:
                for sport in teams_response['sports']:
                    if 'leagues' in sport:
                        for league in sport['leagues']:
                            if 'teams' in league:
                                for team in league['teams']:
                                    team_info = team['team']
                                    team_data = {
                                        'id': team_info['id'],
                                        'name': team_info['name'],
                                        'abbreviation': team_info.get('abbreviation', ''),
                                        'location': team_info.get('location', ''),
                                        'year': year
                                    }
                                    
                                    # Get team stats
                                    try:
                                        stats_url = f"{self.espn_api_base}/teams/{team_info['id']}/statistics"
                                        stats_response = self._make_request(stats_url)
                                        
                                        if 'splits' in stats_response:
                                            for category in stats_response['splits']:
                                                for stat in category['stats']:
                                                    name = stat['name'].lower().replace(' ', '_')
                                                    value = stat['value']
                                                    team_data[name] = value
                                    except:
                                        print(f"Could not fetch stats for team {team_info['name']}")
                                    
                                    teams_data.append(team_data)
            
            df = pd.DataFrame(teams_data)
            
            # Save raw data
            self.save_data(df, f'team_stats_{year}.csv')
            print(f"Saved stats for {len(teams_data)} teams")
            
            return df
            
        except Exception as e:
            print(f"Error collecting team stats: {e}")
            return pd.DataFrame()

    def collect_tournament_data(self, year: int = 2023) -> pd.DataFrame:
        """
        Collect tournament data from ESPN's bracket API
        """
        print(f"Collecting tournament data for {year}...")
        
        # Tournament bracket endpoint
        url = f"{self.espn_api_base}/tournaments/ncaa-mens-basketball-championship/{year}/bracket"
        
        try:
            response = self._make_request(url)
            games_data = []
            
            if 'bracketRegions' in response:
                for region in response['bracketRegions']:
                    region_name = region['name']
                    if 'slots' in region:
                        for slot in region['slots']:
                            if 'competitors' in slot and len(slot['competitors']) == 2:
                                game_data = {
                                    'year': year,
                                    'region': region_name,
                                    'round': slot.get('round', {}).get('roundNumber', ''),
                                    'team1': slot['competitors'][0]['team']['name'],
                                    'seed1': slot['competitors'][0].get('seedNumber', ''),
                                    'score1': slot['competitors'][0].get('score', ''),
                                    'team2': slot['competitors'][1]['team']['name'],
                                    'seed2': slot['competitors'][1].get('seedNumber', ''),
                                    'score2': slot['competitors'][1].get('score', ''),
                                    'winner': slot['competitors'][0]['team']['name'] 
                                             if slot['competitors'][0].get('winner', False) 
                                             else slot['competitors'][1]['team']['name']
                                             if slot['competitors'][1].get('winner', False)
                                             else ''
                                }
                                games_data.append(game_data)
            
            # Also get Final Four and Championship games
            if 'finalFourSlots' in response:
                for slot in response['finalFourSlots']:
                    if 'competitors' in slot and len(slot['competitors']) == 2:
                        game_data = {
                            'year': year,
                            'region': 'Final Four',
                            'round': slot.get('round', {}).get('roundNumber', ''),
                            'team1': slot['competitors'][0]['team']['name'],
                            'seed1': slot['competitors'][0].get('seedNumber', ''),
                            'score1': slot['competitors'][0].get('score', ''),
                            'team2': slot['competitors'][1]['team']['name'],
                            'seed2': slot['competitors'][1].get('seedNumber', ''),
                            'score2': slot['competitors'][1].get('score', ''),
                            'winner': slot['competitors'][0]['team']['name'] 
                                     if slot['competitors'][0].get('winner', False) 
                                     else slot['competitors'][1]['team']['name']
                                     if slot['competitors'][1].get('winner', False)
                                     else ''
                        }
                        games_data.append(game_data)
            
            if 'championshipSlot' in response:
                slot = response['championshipSlot']
                if 'competitors' in slot and len(slot['competitors']) == 2:
                    game_data = {
                        'year': year,
                        'region': 'Championship',
                        'round': slot.get('round', {}).get('roundNumber', ''),
                        'team1': slot['competitors'][0]['team']['name'],
                        'seed1': slot['competitors'][0].get('seedNumber', ''),
                        'score1': slot['competitors'][0].get('score', ''),
                        'team2': slot['competitors'][1]['team']['name'],
                        'seed2': slot['competitors'][1].get('seedNumber', ''),
                        'score2': slot['competitors'][1].get('score', ''),
                        'winner': slot['competitors'][0]['team']['name'] 
                                 if slot['competitors'][0].get('winner', False) 
                                 else slot['competitors'][1]['team']['name']
                                 if slot['competitors'][1].get('winner', False)
                                 else ''
                    }
                    games_data.append(game_data)
            
            df = pd.DataFrame(games_data)
            
            # Save raw data
            self.save_data(df, f'tournament_results_{year}.csv')
            print(f"Saved {len(games_data)} tournament games")
            
            return df
            
        except Exception as e:
            print(f"Error collecting tournament data: {e}")
            return pd.DataFrame()

    def collect_player_stats(self, year: int = 2024) -> pd.DataFrame:
        """
        Collect player statistics from ESPN
        """
        print(f"Collecting player stats for {year}...")
        
        # Get all teams first to iterate through their rosters
        teams_url = f"{self.espn_api_base}/teams"
        players_data = []
        
        try:
            teams_response = self._make_request(teams_url)
            
            if 'sports' in teams_response:
                for sport in teams_response['sports']:
                    if 'leagues' in sport:
                        for league in sport['leagues']:
                            if 'teams' in league:
                                for team in league['teams']:
                                    team_info = team['team']
                                    
                                    # Get team roster and stats
                                    try:
                                        roster_url = f"{self.espn_api_base}/teams/{team_info['id']}/roster"
                                        roster_response = self._make_request(roster_url)
                                        
                                        if 'athletes' in roster_response:
                                            for athlete in roster_response['athletes']:
                                                player_data = {
                                                    'id': athlete['id'],
                                                    'name': athlete['fullName'],
                                                    'team': team_info['name'],
                                                    'position': athlete.get('position', {}).get('name', ''),
                                                    'jersey': athlete.get('jersey', ''),
                                                    'year': year
                                                }
                                                
                                                # Get player stats
                                                try:
                                                    stats_url = f"{self.espn_api_base}/athletes/{athlete['id']}/statistics"
                                                    stats_response = self._make_request(stats_url)
                                                    
                                                    if 'splits' in stats_response:
                                                        for category in stats_response['splits']:
                                                            for stat in category['stats']:
                                                                name = stat['name'].lower().replace(' ', '_')
                                                                value = stat['value']
                                                                player_data[name] = value
                                                except:
                                                    print(f"Could not fetch stats for player {athlete['fullName']}")
                                                
                                                players_data.append(player_data)
                                    except:
                                        print(f"Could not fetch roster for team {team_info['name']}")
            
            df = pd.DataFrame(players_data)
            
            # Save raw data
            self.save_data(df, f'player_stats_{year}.csv')
            print(f"Saved stats for {len(players_data)} players")
            
            return df
            
        except Exception as e:
            print(f"Error collecting player stats: {e}")
            return pd.DataFrame()

    def save_data(self, data: Any, filename: str) -> None:
        """
        Save collected data to the raw data directory
        """
        filepath = os.path.join(self.raw_path, filename)
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f)
        print(f"Saved data to {filepath}")

def main():
    collector = BasketballDataCollector()
    
    # Collect current season data
    team_stats = collector.collect_team_stats()
    player_stats = collector.collect_player_stats()
    
    # Collect tournament data from last year
    tournament_data = collector.collect_tournament_data(2023)
    
    print("Data collection complete!")

if __name__ == "__main__":
    main()
