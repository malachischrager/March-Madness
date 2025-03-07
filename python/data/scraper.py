import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from time import sleep

# Define the raw data directory
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')

def ensure_dir_exists(directory):
    """Ensure the directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def scrape_team_stats(year):
    """Scrape detailed team stats from Sports Reference for a given year"""
    url = f"https://www.sports-reference.com/cbb/seasons/{year}-school-stats.html"
    try:
        # Add delay to be respectful to the server
        sleep(2)
        response = requests.get(url)
        response.raise_for_status()
        
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
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {year}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing data for {year}: {e}")
        return None

def main():
    # Ensure raw data directory exists
    ensure_dir_exists(RAW_DATA_DIR)
    
    # Scrape multiple years
    all_team_stats = []
    for year in range(2010, 2024):  # Adjust year range as needed
        yearly_stats = scrape_team_stats(year)
        if yearly_stats is not None:
            all_team_stats.append(yearly_stats)
    
    if not all_team_stats:
        print("No data was collected. Please check the errors above.")
        return
    
    # Concatenate dataframes
    team_stats_df = pd.concat(all_team_stats, ignore_index=True)
    
    # Save to CSV in raw data directory
    output_file = os.path.join(RAW_DATA_DIR, 'ncaa_team_stats.csv')
    team_stats_df.to_csv(output_file, index=False)
    print(f"\nDetailed team stats scraping complete. CSV file saved to: {output_file}")
    print(f"Total records collected: {len(team_stats_df)}")

if __name__ == '__main__':
    main()
