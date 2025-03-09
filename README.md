# 🏀 March Madness Prediction System

A sophisticated quantitative analysis and prediction system for NCAA March Madness tournaments, applying factor investing principles and machine learning techniques to college basketball data.

## 📊 Quantitative Signals

### Core Predictive Factors

1. **Conference Strength** 📈
   - Win percentage aggregation
   - Simple Rating System (SRS)
   - Strength of Schedule (SOS)
   - Scoring metrics
   - Point differential analysis

2. **Team Experience** 👥
   - Player class distribution (FR/SO/JR/SR)
   - Games played metrics
   - Minutes played analysis
   - Team cohesion indicators

3. **Momentum Factors** 🔥
   - Recent win percentage trends
   - Point differential momentum
   - Performance trajectory analysis
   - Pre-tournament form

### Coming Soon
- Coach experience metrics
- Historical tournament performance
- Player efficiency indicators
- Non-conference schedule strength

## 🔬 Backtesting Framework

### Signal Evaluation
- Individual signal performance tracking
- Year-by-year analysis (2010-2024)
- Tournament game prediction accuracy
- Signal combination optimization

### Performance Metrics
- Prediction accuracy by round
- Signal reliability scores
- Historical effectiveness trends
- Cross-validation results

## 🛠 Tech Stack

### Data Pipeline
- **Python**: Data analysis, machine learning, and backtesting
  - pandas: Data processing
  - scikit-learn: Statistical analysis
  - BeautifulSoup4: Data collection

### API Layer
- **Go**: High-performance web server
  - Fast data processing
  - RESTful API endpoints
  - Real-time predictions

### Storage & Frontend
- **Database**: PostgreSQL
- **Frontend**: React with TypeScript

## 📁 Project Structure

```bash
├── data/               # Data storage
│   ├── raw/           # Raw scraped data
│   └── processed/     # Processed datasets
├── python/
│   ├── analysis/      # Factor models
│   ├── backtesting/   # Testing framework
│   └── data/          # Data processing
├── go/
│   ├── api/          # REST endpoints
│   ├── models/       # Data structures
│   └── services/     # Business logic
└── web/              # Frontend app
```

## 🚀 Getting Started

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix
   pip install -r requirements.txt
   ```

2. **Data Collection**
   ```bash
   cd python/data
   python scraper.py  # Collect historical data
   ```

3. **Signal Processing**
   ```bash
   python processor.py  # Generate signals
   ```

4. **View Results**
   - Check `data/processed/backtest_results.csv` for signal performance
   - Analyze prediction accuracy by year and signal

## 📈 Performance Tracking

Track the performance of our predictive signals in the `backtest_results.csv` file, which includes:
- Signal accuracy by tournament year
- Comparative signal effectiveness
- Combined signal performance
- Historical trend analysis

## 🤝 Contributing

Interested in contributing? We welcome:
- Additional predictive signals
- Backtesting improvements
- Data source integrations
- UI/UX enhancements

## 📝 License

MIT License - Feel free to use and modify for your own March Madness analysis!
