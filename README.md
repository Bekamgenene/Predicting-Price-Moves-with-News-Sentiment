# Predicting-Price-Moves-with-News-Sentiment
A comprehensive project for analyzing financial news sentiment and its correlation with stock price movements.

## Overview

This project focuses on analyzing a large corpus of financial news data to discover correlations between news sentiment and stock market movements. The analysis includes sentiment analysis of news headlines and correlation analysis between sentiment and stock price fluctuations.

## Business Objective

Enhance predictive analytics capabilities to boost financial forecasting accuracy and operational efficiency through advanced data analysis. The project has two main focuses:

1. **Sentiment Analysis**: Quantifying the tone and sentiment in financial news headlines using NLP techniques.
2. **Correlation Analysis**: Establishing statistical correlations between news sentiment and corresponding stock price movements.

## Dataset

The Financial News and Stock Price Integration Dataset (FNSPID) includes:

- **headline**: Article release headline/title
- **url**: Direct link to the full news article
- **publisher**: Author/creator of article
- **date**: Publication date and time (UTC-4 timezone)
- **stock**: Stock ticker symbol (e.g., AAPL for Apple)

## Project Structure

```
├── .vscode/
│   └── settings.json        # VS Code settings
├── .github/
│   └── workflows
│       ├── unittests.yml    # CI/CD pipeline configuration
├── .gitignore               # Git ignore file
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── src/                     # Source code
│   ├── __init__.py
│   ├── data/                # Data processing modules
│   ├── features/            # Feature engineering
│   └── visualization/       # Data visualization
├── notebooks/               # Jupyter notebooks
│   ├── __init__.py
│   └── README.md            # Notebook documentation
├── tests/                   # Unit tests
│   ├── __init__.py
└── scripts/                 # Utility scripts
    ├── __init__.py
```

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/Bekamgenene/Predicting-Price-Moves-with-News-Sentiment.git
cd Predicting-Price-Moves-with-News-Sentiment
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Detailed instructions for running the analysis will be provided in the notebooks directory.

## Contributors

Bekam Genene

## License