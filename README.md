# Predicting Price Moves with News Sentiment

A comprehensive project for analyzing financial news sentiment and its correlation with stock price movements.

## Overview

This project focuses on analyzing a large corpus of financial news data to discover correlations between news sentiment and stock market movements. The analysis includes sentiment analysis of news headlines and correlation analysis between sentiment and stock price fluctuations.

## Business Objective

Enhance predictive analytics capabilities to boost financial forecasting accuracy and operational efficiency through advanced data analysis. The project has two main focuses:

1. **Sentiment Analysis**: Quantifying the tone and sentiment in financial news headlines using NLP techniques.
2. **Correlation Analysis**: Establishing statistical correlations between news sentiment and corresponding stock price movements.

## Methodology

### Data Processing Pipeline

1. **Data Collection**: Historical stock price data and financial news headlines are collected and preprocessed
2. **Data Cleaning**: Removal of duplicates, handling missing values, and normalization of dates
3. **Sentiment Analysis**: 
   - TextBlob is used to extract sentiment polarity (-1 to 1) and subjectivity (0 to 1)
   - Headlines are categorized as positive, neutral, or negative based on polarity thresholds
4. **Feature Engineering**:
   - Daily stock returns calculation using percentage change in closing prices
   - Aggregation of multiple daily headlines into a single sentiment score
   - Creation of lagged features to detect time-delayed effects
5. **Correlation Analysis**:
   - Pearson correlation between sentiment metrics and stock returns
   - Statistical significance testing with p-values
   - Analysis of correlations with various time lags (0-5 days)

### Statistical Approach

The project uses a rigorous statistical approach to ensure reliable results:

- **Null Hypothesis**: News sentiment has no correlation with stock price movements
- **Alternative Hypothesis**: News sentiment is correlated with stock price movements
- **Significance Level**: Alpha = 0.05 for hypothesis testing
- **Validation**: Cross-validation techniques to avoid overfitting

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
│                
│   ├── features/            # Feature engineering
│   ├── models/              # Model training and evaluation
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

## Features

### Task 3: News Sentiment and Stock Correlation Analysis

A complete implementation of the correlation analysis between news headline sentiment and stock price movements. This feature includes:

- **Date Alignment**: Normalizes dates between news and stock datasets for accurate correlation
- **Sentiment Analysis**: Uses TextBlob to analyze sentiment in financial headlines
- **Stock Returns Calculation**: Computes daily percentage changes in stock closing prices
- **Aggregation**: Combines multiple headlines per day into a single sentiment score
- **Correlation Analysis**: Calculates Pearson correlation coefficient between sentiment and stock returns
- **Lagged Analysis**: Examines time-delayed relationships between sentiment and price movements
- **Visualization**: Provides comprehensive visualizations for analyzing the sentiment-price relationship

### Key Modules

- `src/features/sentiment_correlation.py`: Core implementation for sentiment analysis and correlation
- `src/visualization/sentiment_visualization.py`: Visualization tools for sentiment-price relationships
- `src/examples/sentiment_correlation_example.py`: Example script demonstrating the full analysis pipeline

## Usage

### Running the Sentiment Correlation Analysis

```bash
# Navigate to the project directory
cd Predicting-Price-Moves-with-News-Sentiment

# Run the example script
python src/examples/sentiment_correlation_example.py
```

This will:
1. Load news headlines and stock price data
2. Perform sentiment analysis on headlines
3. Calculate daily stock returns
4. Determine correlation between sentiment and returns
5. Generate visualizations in the output directory

Detailed instructions for running other analyses are provided in the notebooks directory.

## Testing

The project includes a comprehensive testing suite:

```bash
# Run all tests
python -m pytest

# Run with coverage report
python -m pytest --cov=src
```

Tests are organized by module and include:
- Unit tests for individual functions
- Integration tests for component interactions
- Regression tests to ensure consistency

The CI/CD pipeline automatically runs tests on every push and pull request.

## Development Workflow

### Branching Strategy

- `main`: Production-ready code only
- `develop`: Integration branch for feature development
- `feature/*`: Individual feature branches
- `bugfix/*`: Bug fix branches
- `release/*`: Release preparation branches

### Commit Message Guidelines

The project follows conventional commit message format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types include:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or modifying tests
- `refactor`: Code changes that neither fix bugs nor add features
- `style`: Changes that don't affect code meaning (formatting, etc.)
- `chore`: Changes to build process, dependencies, etc.

Example:
```
feat(sentiment): implement daily sentiment aggregation

- Add functionality to aggregate multiple headlines per day
- Include article count and average sentiment score calculation

Closes #123
```

### Pull Request Process

1. Create a branch from `develop` following the naming convention
2. Make changes and commit following the commit message guidelines
3. Push branch and create a pull request to `develop`
4. Ensure tests pass and code meets quality standards
5. Get at least one code review approval
6. Merge with squash if multiple commits are present

## Contributors

Bekam Genene

## License

MIT