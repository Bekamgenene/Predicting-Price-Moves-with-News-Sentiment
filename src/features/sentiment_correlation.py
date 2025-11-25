import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from textblob import TextBlob
import os
import re
from datetime import datetime, timedelta
from scipy import stats


def load_data(news_path: str, stock_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load news and stock data from CSV files.
    
    Args:
        news_path: Path to news CSV file
        stock_path: Path to stock CSV file
        
    Returns:
        Tuple containing (news DataFrame, stock DataFrame)
    """
    # Load news data
    news_df = pd.read_csv(news_path)
    
    # Load stock data
    stock_df = pd.read_csv(stock_path)
    
    return news_df, stock_df


def normalize_dates(news_df: pd.DataFrame, stock_df: pd.DataFrame, 
                   news_date_col: str = 'date', 
                   stock_date_col: str = 'Date') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize date formats between news and stock DataFrames.
    
    Args:
        news_df: News DataFrame
        stock_df: Stock DataFrame
        news_date_col: Column name for dates in news DataFrame
        stock_date_col: Column name for dates in stock DataFrame
        
    Returns:
        Tuple of DataFrames with normalized date columns
    """
    # Create copies to avoid modifying originals
    news = news_df.copy()
    stock = stock_df.copy()
    
    news[news_date_col] = pd.to_datetime(news[news_date_col], errors='coerce')
    stock[stock_date_col] = pd.to_datetime(stock[stock_date_col], errors='coerce')
    
    # Drop rows with invalid dates
    news = news.dropna(subset=[news_date_col])
    stock = stock.dropna(subset=[stock_date_col])
    
    # Ensure dates are only the date part (no time)
    news[news_date_col] = news[news_date_col].dt.date
    stock[stock_date_col] = stock[stock_date_col].dt.date
    
    # Convert back to datetime for easier manipulation
    news[news_date_col] = pd.to_datetime(news[news_date_col])
    stock[stock_date_col] = pd.to_datetime(stock[stock_date_col])
    
    return news, stock


def analyze_sentiment(df: pd.DataFrame, text_column: str = 'headline') -> pd.DataFrame:
    """
    Perform sentiment analysis on text data using TextBlob.
    
    Args:
        df: DataFrame containing text data
        text_column: Column containing text to analyze
        
    Returns:
        DataFrame with sentiment analysis results
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Apply sentiment analysis
    result_df['polarity'] = result_df[text_column].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if isinstance(x, str) else 0
    )
    
    result_df['subjectivity'] = result_df[text_column].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity if isinstance(x, str) else 0
    )
    
    # Categorize sentiment
    result_df['sentiment_category'] = result_df['polarity'].apply(
        lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
    )
    
    return result_df


def calculate_daily_returns(stock_df: pd.DataFrame, 
                           date_col: str = 'Date', 
                           close_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate daily returns from stock prices.
    
    Args:
        stock_df: Stock DataFrame with price data
        date_col: Column containing dates
        close_col: Column containing closing prices
        
    Returns:
        DataFrame with daily returns
    """
    # Create a copy
    stock = stock_df.copy()
    
    # Ensure the DataFrame is sorted by date
    stock = stock.sort_values(by=date_col)
    
    # Calculate daily returns
    stock['daily_return'] = stock[close_col].pct_change()
    
    # Remove the first row which will have NaN for return
    stock = stock.dropna(subset=['daily_return'])
    
    return stock


def aggregate_daily_sentiment(news_df: pd.DataFrame, 
                             date_col: str = 'date') -> pd.DataFrame:
    """
    Aggregate sentiment scores by day.
    
    Args:
        news_df: News DataFrame with sentiment scores
        date_col: Column containing dates
        
    Returns:
        DataFrame with aggregated daily sentiment
    """
    # Group by date and calculate mean sentiment
    daily_sentiment = news_df.groupby(date_col).agg({
        'polarity': 'mean',
        'subjectivity': 'mean',
        'headline': 'count'
    }).reset_index()
    
    # Rename columns
    daily_sentiment = daily_sentiment.rename(columns={
        'polarity': 'avg_sentiment',
        'subjectivity': 'avg_subjectivity',
        'headline': 'article_count'
    })
    
    return daily_sentiment


def merge_sentiment_with_returns(sentiment_df: pd.DataFrame, 
                                returns_df: pd.DataFrame,
                                sentiment_date_col: str = 'date',
                                returns_date_col: str = 'Date') -> pd.DataFrame:
    """
    Merge sentiment data with stock returns.
    
    Args:
        sentiment_df: DataFrame with daily sentiment
        returns_df: DataFrame with daily stock returns
        sentiment_date_col: Date column in sentiment DataFrame
        returns_date_col: Date column in returns DataFrame
        
    Returns:
        Merged DataFrame
    """
    # Merge on date
    merged_df = pd.merge(sentiment_df, returns_df[[returns_date_col, 'daily_return']],
                        left_on=sentiment_date_col, right_on=returns_date_col,
                        how='inner')
    
    # Drop duplicate date column if different names were used
    if sentiment_date_col != returns_date_col and returns_date_col in merged_df.columns:
        merged_df = merged_df.drop(columns=[returns_date_col])
    
    return merged_df


def calculate_correlation(merged_df: pd.DataFrame, 
                         sentiment_col: str = 'avg_sentiment',
                         returns_col: str = 'daily_return',
                         method: str = 'pearson') -> Dict:
    """
    Calculate correlation between sentiment and stock returns.
    
    Args:
        merged_df: Merged DataFrame with sentiment and returns
        sentiment_col: Column with sentiment scores
        returns_col: Column with stock returns
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        Dictionary with correlation results
    """
    # Calculate correlation
    correlation, p_value = stats.pearsonr(
        merged_df[sentiment_col], 
        merged_df[returns_col]
    )
    
    # Calculate lagged correlations
    lagged_correlations = []
    for lag in range(1, 6):  # Check lags 1-5 days
        # Sentiment leading returns (past sentiment → future returns)
        lead_corr, lead_p = stats.pearsonr(
            merged_df[sentiment_col].iloc[:-lag].values,
            merged_df[returns_col].iloc[lag:].values
        )
        
        # Returns leading sentiment (past returns → future sentiment)
        lag_corr, lag_p = stats.pearsonr(
            merged_df[returns_col].iloc[:-lag].values,
            merged_df[sentiment_col].iloc[lag:].values
        )
        
        lagged_correlations.append({
            'lag': lag,
            'sentiment_leading_returns_corr': lead_corr,
            'sentiment_leading_returns_p': lead_p,
            'returns_leading_sentiment_corr': lag_corr,
            'returns_leading_sentiment_p': lag_p
        })
    
    # Return results
    result = {
        'correlation': correlation,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': interpret_correlation(correlation, p_value),
        'lagged_correlations': lagged_correlations
    }
    
    return result


def interpret_correlation(correlation: float, p_value: float) -> str:
    """
    Interpret the correlation coefficient and its significance.
    
    Args:
        correlation: Correlation coefficient
        p_value: P-value of the correlation
        
    Returns:
        String interpretation
    """
    strength = ''
    if abs(correlation) < 0.1:
        strength = 'negligible'
    elif abs(correlation) < 0.3:
        strength = 'weak'
    elif abs(correlation) < 0.5:
        strength = 'moderate'
    elif abs(correlation) < 0.7:
        strength = 'strong'
    else:
        strength = 'very strong'
    
    direction = 'positive' if correlation > 0 else 'negative'
    significance = 'statistically significant' if p_value < 0.05 else 'not statistically significant'
    
    interpretation = f"There is a {strength} {direction} correlation ({correlation:.3f}) that is {significance} (p={p_value:.3f})."
    
    if p_value < 0.05:
        if correlation > 0:
            interpretation += " This suggests that positive news sentiment tends to be associated with positive stock returns."
        else:
            interpretation += " This suggests that positive news sentiment tends to be associated with negative stock returns."
    else:
        interpretation += " The data does not provide strong evidence of a relationship between news sentiment and stock returns."
    
    return interpretation


def analyze_sentiment_stock_correlation(news_path: str, stock_path: str, 
                                       news_date_col: str = 'date',
                                       stock_date_col: str = 'Date',
                                       text_col: str = 'headline',
                                       close_col: str = 'Close') -> Dict:
    """
    Full pipeline to analyze correlation between news sentiment and stock returns.
    
    Args:
        news_path: Path to news CSV file
        stock_path: Path to stock CSV file
        news_date_col: Date column in news DataFrame
        stock_date_col: Date column in stock DataFrame
        text_col: Column containing text for sentiment analysis
        close_col: Column containing closing prices
        
    Returns:
        Dictionary with correlation results
    """
    # Load data
    news_df, stock_df = load_data(news_path, stock_path)
    
    # Normalize dates
    news_df, stock_df = normalize_dates(news_df, stock_df, news_date_col, stock_date_col)
    
    # Analyze sentiment
    news_df = analyze_sentiment(news_df, text_col)
    
    # Calculate daily returns
    stock_df = calculate_daily_returns(stock_df, stock_date_col, close_col)
    
    # Aggregate daily sentiment
    daily_sentiment = aggregate_daily_sentiment(news_df, news_date_col)
    
    # Merge sentiment with returns
    merged_df = merge_sentiment_with_returns(daily_sentiment, stock_df, news_date_col, stock_date_col)
    
    # Calculate correlation
    correlation_results = calculate_correlation(merged_df)
    
    # Add merged data to results
    correlation_results['data'] = merged_df
    
    return correlation_results
