import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def resample_news_frequency(df: pd.DataFrame, date_column: str = 'date',
                          freq: str = 'D') -> pd.DataFrame:
    """
    Resample news frequency at specified intervals.
    
    Args:
        df: DataFrame containing date information
        date_column: Column containing dates
        freq: Frequency for resampling ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
        DataFrame with resampled frequencies
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
    
    # Set date as index
    df_indexed = df.set_index(date_column)
    
    # Resample and count
    resampled = df_indexed.resample(freq).size().reset_index()
    resampled.columns = [date_column, 'article_count']
    
    return resampled


def detect_news_spikes(df: pd.DataFrame, date_column: str = 'date',
                     freq: str = 'D', threshold: float = 2.0,
                     window: int = 30) -> pd.DataFrame:
    """
    Detect abnormal spikes in news publication frequency.
    
    Args:
        df: DataFrame containing date information
        date_column: Column containing dates
        freq: Frequency for resampling
        threshold: Number of standard deviations above rolling mean to consider a spike
        window: Rolling window size for baseline calculation
        
    Returns:
        DataFrame with dates and magnitudes of detected spikes
    """
    # Resample to get frequency
    resampled = resample_news_frequency(df, date_column, freq)
    
    # Calculate rolling statistics
    resampled['rolling_mean'] = resampled['article_count'].rolling(window=window, center=True).mean()
    resampled['rolling_std'] = resampled['article_count'].rolling(window=window, center=True).std()
    
    # Calculate z-scores
    resampled['z_score'] = (resampled['article_count'] - resampled['rolling_mean']) / resampled['rolling_std']
    
    # Identify spikes
    spikes = resampled[resampled['z_score'] > threshold].copy()
    
    # Calculate magnitude (how many times above normal)
    spikes['magnitude'] = spikes['article_count'] / spikes['rolling_mean']
    
    # Sort by magnitude
    spikes = spikes.sort_values('magnitude', ascending=False).reset_index(drop=True)
    
    return spikes


def analyze_intraday_patterns(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Analyze patterns in news publication within the day.
    
    Args:
        df: DataFrame containing date information
        date_column: Column containing dates with time information
        
    Returns:
        DataFrame with hourly patterns
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
    
    # Extract hour
    df_with_hour = df.copy()
    df_with_hour['hour'] = df_with_hour[date_column].dt.hour
    
    # Count articles by hour
    hourly_counts = df_with_hour['hour'].value_counts().sort_index().reset_index()
    hourly_counts.columns = ['hour', 'article_count']
    
    # Calculate percentage
    hourly_counts['percentage'] = (hourly_counts['article_count'] / hourly_counts['article_count'].sum()) * 100
    
    # Identify peak hours (local maxima)
    peak_hours = []
    for i in range(1, 23):
        if (hourly_counts.loc[hourly_counts['hour'] == i, 'article_count'].values[0] > 
            hourly_counts.loc[hourly_counts['hour'] == i-1, 'article_count'].values[0] and
            hourly_counts.loc[hourly_counts['hour'] == i, 'article_count'].values[0] > 
            hourly_counts.loc[hourly_counts['hour'] == i+1, 'article_count'].values[0]):
            peak_hours.append(i)
    
    # Add peak hour indicator
    hourly_counts['is_peak'] = hourly_counts['hour'].isin(peak_hours)
    
    return hourly_counts


def correlate_news_volume_with_market(news_df: pd.DataFrame, market_df: pd.DataFrame,
                                    news_date_column: str = 'date',
                                    market_date_column: str = 'Date',
                                    market_metric: str = 'Close',
                                    freq: str = 'D') -> pd.DataFrame:
    """
    Correlate news volume with market movements.
    
    Args:
        news_df: DataFrame containing news data
        market_df: DataFrame containing market data (e.g., index prices)
        news_date_column: Date column in news DataFrame
        market_date_column: Date column in market DataFrame
        market_metric: Column in market DataFrame to use for correlation
        freq: Frequency for resampling
        
    Returns:
        DataFrame with correlation results
    """
    # Resample news frequency
    news_volume = resample_news_frequency(news_df, news_date_column, freq)
    
    # Ensure market date is datetime
    market_df = market_df.copy()
    market_df[market_date_column] = pd.to_datetime(market_df[market_date_column])
    
    # Merge datasets on date
    merged = pd.merge(news_volume, market_df[[market_date_column, market_metric]],
                     left_on=news_date_column, right_on=market_date_column, how='inner')
    
    # Calculate returns
    merged['market_return'] = merged[market_metric].pct_change()
    
    # Calculate correlations for different lags
    correlations = []
    for lag in range(-5, 6):  # -5 to +5 days
        if lag < 0:
            label = f'News volume leads market by {abs(lag)} days'
            corr = merged['article_count'].shift(-lag).corr(merged['market_return'])
        elif lag > 0:
            label = f'Market leads news volume by {lag} days'
            corr = merged['article_count'].corr(merged['market_return'].shift(lag))
        else:
            label = 'Same day'
            corr = merged['article_count'].corr(merged['market_return'])
        
        correlations.append({
            'lag': lag,
            'description': label,
            'correlation': corr
        })
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlations)
    
    return corr_df


def analyze_news_seasonality(df: pd.DataFrame, date_column: str = 'date') -> Dict[str, pd.DataFrame]:
    """
    Analyze seasonal patterns in news publication.
    
    Args:
        df: DataFrame containing date information
        date_column: Column containing dates
        
    Returns:
        Dictionary with different seasonal analyses
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
    
    # Extract date components
    df_with_components = df.copy()
    df_with_components['year'] = df_with_components[date_column].dt.year
    df_with_components['month'] = df_with_components[date_column].dt.month
    df_with_components['day'] = df_with_components[date_column].dt.day
    df_with_components['weekday'] = df_with_components[date_column].dt.weekday
    df_with_components['quarter'] = df_with_components[date_column].dt.quarter
    df_with_components['day_of_year'] = df_with_components[date_column].dt.dayofyear
    df_with_components['week_of_year'] = df_with_components[date_column].dt.isocalendar().week
    
    # Monthly patterns
    monthly_pattern = df_with_components.groupby('month').size().reset_index()
    monthly_pattern.columns = ['month', 'article_count']
    monthly_pattern['month_name'] = monthly_pattern['month'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
    
    # Weekly patterns
    weekly_pattern = df_with_components.groupby('weekday').size().reset_index()
    weekly_pattern.columns = ['weekday', 'article_count']
    weekly_pattern['day_name'] = weekly_pattern['weekday'].apply(
        lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
    )
    
    # Quarterly patterns
    quarterly_pattern = df_with_components.groupby('quarter').size().reset_index()
    quarterly_pattern.columns = ['quarter', 'article_count']
    
    # Return dictionary of patterns
    return {
        'monthly': monthly_pattern,
        'weekly': weekly_pattern,
        'quarterly': quarterly_pattern
    }
