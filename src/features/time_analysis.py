import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def get_time_distribution(df: pd.DataFrame, date_column: str = 'date') -> Dict[str, pd.Series]:
    """
    Analyze the distribution of articles over different time periods.
    
    Args:
        df: DataFrame containing date information
        date_column: Name of the column containing the dates
        
    Returns:
        Dictionary with different time period distributions
    """
    # Check if column exists and is datetime
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract time components if not already present
    if 'year' not in df.columns:
        df['year'] = df[date_column].dt.year
    if 'month' not in df.columns:
        df['month'] = df[date_column].dt.month
    if 'day' not in df.columns:
        df['day'] = df[date_column].dt.day
    if 'weekday' not in df.columns:
        df['weekday'] = df[date_column].dt.day_name()
    if 'hour' not in df.columns:
        df['hour'] = df[date_column].dt.hour
    
    # Calculate distributions
    distributions = {
        'yearly': df['year'].value_counts().sort_index(),
        'monthly': df.groupby(['year', 'month']).size().reset_index(name='count'),
        'weekday': df['weekday'].value_counts(),
        'hourly': df['hour'].value_counts().sort_index()
    }
    
    # Sort weekday properly
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    distributions['weekday'] = distributions['weekday'].reindex(weekday_order)
    
    return distributions


def find_busy_periods(df: pd.DataFrame, date_column: str = 'date', 
                     freq: str = 'D', threshold: float = 0.75) -> pd.DataFrame:
    """
    Find time periods with unusually high news volume.
    
    Args:
        df: DataFrame containing date information
        date_column: Name of the column containing the dates
        freq: Frequency for resampling ('D' for daily, 'W' for weekly, etc.)
        threshold: Percentile threshold (0-1) for considering a period "busy"
        
    Returns:
        DataFrame with busy periods and their article counts
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by time period and count articles
    period_counts = df.groupby(pd.Grouper(key=date_column, freq=freq)).size()
    
    # Determine busy threshold (e.g., 75th percentile)
    busy_threshold = period_counts.quantile(threshold)
    
    # Filter for busy periods
    busy_periods = period_counts[period_counts > busy_threshold].sort_values(ascending=False)
    
    # Convert to DataFrame
    busy_df = busy_periods.reset_index()
    busy_df.columns = [date_column, 'article_count']
    
    return busy_df


def analyze_time_patterns(df: pd.DataFrame, date_column: str = 'date',
                         additional_columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Analyze patterns in news publication over time with optional additional grouping.
    
    Args:
        df: DataFrame containing date information
        date_column: Name of the column containing the dates
        additional_columns: Optional list of columns to group by along with time
        
    Returns:
        Dictionary with different time pattern analyses
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Initialize results
    results = {}
    
    # Daily trend
    daily_trend = df.groupby(pd.Grouper(key=date_column, freq='D')).size()
    results['daily_trend'] = daily_trend.reset_index()
    results['daily_trend'].columns = [date_column, 'article_count']
    
    # Weekly trend
    weekly_trend = df.groupby(pd.Grouper(key=date_column, freq='W')).size()
    results['weekly_trend'] = weekly_trend.reset_index()
    results['weekly_trend'].columns = [date_column, 'article_count']
    
    # Monthly trend
    monthly_trend = df.groupby(pd.Grouper(key=date_column, freq='M')).size()
    results['monthly_trend'] = monthly_trend.reset_index()
    results['monthly_trend'].columns = [date_column, 'article_count']
    
    # If additional columns provided, do more complex analysis
    if additional_columns:
        for column in additional_columns:
            if column in df.columns:
                # Group by date and the additional column
                grouped = df.groupby([pd.Grouper(key=date_column, freq='W'), column]).size().reset_index()
                grouped.columns = [date_column, column, 'article_count']
                
                # Pivot to get columns for each unique value in the additional column
                pivoted = grouped.pivot(index=date_column, columns=column, values='article_count').fillna(0)
                
                results[f'weekly_{column}_trend'] = pivoted
    
    return results


def detect_anomalies(df: pd.DataFrame, date_column: str = 'date', 
                    freq: str = 'D', window: int = 7, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect anomalies in news publication frequency.
    
    Args:
        df: DataFrame containing date information
        date_column: Name of the column containing the dates
        freq: Frequency for resampling ('D' for daily, 'W' for weekly, etc.)
        window: Window size for rolling statistics
        threshold: Number of standard deviations to consider as anomaly
        
    Returns:
        DataFrame with anomalies and their z-scores
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by time period and count articles
    period_counts = df.groupby(pd.Grouper(key=date_column, freq=freq)).size()
    
    # Calculate rolling statistics
    rolling_mean = period_counts.rolling(window=window, center=True).mean()
    rolling_std = period_counts.rolling(window=window, center=True).std()
    
    # Calculate z-scores
    z_scores = (period_counts - rolling_mean) / rolling_std
    
    # Identify anomalies
    anomalies = z_scores[abs(z_scores) > threshold].dropna()
    
    # Create results DataFrame
    if len(anomalies) > 0:
        anomalies_df = pd.DataFrame({
            date_column: anomalies.index,
            'article_count': period_counts[anomalies.index],
            'expected_count': rolling_mean[anomalies.index],
            'z_score': anomalies
        })
        return anomalies_df.sort_values('z_score', ascending=False)
    else:
        return pd.DataFrame(columns=[date_column, 'article_count', 'expected_count', 'z_score'])


def get_publication_weekday_hourly_heatmap(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Create a heatmap data of publication frequency by weekday and hour.
    
    Args:
        df: DataFrame containing date information
        date_column: Name of the column containing the dates
        
    Returns:
        DataFrame with weekday-hour combinations and their frequencies
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract weekday and hour
    df['weekday'] = df[date_column].dt.day_name()
    df['hour'] = df[date_column].dt.hour
    
    # Count articles by weekday and hour
    heatmap_data = df.groupby(['weekday', 'hour']).size().reset_index(name='article_count')
    
    # Pivot for heatmap format
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_data.pivot(index='weekday', columns='hour', values='article_count').fillna(0)
    heatmap_pivot = heatmap_pivot.reindex(weekday_order)
    
    return heatmap_pivot
