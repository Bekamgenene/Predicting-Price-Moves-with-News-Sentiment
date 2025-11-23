import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def get_publisher_counts(df: pd.DataFrame, publisher_column: str = 'publisher') -> pd.Series:
    """
    Count the number of articles per publisher.
    
    Args:
        df: DataFrame containing publisher data
        publisher_column: Name of the column containing publisher information
        
    Returns:
        Series containing publisher counts in descending order
    """
    # Check if column exists
    if publisher_column not in df.columns:
        raise ValueError(f"Column '{publisher_column}' not found in DataFrame")
    
    # Count articles per publisher
    publisher_counts = df[publisher_column].value_counts()
    
    return publisher_counts


def get_top_publishers(df: pd.DataFrame, n: int = 10, 
                       publisher_column: str = 'publisher') -> pd.Series:
    """
    Get the top N publishers by article count.
    
    Args:
        df: DataFrame containing publisher data
        n: Number of top publishers to return
        publisher_column: Name of the column containing publisher information
        
    Returns:
        Series containing top N publishers with counts
    """
    publisher_counts = get_publisher_counts(df, publisher_column)
    return publisher_counts.head(n)


def calculate_publisher_statistics(df: pd.DataFrame, 
                                  publisher_column: str = 'publisher',
                                  metric_column: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate statistics for each publisher, optionally grouped by a metric column.
    
    Args:
        df: DataFrame containing publisher data
        publisher_column: Name of the column containing publisher information
        metric_column: Optional column to calculate metrics on (e.g., headline_length)
        
    Returns:
        DataFrame with publisher statistics
    """
    # Group by publisher
    publisher_groups = df.groupby(publisher_column)
    
    # Calculate basic stats
    results = {
        'article_count': publisher_groups.size()
    }
    
    # If metric column is provided, calculate additional statistics
    if metric_column and metric_column in df.columns:
        metric_stats = publisher_groups[metric_column].agg(['mean', 'median', 'min', 'max', 'std'])
        
        # Merge the results
        for stat in ['mean', 'median', 'min', 'max', 'std']:
            results[f'{metric_column}_{stat}'] = metric_stats[stat]
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).sort_values('article_count', ascending=False)
    
    return results_df


def analyze_publisher_stock_focus(df: pd.DataFrame,
                                 publisher_column: str = 'publisher',
                                 stock_column: str = 'stock') -> Dict[str, pd.Series]:
    """
    Analyze which stocks each publisher focuses on the most.
    
    Args:
        df: DataFrame containing publisher and stock data
        publisher_column: Name of the column containing publisher information
        stock_column: Name of the column containing stock symbols
        
    Returns:
        Dictionary mapping each publisher to their most covered stocks
    """
    # Check if columns exist
    if publisher_column not in df.columns or stock_column not in df.columns:
        missing = []
        if publisher_column not in df.columns:
            missing.append(publisher_column)
        if stock_column not in df.columns:
            missing.append(stock_column)
        raise ValueError(f"Columns {missing} not found in DataFrame")
    
    # Get top publishers
    top_publishers = get_publisher_counts(df, publisher_column).head(10).index
    
    # Initialize result dictionary
    publisher_focus = {}
    
    # For each top publisher, get their stock focus
    for publisher in top_publishers:
        publisher_data = df[df[publisher_column] == publisher]
        stock_counts = publisher_data[stock_column].value_counts()
        publisher_focus[publisher] = stock_counts
    
    return publisher_focus


def calculate_publisher_diversity(df: pd.DataFrame,
                                 publisher_column: str = 'publisher',
                                 stock_column: str = 'stock') -> pd.Series:
    """
    Calculate a diversity score for each publisher based on how many different stocks they cover.
    
    Args:
        df: DataFrame containing publisher and stock data
        publisher_column: Name of the column containing publisher information
        stock_column: Name of the column containing stock symbols
        
    Returns:
        Series containing diversity scores for each publisher
    """
    # Group by publisher
    publisher_groups = df.groupby(publisher_column)
    
    # Calculate diversity metrics
    diversity = {}
    
    for publisher, group in publisher_groups:
        # Count total articles
        total_articles = len(group)
        
        # Count unique stocks
        unique_stocks = group[stock_column].nunique()
        
        # Calculate entropy (higher means more diverse)
        stock_probabilities = group[stock_column].value_counts(normalize=True)
        entropy = -sum(p * np.log2(p) for p in stock_probabilities)
        
        # Calculate diversity score (combining unique count and entropy)
        diversity[publisher] = {
            'total_articles': total_articles,
            'unique_stocks': unique_stocks,
            'entropy': entropy,
            'diversity_score': (unique_stocks / len(df[stock_column].unique())) * entropy
        }
    
    # Convert to DataFrame and sort
    diversity_df = pd.DataFrame(diversity).T
    diversity_df = diversity_df.sort_values('diversity_score', ascending=False)
    
    return diversity_df
