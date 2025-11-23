import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def calculate_text_statistics(df: pd.DataFrame, text_column: str = 'headline') -> Dict[str, Any]:
    """
    Calculate basic statistics for text lengths.
    
    Args:
        df: DataFrame containing text data
        text_column: Name of the column containing the text to analyze
        
    Returns:
        Dictionary containing text statistics
    """
    # Check if column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Create length columns if they don't exist
    length_col = f"{text_column}_length"
    word_count_col = "word_count"
    
    if length_col not in df.columns:
        df[length_col] = df[text_column].apply(len)
    
    if word_count_col not in df.columns:
        df[word_count_col] = df[text_column].apply(lambda x: len(str(x).split()))
    
    # Calculate statistics
    length_stats = df[length_col].describe().to_dict()
    word_count_stats = df[word_count_col].describe().to_dict()
    
    # Combine results
    stats = {
        'char_length': length_stats,
        'word_count': word_count_stats,
        'total_items': len(df),
        'null_count': df[text_column].isnull().sum(),
        'unique_count': df[text_column].nunique()
    }
    
    return stats


def analyze_headline_complexity(df: pd.DataFrame, text_column: str = 'headline') -> pd.DataFrame:
    """
    Analyze the complexity of headlines.
    
    Args:
        df: DataFrame containing headline data
        text_column: Name of the column containing the headlines
        
    Returns:
        DataFrame with additional complexity metrics
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate average word length
    result_df['avg_word_length'] = result_df[text_column].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    
    # Calculate headline complexity (simplified measure)
    # Using a combination of headline length and average word length
    result_df['complexity_score'] = (
        (result_df[f"{text_column}_length"] / result_df[f"{text_column}_length"].max()) * 0.5 + 
        (result_df['avg_word_length'] / result_df['avg_word_length'].max()) * 0.5
    )
    
    return result_df


def get_length_categories(df: pd.DataFrame, length_column: str = 'headline_length') -> pd.DataFrame:
    """
    Categorize headlines by length into buckets.
    
    Args:
        df: DataFrame containing headline length data
        length_column: Name of the column containing the text lengths
        
    Returns:
        DataFrame with length category column added
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Create length categories
    bins = [0, 50, 100, 150, 200, float('inf')]
    labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    
    result_df['length_category'] = pd.cut(result_df[length_column], bins=bins, labels=labels)
    
    return result_df


def find_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find outliers in a specific column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to analyze for outliers
        method: Method to use for outlier detection ('iqr' or 'zscore')
        
    Returns:
        Tuple containing (DataFrame without outliers, DataFrame of outliers)
    """
    if method.lower() == 'iqr':
        # IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        non_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
    elif method.lower() == 'zscore':
        # Z-score method
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        abs_z_scores = abs(z_scores)
        
        outliers = df[abs_z_scores > 3]
        non_outliers = df[abs_z_scores <= 3]
        
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
    
    return non_outliers, outliers
