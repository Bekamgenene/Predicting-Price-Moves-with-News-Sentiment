import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter


def extract_email_domains(df: pd.DataFrame, publisher_column: str = 'publisher') -> pd.DataFrame:
    """
    Extract and analyze email domains from publisher information.
    
    Args:
        df: DataFrame containing publisher information
        publisher_column: Column containing publisher information
        
    Returns:
        DataFrame with domain statistics
    """
    # Function to extract domain from email
    def extract_domain(text):
        if not isinstance(text, str):
            return None
        
        # Check if it's an email
        email_pattern = r'[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        match = re.search(email_pattern, text)
        
        if match:
            return match.group(1).lower()
        return None
    
    # Extract domains
    domains = df[publisher_column].apply(extract_domain)
    
    # Count non-None domains
    domain_counts = domains[domains.notnull()].value_counts().reset_index()
    domain_counts.columns = ['domain', 'count']
    
    # Calculate percentage
    domain_counts['percentage'] = (domain_counts['count'] / len(df)) * 100
    
    # Add domain category (financial, news, other)
    financial_domains = ['bloomberg', 'reuters', 'cnbc', 'wsj', 'ft', 'morningstar', 'fool', 'marketwatch']
    news_domains = ['nytimes', 'washingtonpost', 'cnn', 'bbc', 'guardian', 'ap', 'npr']
    
    def categorize_domain(domain):
        for fin_domain in financial_domains:
            if fin_domain in domain:
                return 'Financial'
        for news_domain in news_domains:
            if news_domain in domain:
                return 'General News'
        return 'Other'
    
    domain_counts['category'] = domain_counts['domain'].apply(categorize_domain)
    
    return domain_counts


def analyze_publisher_sentiment(df: pd.DataFrame, publisher_column: str = 'publisher', 
                              sentiment_column: str = 'sentiment') -> pd.DataFrame:
    """
    Analyze sentiment patterns by publisher.
    
    Args:
        df: DataFrame containing publisher and sentiment information
        publisher_column: Column containing publisher information
        sentiment_column: Column containing sentiment scores
        
    Returns:
        DataFrame with publisher sentiment statistics
    """
    if sentiment_column not in df.columns:
        raise ValueError(f"Column '{sentiment_column}' not found in DataFrame")
    
    # Group by publisher and calculate sentiment statistics
    publisher_sentiment = df.groupby(publisher_column)[sentiment_column].agg(
        ['mean', 'median', 'std', 'min', 'max', 'count']
    ).reset_index()
    
    # Sort by count (most active publishers first)
    publisher_sentiment = publisher_sentiment.sort_values('count', ascending=False)
    
    # Categorize publishers by sentiment tendency
    def categorize_sentiment(row):
        mean_sentiment = row['mean']
        if mean_sentiment > 0.1:
            return 'Positive'
        elif mean_sentiment < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    publisher_sentiment['sentiment_tendency'] = publisher_sentiment.apply(categorize_sentiment, axis=1)
    
    return publisher_sentiment


def analyze_publisher_focus(df: pd.DataFrame, publisher_column: str = 'publisher', 
                          stock_column: str = 'stock', text_column: str = 'headline') -> Dict[str, pd.DataFrame]:
    """
    Analyze what stocks and topics each publisher focuses on.
    
    Args:
        df: DataFrame containing publisher, stock, and text information
        publisher_column: Column containing publisher information
        stock_column: Column containing stock symbols
        text_column: Column containing headline text
        
    Returns:
        Dictionary with publisher focus analysis
    """
    # Get top publishers
    top_publishers = df[publisher_column].value_counts().head(10).index
    
    # Initialize results
    results = {}
    
    for publisher in top_publishers:
        # Filter for this publisher
        publisher_df = df[df[publisher_column] == publisher]
        
        # Analyze stock focus
        stock_focus = publisher_df[stock_column].value_counts().head(5).reset_index()
        stock_focus.columns = ['stock', 'count']
        stock_focus['percentage'] = (stock_focus['count'] / len(publisher_df)) * 100
        
        # Analyze common words (simplified)
        all_words = ' '.join(publisher_df[text_column].dropna()).lower()
        all_words = re.sub(r'[^a-zA-Z\s]', '', all_words)
        word_counts = Counter(all_words.split())
        
        # Remove common stop words
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'in', 'to', 'for', 'with', 'on', 'at', 'from']
        for word in stop_words:
            if word in word_counts:
                del word_counts[word]
        
        # Get top words
        top_words = pd.DataFrame({
            'word': [word for word, count in word_counts.most_common(10)],
            'count': [count for word, count in word_counts.most_common(10)]
        })
        
        # Store results
        results[publisher] = {
            'stock_focus': stock_focus,
            'top_words': top_words,
            'article_count': len(publisher_df)
        }
    
    return results


def compare_publisher_activity(df: pd.DataFrame, publisher_column: str = 'publisher', 
                             date_column: str = 'date', freq: str = 'M') -> pd.DataFrame:
    """
    Compare publication activity over time across different publishers.
    
    Args:
        df: DataFrame containing publisher and date information
        publisher_column: Column containing publisher information
        date_column: Column containing date information
        freq: Frequency for time series analysis
        
    Returns:
        DataFrame with publisher activity over time
    """
    # Ensure date column is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
    
    # Get top publishers
    top_publishers = df[publisher_column].value_counts().head(5).index
    
    # Filter for top publishers
    df_top = df[df[publisher_column].isin(top_publishers)]
    
    # Group by date and publisher
    df_top['period'] = df_top[date_column].dt.to_period(freq)
    activity = df_top.groupby(['period', publisher_column]).size().unstack().fillna(0)
    
    # Convert period index to datetime for easier plotting
    activity.index = activity.index.to_timestamp()
    
    # Reset index for easier handling
    activity = activity.reset_index()
    
    return activity


def analyze_publisher_diversity(df: pd.DataFrame, publisher_column: str = 'publisher',
                              stock_column: str = 'stock') -> pd.DataFrame:
    """
    Analyze diversity of stock coverage by publisher.
    
    Args:
        df: DataFrame containing publisher and stock information
        publisher_column: Column containing publisher information
        stock_column: Column containing stock symbols
        
    Returns:
        DataFrame with publisher diversity metrics
    """
    # Get publishers with at least 10 articles
    publisher_counts = df[publisher_column].value_counts()
    active_publishers = publisher_counts[publisher_counts >= 10].index
    
    # Initialize results
    results = []
    
    for publisher in active_publishers:
        # Filter for this publisher
        publisher_df = df[df[publisher_column] == publisher]
        
        # Count articles and unique stocks
        article_count = len(publisher_df)
        unique_stocks = publisher_df[stock_column].nunique()
        
        # Calculate stock concentration (Herfindahl-Hirschman Index)
        stock_proportions = publisher_df[stock_column].value_counts(normalize=True)
        hhi = (stock_proportions ** 2).sum()
        
        # Calculate diversity score (inverse of HHI)
        diversity_score = 1 - hhi
        
        # Store results
        results.append({
            'publisher': publisher,
            'article_count': article_count,
            'unique_stocks': unique_stocks,
            'stocks_per_article': unique_stocks / article_count,
            'hhi': hhi,
            'diversity_score': diversity_score
        })
    
    # Convert to DataFrame
    diversity_df = pd.DataFrame(results)
    
    # Sort by diversity score
    diversity_df = diversity_df.sort_values('diversity_score', ascending=False).reset_index(drop=True)
    
    return diversity_df
