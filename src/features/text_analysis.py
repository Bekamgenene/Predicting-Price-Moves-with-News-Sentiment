import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional

# No NLTK imports

# Define stopwords manually
STOPWORDS = {
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', 'should', 'now', 'do', 'does', 'did',
    'has', 'have', 'had', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'having', 'do', 'does', 'did', 'doing', 'would', 'could', 'should', 'might'
}

def custom_tokenize(text):
    """Simple tokenizer that doesn't rely on NLTK"""
    if not isinstance(text, str):
        return []
    # Simple tokenization by splitting on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words without using NLTK.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    if not isinstance(text, str):
        return []
    
    # Clean the text first
    cleaned_text = clean_text(text)
    
    # Split into words
    tokens = cleaned_text.split()
    
    # Filter stopwords and short words
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    
    return tokens


def extract_common_keywords(df: pd.DataFrame, text_column: str = 'headline',
                          n: int = 20) -> pd.DataFrame:
    """
    Extract most common keywords from text data without using NLTK.
    
    Args:
        df: DataFrame containing text data
        text_column: Column containing text data
        n: Number of top keywords to return
        
    Returns:
        DataFrame with keyword frequencies
    """
    try:
        # Process all texts
        all_tokens = []
        
        for text in df[text_column]:
            tokens = tokenize_text(text)  # Using our NLTK-free tokenizer
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Convert to DataFrame
        keyword_df = pd.DataFrame({
            'keyword': list(token_counts.keys()),
            'frequency': list(token_counts.values())
        })
        
        # Sort by frequency and get top n
        keyword_df = keyword_df.sort_values('frequency', ascending=False).head(n).reset_index(drop=True)
        
        return keyword_df
    except Exception as e:
        print(f"Error in extract_common_keywords: {e}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['keyword', 'frequency'])


def extract_keyword_bigrams(df: pd.DataFrame, text_column: str = 'headline',
                          n: int = 20) -> pd.DataFrame:
    """
    Extract most common bigrams (two-word phrases) from text data without using NLTK.
    
    Args:
        df: DataFrame containing text data
        text_column: Column containing text data
        n: Number of top bigrams to return
        
    Returns:
        DataFrame with bigram frequencies
    """
    try:
        # Process all texts to find bigrams
        bigrams_counter = Counter()
        
        for text in df[text_column]:
            # Get tokens using our custom tokenizer
            tokens = tokenize_text(text)
            
            # Create bigrams manually
            if len(tokens) > 1:
                for i in range(len(tokens) - 1):
                    bigram = (tokens[i], tokens[i + 1])
                    bigrams_counter[bigram] += 1
        
        # Get top n bigrams
        top_bigrams = bigrams_counter.most_common(n)
        
        # Convert to DataFrame
        bigram_df = pd.DataFrame({
            'bigram': [' '.join(bigram) for bigram, _ in top_bigrams],
            'frequency': [count for _, count in top_bigrams]
        })
        
        return bigram_df
    except Exception as e:
        print(f"Error in extract_keyword_bigrams: {e}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['bigram', 'frequency'])


def identify_key_phrases(df: pd.DataFrame, text_column: str = 'headline',
                       phrases: List[str] = None) -> pd.DataFrame:
    """
    Identify occurrences of specific key phrases in text without using NLTK.
    
    Args:
        df: DataFrame containing text data
        text_column: Column containing text data
        phrases: List of phrases to search for, if None uses default financial phrases
        
    Returns:
        DataFrame with phrase occurrence statistics
    """
    try:
        # Default financial phrases if none provided
        if phrases is None:
            phrases = [
                'price target', 'buy rating', 'sell rating', 'hold rating', 'overweight',
                'underweight', 'outperform', 'underperform', 'neutral rating',
                'upgraded', 'downgraded', 'initiated coverage', 'raised', 'lowered',
                'earnings beat', 'earnings miss', 'guidance', 'forecast',
                'dividend', 'stock split', 'acquisition', 'merger'
            ]
        
        # Count occurrences of each phrase
        results = []
        total_articles = len(df)
        
        for phrase in phrases:
            # Count headlines containing the phrase
            count = 0
            for text in df[text_column]:
                if isinstance(text, str) and phrase in text.lower():
                    count += 1
            
            # Calculate percentage
            percentage = (count / total_articles) * 100 if total_articles > 0 else 0
            
            results.append({
                'phrase': phrase,
                'count': count,
                'percentage': percentage
            })
        
        # Convert to DataFrame and sort by count
        phrase_df = pd.DataFrame(results)
        phrase_df = phrase_df.sort_values('count', ascending=False).reset_index(drop=True)
        
        return phrase_df
    except Exception as e:
        print(f"Error in identify_key_phrases: {e}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['phrase', 'count', 'percentage'])


def analyze_stock_coverage(df: pd.DataFrame, text_column: str = 'headline',
                        stock_column: str = 'stock') -> Dict[str, Dict]:
    """
    Analyze which keywords are most associated with specific stocks without using NLTK.
    
    Args:
        df: DataFrame containing text and stock data
        text_column: Column containing text data
        stock_column: Column containing stock symbols
        
    Returns:
        Dictionary mapping stocks to their most common keywords
    """
    try:
        # Check if stock column exists
        if stock_column not in df.columns:
            print(f"Stock column '{stock_column}' not found in DataFrame")
            return {}
        
        # Get unique stocks
        stocks = df[stock_column].unique()
        
        # Initialize results dictionary
        stock_keywords = {}
        
        # For each stock, find most common keywords
        for stock in stocks[:10]:  # Limit to top 10 stocks for efficiency
            if not isinstance(stock, (str, int, float)):
                continue  # Skip if stock is not a valid value
                
            # Filter for this stock
            stock_df = df[df[stock_column] == stock]
            
            if len(stock_df) == 0:
                continue
                
            # Extract keywords
            keywords_df = extract_common_keywords(stock_df, text_column, n=10)
            
            # Store in dictionary
            stock_keywords[str(stock)] = {
                'keywords': keywords_df,
                'article_count': len(stock_df)
            }
        
        return stock_keywords
    except Exception as e:
        print(f"Error in analyze_stock_coverage: {e}")
        return {}
