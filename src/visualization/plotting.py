"""
Visualization module for financial news data analysis.
This module provides functions for creating various plots for data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any


def set_plotting_style(font_scale: float = 1.2) -> None:
    """
    Set the default plotting style for consistent visualizations.
    
    Args:
        font_scale: Scale factor for font sizes
    """
    # Use most basic approach that works with any matplotlib version
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'xtick.labelsize': 10 * font_scale,
        'ytick.labelsize': 10 * font_scale,
        'legend.fontsize': 10 * font_scale,
    })
    # Use seaborn's set function which is more stable across versions
    sns.set(style="whitegrid", font_scale=font_scale)


def plot_headline_length_distribution(df: pd.DataFrame, length_column: str = 'headline_length',
                                     bins: int = 50, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot the distribution of headline lengths.
    
    Args:
        df: DataFrame containing headline length data
        length_column: Name of the column containing the text lengths
        bins: Number of bins for the histogram
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Get basic statistics
    length_stats = df[length_column].describe()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot distribution
    sns.histplot(df[length_column], bins=bins, kde=True, ax=ax)
    
    # Add vertical lines for mean and median
    ax.axvline(length_stats['mean'], color='red', linestyle='--', 
              label=f'Mean: {length_stats["mean"]:.2f}')
    ax.axvline(length_stats['50%'], color='green', linestyle='--', 
              label=f'Median: {length_stats["50%"]:.2f}')
    
    # Set labels and title
    ax.set_title('Distribution of Headline Lengths')
    ax.set_xlabel('Number of Characters')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_top_publishers(publisher_counts: pd.Series, n: int = 15, 
                       figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot the top N publishers by article count.
    
    Args:
        publisher_counts: Series containing publisher counts
        n: Number of top publishers to display
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot top publishers
    publisher_counts.head(n).plot(kind='bar', ax=ax)
    
    # Set labels and title
    ax.set_title(f'Top {n} Publishers by Number of Articles')
    ax.set_xlabel('Publisher')
    ax.set_ylabel('Number of Articles')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_publisher_pie(publisher_counts: pd.Series, n: int = 10, 
                      figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
    """
    Create a pie chart of top publishers.
    
    Args:
        publisher_counts: Series containing publisher counts
        n: Number of top publishers to display separately (others grouped)
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for pie chart
    top_publishers = publisher_counts.head(n)
    others_count = publisher_counts[n:].sum()
    publishers_pie = pd.concat([top_publishers, pd.Series({'Others': others_count})])
    
    # Create pie chart
    ax.pie(publishers_pie, labels=publishers_pie.index, autopct='%1.1f%%', 
          startangle=90, shadow=True)
    ax.axis('equal')
    ax.set_title('Distribution of Articles by Publisher')
    
    plt.tight_layout()
    return fig


def plot_time_series(df: pd.DataFrame, date_column: str, value_column: str,
                    title: str, xlabel: str = 'Date', ylabel: str = 'Count',
                    figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
    """
    Plot a time series of values.
    
    Args:
        df: DataFrame containing the time series data
        date_column: Name of the column containing dates
        value_column: Name of the column containing values to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot time series
    df.plot(x=date_column, y=value_column, kind='line', marker='o', ax=ax)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_weekday_distribution(weekday_counts: pd.Series, 
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot the distribution of articles by day of week.
    
    Args:
        weekday_counts: Series containing counts by weekday
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot weekday distribution
    weekday_counts.plot(kind='bar', ax=ax)
    
    # Set labels and title
    ax.set_title('Number of Articles by Day of Week')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Articles')
    
    plt.tight_layout()
    return fig


def plot_hourly_distribution(hourly_counts: pd.Series,
                            figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot the distribution of articles by hour of day.
    
    Args:
        hourly_counts: Series containing counts by hour
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot hourly distribution
    hourly_counts.plot(kind='bar', ax=ax)
    
    # Set labels and title
    ax.set_title('Number of Articles by Hour of Day')
    ax.set_xlabel('Hour (24-hour format)')
    ax.set_ylabel('Number of Articles')
    
    plt.tight_layout()
    return fig


def plot_weekday_hour_heatmap(heatmap_data: pd.DataFrame,
                             figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create a heatmap of article frequency by weekday and hour.
    
    Args:
        heatmap_data: DataFrame with weekdays as index and hours as columns
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='g', ax=ax)
    
    # Set labels and title
    ax.set_title('Article Publication Frequency by Weekday and Hour')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    
    plt.tight_layout()
    return fig


def plot_publisher_stock_heatmap(df: pd.DataFrame, top_n_publishers: int = 10,
                               top_n_stocks: int = 10, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Create a heatmap showing the relationship between publishers and stocks.
    
    Args:
        df: DataFrame containing publisher and stock data
        top_n_publishers: Number of top publishers to include
        top_n_stocks: Number of top stocks to include
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Get top publishers and stocks
    top_publishers = df['publisher'].value_counts().head(top_n_publishers).index
    top_stocks = df['stock'].value_counts().head(top_n_stocks).index
    
    # Create cross-tabulation
    publisher_stock_counts = pd.crosstab(df['publisher'], df['stock'])
    
    # Filter for top publishers and stocks
    heatmap_data = publisher_stock_counts.loc[top_publishers, top_stocks]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='d', ax=ax)
    
    # Set labels and title
    ax.set_title('Number of Articles by Publisher and Stock')
    
    plt.tight_layout()
    return fig


def create_multiple_plots(plots_config: List[Dict[str, Any]], 
                         figsize: Tuple[int, int] = (18, 12), 
                         nrows: int = 2, ncols: int = 2) -> plt.Figure:
    """
    Create a figure with multiple subplots based on configuration.
    
    Args:
        plots_config: List of dictionaries with plot configurations
        figsize: Overall figure size as (width, height)
        nrows: Number of rows in the subplot grid
        ncols: Number of columns in the subplot grid
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Create each subplot
    for i, plot_config in enumerate(plots_config):
        if i < len(axes):
            plot_type = plot_config.get('type', 'bar')
            data = plot_config.get('data')
            title = plot_config.get('title', f'Plot {i+1}')
            
            if plot_type == 'bar':
                data.plot(kind='bar', ax=axes[i])
            elif plot_type == 'line':
                data.plot(kind='line', marker='o', ax=axes[i])
            elif plot_type == 'pie':
                data.plot(kind='pie', ax=axes[i])
            elif plot_type == 'hist':
                axes[i].hist(data, bins=plot_config.get('bins', 30))
            
            axes[i].set_title(title)
            axes[i].set_xlabel(plot_config.get('xlabel', ''))
            axes[i].set_ylabel(plot_config.get('ylabel', ''))
    
    # Hide any unused subplots
    for i in range(len(plots_config), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
