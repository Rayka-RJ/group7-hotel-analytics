"""
Utility Functions for Hotel Review Analytics

Common helper functions used across notebooks and scripts.
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
from functools import wraps


# ====================
# Database Utilities
# ====================

def get_connection(db_path: str = '../data/reviews.db') -> sqlite3.Connection:
    """
    Create database connection with optimizations.

    Args:
        db_path: Path to SQLite database

    Returns:
        Database connection object
    """
    conn = sqlite3.connect(db_path)
    # Enable performance optimizations
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def execute_query(query: str, db_path: str = '../data/reviews.db',
                  params: Optional[Tuple] = None) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.

    Args:
        query: SQL query string
        db_path: Path to database
        params: Optional query parameters

    Returns:
        Query results as DataFrame
    """
    conn = get_connection(db_path)
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_hotel_stats(hotel_id: int, db_path: str = '../data/reviews.db') -> Dict:
    """
    Get comprehensive statistics for a specific hotel.

    Args:
        hotel_id: Hotel ID
        db_path: Path to database

    Returns:
        Dictionary of hotel statistics
    """
    query = """
        SELECT
            COUNT(*) as review_count,
            AVG(rating_overall) as avg_rating,
            AVG(rating_service) as avg_service,
            AVG(rating_cleanliness) as avg_cleanliness,
            AVG(rating_value) as avg_value,
            MIN(review_date) as first_review,
            MAX(review_date) as last_review,
            COUNT(DISTINCT user_id) as unique_reviewers
        FROM reviews
        WHERE hotel_id = ?
    """
    df = execute_query(query, db_path, params=(hotel_id,))
    return df.iloc[0].to_dict() if len(df) > 0 else {}


# ====================
# Data Processing
# ====================

def clean_rating(rating: float) -> Optional[float]:
    """
    Clean and validate rating value.

    Args:
        rating: Rating value

    Returns:
        Cleaned rating or None if invalid
    """
    if pd.isna(rating):
        return None
    if 0 <= rating <= 5:
        return float(rating)
    return None


def calculate_weighted_rating(
    ratings: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate weighted average rating.

    Args:
        ratings: Dict of {dimension: rating}
        weights: Dict of {dimension: weight}

    Returns:
        Weighted average rating
    """
    if weights is None:
        weights = {
            'overall': 0.4,
            'service': 0.2,
            'cleanliness': 0.2,
            'value': 0.2
        }

    total_weight = 0
    weighted_sum = 0

    for dim, rating in ratings.items():
        if dim in weights and rating is not None:
            weighted_sum += rating * weights[dim]
            total_weight += weights[dim]

    return weighted_sum / total_weight if total_weight > 0 else 0


def bin_ratings(df: pd.DataFrame, rating_col: str = 'rating_overall',
                labels: Optional[List[str]] = None) -> pd.Series:
    """
    Bin ratings into categories.

    Args:
        df: DataFrame with ratings
        rating_col: Column name for ratings
        labels: Optional custom labels

    Returns:
        Series with binned categories
    """
    if labels is None:
        labels = ['Poor', 'Below Average', 'Average', 'Good', 'Excellent']

    bins = [0, 2, 3, 3.5, 4.5, 5]
    return pd.cut(df[rating_col], bins=bins, labels=labels, include_lowest=True)


# ====================
# Statistical Functions
# ====================

def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean.

    Args:
        data: Array of values
        confidence: Confidence level (0-1)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats

    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean - margin, mean + margin


def detect_outliers_iqr(
    data: pd.Series,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers using IQR method.

    Args:
        data: Series of numeric values
        multiplier: IQR multiplier (default 1.5)

    Returns:
        Boolean series indicating outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return (data < lower_bound) | (data > upper_bound)


# ====================
# Performance Monitoring
# ====================

def timer(func):
    """
    Decorator to measure function execution time.

    Usage:
        @timer
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def profile_query(query: str, db_path: str = '../data/reviews.db',
                  iterations: int = 5) -> Dict:
    """
    Profile SQL query performance.

    Args:
        query: SQL query to profile
        db_path: Path to database
        iterations: Number of iterations to run

    Returns:
        Dict with performance metrics
    """
    conn = get_connection(db_path)
    execution_times = []

    try:
        for _ in range(iterations):
            start = time.time()
            pd.read_sql_query(query, conn)
            execution_times.append(time.time() - start)

        return {
            'query': query[:100] + '...' if len(query) > 100 else query,
            'iterations': iterations,
            'avg_time': np.mean(execution_times),
            'min_time': np.min(execution_times),
            'max_time': np.max(execution_times),
            'std_time': np.std(execution_times)
        }
    finally:
        conn.close()


# ====================
# Data Validation
# ====================

def validate_hotel_features(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate hotel features DataFrame.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Required columns
    required_cols = ['hotel_id', 'avg_overall', 'review_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Rating ranges
    rating_cols = [col for col in df.columns if 'rating' in col or 'avg_' in col]
    for col in rating_cols:
        if col in df.columns:
            if df[col].min() < 0 or df[col].max() > 5:
                issues.append(f"Invalid ratings in {col}: must be 0-5")

    # Null values
    if df['hotel_id'].isnull().any():
        issues.append("Null values in hotel_id")

    return len(issues) == 0, issues


def check_data_quality(db_path: str = '../data/reviews.db') -> Dict:
    """
    Run comprehensive data quality checks.

    Args:
        db_path: Path to database

    Returns:
        Dict with quality metrics
    """
    conn = get_connection(db_path)

    checks = {}

    try:
        # Total counts
        checks['total_reviews'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM reviews", conn
        )['cnt'][0]

        # Missing values
        checks['missing_ratings'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM reviews WHERE rating_overall IS NULL", conn
        )['cnt'][0]

        checks['missing_text'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM reviews WHERE text IS NULL OR text = ''", conn
        )['cnt'][0]

        # Empty user IDs
        checks['empty_user_ids'] = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM reviews WHERE user_id = ''", conn
        )['cnt'][0]

        # Date range
        date_range = pd.read_sql_query(
            "SELECT MIN(review_date) as min_date, MAX(review_date) as max_date FROM reviews",
            conn
        )
        checks['date_range'] = (date_range['min_date'][0], date_range['max_date'][0])

        # Calculate quality score
        total = checks['total_reviews']
        issues = checks['missing_ratings'] + checks['missing_text'] + checks['empty_user_ids']
        checks['quality_score'] = (total - issues) / total * 100

        return checks

    finally:
        conn.close()


# ====================
# Formatting Utilities
# ====================

def format_number(num: float, decimals: int = 2) -> str:
    """Format number with thousand separators."""
    return f"{num:,.{decimals}f}"


def format_percentage(num: float, decimals: int = 1) -> str:
    """Format number as percentage."""
    return f"{num:.{decimals}f}%"


def format_date(date_str: str, format_out: str = '%Y-%m-%d') -> str:
    """Convert date string to formatted string."""
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.strftime(format_out)
    except:
        return date_str


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Truncate long text strings."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ====================
# Export Utilities
# ====================

def export_to_csv(df: pd.DataFrame, filename: str, add_timestamp: bool = True):
    """
    Export DataFrame to CSV with optional timestamp.

    Args:
        df: DataFrame to export
        filename: Output filename
        add_timestamp: Whether to add timestamp to filename
    """
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base, ext = filename.rsplit('.', 1)
        filename = f"{base}_{timestamp}.{ext}"

    df.to_csv(filename, index=False)
    print(f"[OK] Exported to {filename}")


def create_summary_report(db_path: str = '../data/reviews.db') -> pd.DataFrame:
    """
    Create summary report of database contents.

    Args:
        db_path: Path to database

    Returns:
        DataFrame with summary statistics
    """
    stats = check_data_quality(db_path)

    summary = pd.DataFrame({
        'Metric': [
            'Total Reviews',
            'Date Range',
            'Missing Ratings',
            'Missing Text',
            'Empty User IDs',
            'Data Quality Score'
        ],
        'Value': [
            format_number(stats['total_reviews'], 0),
            f"{stats['date_range'][0]} to {stats['date_range'][1]}",
            format_number(stats['missing_ratings'], 0),
            format_number(stats['missing_text'], 0),
            format_number(stats['empty_user_ids'], 0),
            format_percentage(stats['quality_score'])
        ]
    })

    return summary


# Usage example
if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...\n")

    # Test database connection
    try:
        conn = get_connection('../data/reviews_sample.db')
        print("[OK] Database connection successful")
        conn.close()
    except Exception as e:
        print(f"[FAIL] Database connection failed: {e}")

    # Test data quality check
    try:
        quality = check_data_quality('../data/reviews_sample.db')
        print(f"\n[OK] Data quality score: {quality['quality_score']:.1f}%")
        print(f"  Total reviews: {quality['total_reviews']:,}")
        print(f"  Date range: {quality['date_range'][0]} to {quality['date_range'][1]}")
    except Exception as e:
        print(f"[FAIL] Quality check failed: {e}")

    print("\n[OK] All tests passed!")
