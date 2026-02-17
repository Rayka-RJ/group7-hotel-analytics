"""
Hotel Analytics Dashboard - Streamlit Application

IS5126 Assignment 1 - HospitalityTech Solutions
A user-friendly dashboard for hotel managers to make data-driven decisions.

Core Features:
1. Overview - Key metrics and data summary
2. Hotel Performance Explorer - Rankings and comparisons
3. Competitive Benchmarking - Find comparable hotels & get recommendations
4. Rating Trends - Review activity over time
5. Best Practices - Top vs bottom performer insights
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Try imports - graceful fallback for optional modules
try:
    from src.benchmarking import HotelBenchmarking, calculate_performance_score, identify_best_practices
    BENCHMARKING_AVAILABLE = True
except ImportError:
    BENCHMARKING_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Hotel Analytics Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Data Loading ====================

def get_db_path():
    """Get database path - prefer main DB, fallback to sample."""
    main_db = PROJECT_ROOT / "data" / "reviews.db"
    sample_db = PROJECT_ROOT / "data" / "reviews_sample.db"
    if main_db.exists():
        return str(main_db), 20  # min_reviews for main (more data)
    if sample_db.exists():
        return str(sample_db), 5   # min_reviews for sample (less data)
    raise FileNotFoundError("No database found. Run 01_data_preparation.ipynb first.")


@st.cache_data(ttl=300)
def load_hotel_features(db_path: str, min_reviews: int = 20) -> pd.DataFrame:
    """Load hotel features from database."""
    conn = sqlite3.connect(db_path)

    hotel_features = pd.read_sql_query(f"""
        SELECT
            r.hotel_id,
            COUNT(*) as review_count,
            COUNT(DISTINCT r.user_id) as unique_reviewers,
            AVG(r.rating_overall) as avg_overall,
            AVG(r.rating_service) as avg_service,
            AVG(r.rating_cleanliness) as avg_cleanliness,
            AVG(r.rating_value) as avg_value,
            AVG(r.rating_location) as avg_location,
            AVG(r.rating_rooms) as avg_rooms,
            MAX(r.rating_overall) - MIN(r.rating_overall) as rating_range
        FROM reviews r
        GROUP BY r.hotel_id
        HAVING review_count >= {min_reviews}
        ORDER BY review_count DESC
    """, conn)

    # Compute std_overall
    std_df = pd.read_sql_query(
        "SELECT hotel_id, rating_overall FROM reviews WHERE rating_overall IS NOT NULL",
        conn
    )
    std_by_hotel = std_df.groupby('hotel_id')['rating_overall'].std().reset_index()
    std_by_hotel.columns = ['hotel_id', 'std_overall']
    hotel_features = hotel_features.merge(std_by_hotel, on='hotel_id', how='left')

    conn.close()

    # Derived features
    hotel_features['performance_score'] = (
        hotel_features['avg_overall'] * 0.4 +
        hotel_features['avg_service'].fillna(hotel_features['avg_overall']) * 0.2 +
        hotel_features['avg_cleanliness'].fillna(hotel_features['avg_overall']) * 0.2 +
        hotel_features['avg_value'].fillna(hotel_features['avg_overall']) * 0.2
    )
    hotel_features['consistency_score'] = 1 / (1 + hotel_features['std_overall'].fillna(1))
    hotel_features['popularity_score'] = np.log1p(hotel_features['review_count'])

    return hotel_features


@st.cache_data(ttl=300)
def load_overview_stats(db_path: str) -> dict:
    """Load overview statistics."""
    conn = sqlite3.connect(db_path)
    stats = {}
    stats['total_reviews'] = pd.read_sql_query("SELECT COUNT(*) as c FROM reviews", conn)['c'][0]
    stats['total_hotels'] = pd.read_sql_query("SELECT COUNT(DISTINCT hotel_id) as c FROM reviews", conn)['c'][0]
    stats['total_users'] = pd.read_sql_query("SELECT COUNT(DISTINCT user_id) as c FROM reviews", conn)['c'][0]
    date_range = pd.read_sql_query("SELECT MIN(review_date) as min_d, MAX(review_date) as max_d FROM reviews", conn)
    stats['date_range'] = (date_range['min_d'][0], date_range['max_d'][0])
    conn.close()
    return stats


@st.cache_data(ttl=300)
def load_monthly_trends(db_path: str) -> pd.DataFrame:
    """Load monthly review counts."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT strftime('%Y-%m', review_date) as month, COUNT(*) as review_count
        FROM reviews
        WHERE review_date IS NOT NULL
        GROUP BY month
        ORDER BY month
    """, conn)
    conn.close()
    return df


# ==================== Feature 1: Overview ====================

def render_overview(db_path: str, stats: dict):
    """Render overview dashboard with key metrics."""
    st.header("📊 Overview")
    st.markdown("**Key metrics for the hotel review analytics platform**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", f"{stats['total_reviews']:,}")
    with col2:
        st.metric("Hotels", f"{stats['total_hotels']:,}")
    with col3:
        st.metric("Unique Reviewers", f"{stats['total_users']:,}")
    with col4:
        st.metric("Date Range", f"{stats['date_range'][0][:7]} to {stats['date_range'][1][:7]}")

    st.markdown("---")
    st.info("💡 **Business Context**: This platform helps hotel managers understand customer satisfaction, "
            "identify improvement opportunities, and benchmark against comparable competitors.")


# ==================== Feature 2: Hotel Performance Explorer ====================

def render_performance_explorer(hotel_features: pd.DataFrame):
    """Render hotel performance ranking and exploration."""
    st.header("🏆 Hotel Performance Explorer")
    st.markdown("**Rank hotels by performance metrics and explore top/bottom performers**")

    sort_by = st.selectbox(
        "Sort by",
        ["performance_score", "avg_overall", "review_count", "avg_service", "avg_cleanliness", "avg_value"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    top_n = st.slider("Show top N hotels", 5, 50, 20)

    ranked = hotel_features.nlargest(top_n, sort_by)
    display_cols = ['hotel_id', 'avg_overall', 'avg_service', 'avg_cleanliness', 'avg_value', 'review_count', 'performance_score']
    display_cols = [c for c in display_cols if c in ranked.columns]

    display_df = ranked[display_cols].copy()
    for col in ['avg_overall', 'avg_service', 'avg_cleanliness', 'avg_value', 'performance_score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    st.dataframe(display_df, use_container_width=True, height=400)

    # Distribution chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(hotel_features['avg_overall'].dropna(), bins=30, edgecolor='white', alpha=0.8)
    ax.set_xlabel("Average Overall Rating")
    ax.set_ylabel("Number of Hotels")
    ax.set_title("Distribution of Hotel Ratings")
    st.pyplot(fig)
    plt.close()


# ==================== Feature 3: Competitive Benchmarking ====================

def render_competitive_benchmarking(hotel_features: pd.DataFrame):
    """Render competitive benchmarking - find similar hotels and get recommendations."""
    st.header("🎯 Competitive Benchmarking")
    st.markdown("**Identify comparable hotels and get actionable improvement recommendations**")

    if not BENCHMARKING_AVAILABLE:
        st.warning("Benchmarking module not available. Ensure src/benchmarking.py exists.")
        return

    # Hotel selector
    hotel_list = hotel_features[['hotel_id', 'avg_overall', 'review_count']].head(300)
    hotel_options = {f"Hotel {row['hotel_id']} (rating: {row['avg_overall']:.2f}, {int(row['review_count'])} reviews)": row['hotel_id']
                     for _, row in hotel_list.iterrows()}

    selected_label = st.selectbox("Select a hotel to analyze", list(hotel_options.keys()))
    target_hotel_id = hotel_options[selected_label]

    top_n = st.slider("Number of comparable hotels to find", 5, 20, 10)

    if st.button("Find Comparable Hotels & Recommendations"):
        with st.spinner("Analyzing..."):
            try:
                # Prepare benchmarking
                clustering_features = ['avg_service', 'avg_cleanliness', 'avg_value', 'avg_location', 'avg_rooms', 'review_count']
                clustering_features = [c for c in clustering_features if c in hotel_features.columns]
                if len(clustering_features) < 3:
                    clustering_features = ['avg_overall', 'avg_service', 'avg_cleanliness', 'review_count']
                    clustering_features = [c for c in clustering_features if c in hotel_features.columns]

                benchmark = HotelBenchmarking(hotel_features.copy())
                benchmark.prepare_features(clustering_features)
                benchmark.perform_clustering(n_clusters=4)

                # Find similar hotels
                similar = benchmark.find_similar_hotels(target_hotel_id, top_n=top_n)
                recs = benchmark.generate_recommendations(target_hotel_id, threshold=0.15)

                # Display similar hotels
                st.subheader("Comparable Hotels")
                display_cols = ['hotel_id', 'avg_overall', 'avg_service', 'avg_cleanliness', 'review_count', 'similarity_score']
                display_cols = [c for c in display_cols if c in similar.columns]
                sim_df = similar[display_cols].copy()
                for c in display_cols:
                    if c != 'hotel_id' and c in sim_df.columns:
                        sim_df[c] = sim_df[c].round(2)
                st.dataframe(sim_df, use_container_width=True)

                # Display recommendations
                st.subheader("Actionable Recommendations")
                if recs:
                    for r in recs:
                        severity = r['severity']
                        icon = "🔴" if severity == 'critical' else "🟡" if severity == 'warning' else "🟢"
                        st.markdown(f"{icon} **{r['metric']}**: {r['message']}")
                else:
                    st.success("No significant gaps found. Hotel is performing well relative to competitors!")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)


# ==================== Feature 4: Rating Trends ====================

def render_rating_trends(monthly_df: pd.DataFrame):
    """Render review activity trends over time."""
    st.header("📈 Rating Trends")
    st.markdown("**Review activity over time**")

    if len(monthly_df) == 0:
        st.warning("No trend data available.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_df['month'], monthly_df['review_count'], marker='o', markersize=4)
    ax.fill_between(range(len(monthly_df)), monthly_df['review_count'], alpha=0.3)
    ax.set_xlabel("Month")
    ax.set_ylabel("Review Count")
    ax.set_title("Monthly Review Volume")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


# ==================== Feature 5: Best Practices ====================

def render_best_practices(hotel_features: pd.DataFrame):
    """Render top vs bottom performer comparison."""
    st.header("💡 Best Practices")
    st.markdown("**What separates top performers from underperformers?**")

    if not BENCHMARKING_AVAILABLE:
        st.warning("Benchmarking module not available.")
        return

    top_pct = st.slider("Define top/bottom percentile", 0.05, 0.3, 0.2, 0.05)

    if st.button("Analyze Best Practices"):
        with st.spinner("Analyzing..."):
            try:
                comparison = identify_best_practices(hotel_features.copy(), top_percentile=top_pct)

                comp_display = comparison.copy()
                comp_display['Top_Performers'] = comp_display['Top_Performers'].round(2)
                comp_display['Bottom_Performers'] = comp_display['Bottom_Performers'].round(2)
                comp_display['Gap'] = comp_display['Gap'].round(2)
                comp_display['Gap_Pct'] = comp_display['Gap_Pct'].astype(str) + '%'
                st.dataframe(comp_display, use_container_width=True)

                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(comparison))
                width = 0.35
                ax.bar(x - width/2, comparison['Top_Performers'], width, label='Top Performers', color='#2ecc71')
                ax.bar(x + width/2, comparison['Bottom_Performers'], width, label='Bottom Performers', color='#e74c3c')
                ax.set_xticks(x)
                ax.set_xticklabels(comparison['Metric'], rotation=45, ha='right')
                ax.set_ylabel("Average Value")
                ax.set_title("Top vs Bottom Performers Comparison")
                ax.legend()
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.error(f"Error: {str(e)}")


# ==================== Main App ====================

def main():
    st.markdown('<p class="main-header">🏨 Hotel Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**HospitalityTech Solutions** | IS5126 Assignment 1")
    st.markdown("---")

    try:
        db_path, min_reviews = get_db_path()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Please run `01_data_preparation.ipynb` to create the database first.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    feature = st.sidebar.radio(
        "Select Feature",
        [
            "📊 Overview",
            "🏆 Hotel Performance Explorer",
            "🎯 Competitive Benchmarking",
            "📈 Rating Trends",
            "💡 Best Practices"
        ]
    )

    # Load data (cached)
    stats = load_overview_stats(db_path)
    hotel_features = load_hotel_features(db_path, min_reviews)
    monthly_trends = load_monthly_trends(db_path)

    # Route to feature
    if feature == "📊 Overview":
        render_overview(db_path, stats)
    elif feature == "🏆 Hotel Performance Explorer":
        render_performance_explorer(hotel_features)
    elif feature == "🎯 Competitive Benchmarking":
        render_competitive_benchmarking(hotel_features)
    elif feature == "📈 Rating Trends":
        render_rating_trends(monthly_trends)
    elif feature == "💡 Best Practices":
        render_best_practices(hotel_features)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Database: {Path(db_path).name}")
    st.sidebar.caption(f"Hotels loaded: {len(hotel_features):,}")


if __name__ == "__main__":
    main()
