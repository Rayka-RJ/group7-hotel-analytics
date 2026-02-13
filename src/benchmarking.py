"""
Hotel Benchmarking Module

Provides utilities for competitive hotel analysis including:
- Hotel similarity calculation
- Clustering and segmentation
- Performance comparison
- Recommendation generation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from typing import List, Dict, Tuple


class HotelBenchmarking:
    """
    Main class for hotel competitive benchmarking analysis.
    """

    def __init__(self, hotel_features: pd.DataFrame):
        """
        Initialize with hotel features DataFrame.

        Args:
            hotel_features: DataFrame with hotel_id and feature columns
        """
        self.hotel_features = hotel_features.copy()
        self.scaler = StandardScaler()
        self.kmeans = None
        self.X_scaled = None
        self.clustering_features = None

    def prepare_features(self, feature_columns: List[str]) -> np.ndarray:
        """
        Prepare and scale features for clustering.

        Args:
            feature_columns: List of column names to use for clustering

        Returns:
            Scaled feature matrix
        """
        self.clustering_features = feature_columns
        X = self.hotel_features[feature_columns].fillna(
            self.hotel_features[feature_columns].mean()
        )
        self.X_scaled = self.scaler.fit_transform(X)
        return self.X_scaled

    def perform_clustering(self, n_clusters: int = 5, random_state: int = 42) -> np.ndarray:
        """
        Perform K-Means clustering on hotels.

        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility

        Returns:
            Cluster labels
        """
        if self.X_scaled is None:
            raise ValueError("Must call prepare_features() first")

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        self.hotel_features['cluster'] = self.kmeans.fit_predict(self.X_scaled)
        return self.hotel_features['cluster'].values

    def find_similar_hotels(
        self,
        target_hotel_id: int,
        top_n: int = 10,
        same_cluster_only: bool = True
    ) -> pd.DataFrame:
        """
        Find most similar hotels to a target hotel.

        Args:
            target_hotel_id: ID of target hotel
            top_n: Number of similar hotels to return
            same_cluster_only: If True, only consider hotels in same cluster

        Returns:
            DataFrame of similar hotels with similarity scores
        """
        if self.X_scaled is None:
            raise ValueError("Must call prepare_features() first")

        # Get target hotel info
        target_idx = self.hotel_features[
            self.hotel_features['hotel_id'] == target_hotel_id
        ].index[0]
        target_features = self.X_scaled[target_idx]

        # Filter candidates
        if same_cluster_only and 'cluster' in self.hotel_features.columns:
            target_cluster = self.hotel_features.loc[target_idx, 'cluster']
            candidates = self.hotel_features[
                self.hotel_features['cluster'] == target_cluster
            ]
        else:
            candidates = self.hotel_features

        # Calculate distances
        distances = []
        for idx in candidates.index:
            if idx != target_idx:
                dist = euclidean(target_features, self.X_scaled[idx])
                distances.append((idx, dist))

        # Sort and get top N
        distances.sort(key=lambda x: x[1])
        top_indices = [idx for idx, _ in distances[:top_n]]

        similar_hotels = self.hotel_features.loc[top_indices].copy()
        similar_hotels['distance'] = [dist for _, dist in distances[:top_n]]
        similar_hotels['similarity_score'] = 1 / (1 + similar_hotels['distance'])

        return similar_hotels.sort_values('similarity_score', ascending=False)

    def compare_with_competitors(
        self,
        target_hotel_id: int,
        competitor_count: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare target hotel with its competitors.

        Args:
            target_hotel_id: ID of target hotel
            competitor_count: Number of competitors to include

        Returns:
            Tuple of (comparison_df, competitors_df)
        """
        target = self.hotel_features[
            self.hotel_features['hotel_id'] == target_hotel_id
        ].iloc[0]

        competitors = self.find_similar_hotels(
            target_hotel_id,
            top_n=competitor_count
        )

        # Build comparison
        metrics = [
            'avg_overall', 'avg_service', 'avg_cleanliness',
            'avg_value', 'avg_location', 'avg_rooms'
        ]

        comparison_data = []
        for metric in metrics:
            if metric in target.index and metric in competitors.columns:
                comparison_data.append({
                    'Metric': metric.replace('avg_', '').title(),
                    'Target': target[metric],
                    'Competitor_Avg': competitors[metric].mean(),
                    'Difference': target[metric] - competitors[metric].mean()
                })

        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df, competitors

    def generate_recommendations(
        self,
        target_hotel_id: int,
        threshold: float = 0.2
    ) -> List[Dict[str, str]]:
        """
        Generate actionable recommendations based on competitive gaps.

        Args:
            target_hotel_id: ID of target hotel
            threshold: Minimum gap to trigger recommendation

        Returns:
            List of recommendation dictionaries
        """
        comparison_df, competitors = self.compare_with_competitors(target_hotel_id)

        recommendations = []

        for _, row in comparison_df.iterrows():
            metric = row['Metric']
            diff = row['Difference']

            if abs(diff) >= threshold:
                if diff < 0:
                    severity = 'critical' if diff < -0.3 else 'warning'
                    recommendations.append({
                        'severity': severity,
                        'metric': metric,
                        'gap': abs(diff),
                        'message': f"{metric} is {abs(diff):.2f} points below competitors"
                    })
                else:
                    recommendations.append({
                        'severity': 'strength',
                        'metric': metric,
                        'gap': diff,
                        'message': f"{metric} is {diff:.2f} points ahead (competitive advantage)"
                    })

        return recommendations

    def get_cluster_profile(self, cluster_id: int) -> pd.Series:
        """
        Get aggregated profile of a cluster.

        Args:
            cluster_id: ID of cluster to profile

        Returns:
            Series with cluster statistics
        """
        if 'cluster' not in self.hotel_features.columns:
            raise ValueError("Must perform clustering first")

        cluster_data = self.hotel_features[
            self.hotel_features['cluster'] == cluster_id
        ]

        profile = {
            'hotel_count': len(cluster_data),
            'avg_rating': cluster_data['avg_overall'].mean(),
            'avg_review_count': cluster_data['review_count'].mean(),
            'avg_service': cluster_data['avg_service'].mean(),
            'avg_cleanliness': cluster_data['avg_cleanliness'].mean(),
            'avg_value': cluster_data['avg_value'].mean(),
        }

        return pd.Series(profile)

    def rank_hotels_in_cluster(
        self,
        cluster_id: int,
        by: str = 'avg_overall',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Rank hotels within a cluster by specified metric.

        Args:
            cluster_id: ID of cluster
            by: Column to rank by
            top_n: Number of top hotels to return

        Returns:
            DataFrame of top hotels in cluster
        """
        if 'cluster' not in self.hotel_features.columns:
            raise ValueError("Must perform clustering first")

        cluster_data = self.hotel_features[
            self.hotel_features['cluster'] == cluster_id
        ]

        return cluster_data.nlargest(top_n, by)


def calculate_performance_score(
    df: pd.DataFrame,
    weights: Dict[str, float] = None
) -> pd.Series:
    """
    Calculate weighted performance score for hotels.

    Args:
        df: DataFrame with rating columns
        weights: Dict of {column: weight}. If None, uses default weights.

    Returns:
        Series of performance scores
    """
    if weights is None:
        weights = {
            'avg_overall': 0.4,
            'avg_service': 0.2,
            'avg_cleanliness': 0.2,
            'avg_value': 0.2
        }

    score = pd.Series(0, index=df.index)
    for col, weight in weights.items():
        if col in df.columns:
            score += df[col] * weight

    return score


def identify_best_practices(
    hotel_features: pd.DataFrame,
    top_percentile: float = 0.2
) -> pd.DataFrame:
    """
    Identify characteristics of top-performing hotels.

    Args:
        hotel_features: DataFrame with hotel features
        top_percentile: Percentile to define "top performers"

    Returns:
        DataFrame comparing top vs bottom performers
    """
    # Calculate performance scores if not present
    if 'performance_score' not in hotel_features.columns:
        hotel_features['performance_score'] = calculate_performance_score(hotel_features)

    n_top = int(len(hotel_features) * top_percentile)
    top = hotel_features.nlargest(n_top, 'performance_score')
    bottom = hotel_features.nsmallest(n_top, 'performance_score')

    metrics = ['avg_overall', 'avg_service', 'avg_cleanliness', 'avg_value',
               'avg_location', 'avg_rooms', 'review_count']

    comparison = pd.DataFrame({
        'Metric': [m.replace('avg_', '').title() for m in metrics],
        'Top_Performers': [top[m].mean() for m in metrics],
        'Bottom_Performers': [bottom[m].mean() for m in metrics]
    })

    comparison['Gap'] = comparison['Top_Performers'] - comparison['Bottom_Performers']
    comparison['Gap_Pct'] = (comparison['Gap'] / comparison['Bottom_Performers'] * 100).round(1)

    return comparison


# Usage example
if __name__ == "__main__":
    import sqlite3

    # Connect to database
    conn = sqlite3.connect('../data/reviews_sample.db')

    # Load hotel features
    hotel_features = pd.read_sql_query("""
        SELECT
            hotel_id,
            COUNT(*) as review_count,
            AVG(rating_overall) as avg_overall,
            AVG(rating_service) as avg_service,
            AVG(rating_cleanliness) as avg_cleanliness,
            AVG(rating_value) as avg_value,
            AVG(rating_location) as avg_location,
            AVG(rating_rooms) as avg_rooms
        FROM reviews
        GROUP BY hotel_id
        HAVING review_count >= 5
    """, conn)

    conn.close()

    # Initialize benchmarking
    benchmark = HotelBenchmarking(hotel_features)

    # Prepare features
    features = ['avg_overall', 'avg_service', 'avg_cleanliness', 'avg_value',
                'avg_location', 'avg_rooms', 'review_count']
    benchmark.prepare_features(features)

    # Perform clustering
    benchmark.perform_clustering(n_clusters=3)

    # Find similar hotels
    target_hotel = hotel_features.iloc[0]['hotel_id']
    similar = benchmark.find_similar_hotels(target_hotel, top_n=5)

    print(f"Top 5 similar hotels to {target_hotel}:")
    print(similar[['hotel_id', 'avg_overall', 'similarity_score']])

    # Generate recommendations
    recs = benchmark.generate_recommendations(target_hotel)
    print(f"\nRecommendations for hotel {target_hotel}:")
    for rec in recs:
        print(f"  {rec['severity'].upper()}: {rec['message']}")
