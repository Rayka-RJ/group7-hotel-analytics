import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class HotelReviewDataProcessor:
    def __init__(self, db_path='data/reviews.db'):
        self.db_path = db_path
        self.conn = None

    def create_database(self):
        """Create schema and table structures."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hotels (
                hotel_id INTEGER PRIMARY KEY,
                hotel_name TEXT,
                first_review_date DATE,
                last_review_date DATE,
                total_reviews INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                location TEXT,
                num_reviews INTEGER,
                num_cities INTEGER,
                num_helpful_votes INTEGER,
                num_type_reviews INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                review_id INTEGER PRIMARY KEY,
                hotel_id INTEGER,
                user_id TEXT,
                review_date DATE,
                date_stayed TEXT,
                title TEXT,
                text TEXT,
                via_mobile BOOLEAN,
                num_helpful_votes INTEGER,
                rating_overall REAL,
                rating_service REAL,
                rating_cleanliness REAL,
                rating_value REAL,
                rating_location REAL,
                rating_sleep_quality REAL,
                rating_rooms REAL,
                FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        print("Database schema created")

    def create_indexes(self):
        """Create indexes to optimize query performance."""
        cursor = self.conn.cursor()

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_hotel_id ON reviews(hotel_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_id ON reviews(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_review_date ON reviews(review_date)",
            "CREATE INDEX IF NOT EXISTS idx_rating_overall ON reviews(rating_overall)",
            "CREATE INDEX IF NOT EXISTS idx_date_stayed ON reviews(date_stayed)",
            "CREATE INDEX IF NOT EXISTS idx_hotel_date ON reviews(hotel_id, review_date)",
            "CREATE INDEX IF NOT EXISTS idx_hotel_rating ON reviews(hotel_id, rating_overall)",
        ]

        for idx_sql in indexes:
            cursor.execute(idx_sql)

        self.conn.commit()
        print("Indexes created")

    def parse_date(self, date_str):
        """Parse date from 'Month Day, Year' to 'YYYY-MM-DD' format."""
        try:
            return datetime.strptime(date_str, "%B %d, %Y").strftime("%Y-%m-%d")
        except:
            return None

    def load_json_data(self, json_path, limit=None, year_start=2008, year_end=2012, sample_rate=None):
        """
        Load and filter JSON data into the database.

        Args:
            json_path: JSON file path
            limit: max number of records to process
            year_start: start year (inclusive)
            year_end: end year (inclusive)
            sample_rate: sampling rate (0-1), e.g., 0.1 means 10%
        """
        import random

        users_data = {}
        reviews_data = []
        hotels_data = {}

        print(f"Reading {json_path}...")
        print(f"Filtering: {year_start}-{year_end}")
        if sample_rate:
            print(f"Sampling: {sample_rate*100}% of data")

        skipped_by_year = 0
        skipped_by_sample = 0

        with open(json_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=878561)):
                if limit and i >= limit:
                    break

                try:
                    review = json.loads(line)

                    # Step 1: Date filter
                    review_date = self.parse_date(review['date'])
                    if review_date:
                        year = int(review_date[:4])
                        if year < year_start or year > year_end:
                            skipped_by_year += 1
                            continue
                    else:
                        skipped_by_year += 1
                        continue

                    # Step 2: Random sampling
                    if sample_rate:
                        if random.random() > sample_rate:
                            skipped_by_sample += 1
                            continue

                    # Extract user info
                    author = review['author']
                    user_id = author['id']
                    if user_id not in users_data:
                        users_data[user_id] = (
                            user_id,
                            author.get('username'),
                            author.get('location'),
                            author.get('num_reviews'),
                            author.get('num_cities'),
                            author.get('num_helpful_votes'),
                            author.get('num_type_reviews')
                        )

                    # Extract hotel info
                    hotel_id = review['offering_id']
                    if hotel_id not in hotels_data:
                        hotels_data[hotel_id] = {
                            'first_date': review_date,
                            'last_date': review_date,
                            'count': 0
                        }
                    hotels_data[hotel_id]['count'] += 1
                    if review_date:
                        if review_date < hotels_data[hotel_id]['first_date']:
                            hotels_data[hotel_id]['first_date'] = review_date
                        if review_date > hotels_data[hotel_id]['last_date']:
                            hotels_data[hotel_id]['last_date'] = review_date

                    # Extract review data
                    ratings = review.get('ratings', {})
                    reviews_data.append((
                        review['id'],
                        hotel_id,
                        user_id,
                        review_date,
                        review.get('date_stayed'),
                        review.get('title'),
                        review.get('text'),
                        review.get('via_mobile'),
                        review.get('num_helpful_votes'),
                        ratings.get('overall'),
                        ratings.get('service'),
                        ratings.get('cleanliness'),
                        ratings.get('value'),
                        ratings.get('location'),
                        ratings.get('sleep_quality'),
                        ratings.get('rooms')
                    ))

                except json.JSONDecodeError:
                    continue

        print(f"\nFiltering Results:")
        print(f"   Skipped by year filter: {skipped_by_year:,}")
        if sample_rate:
            print(f"   Skipped by sampling: {skipped_by_sample:,}")
        print(f"   Kept: {len(reviews_data):,} reviews")
        print(f"   Users: {len(users_data):,}")
        print(f"   Hotels: {len(hotels_data):,}")

        # Batch insert into database
        cursor = self.conn.cursor()

        print("Inserting users...")
        cursor.executemany('''
            INSERT OR IGNORE INTO users
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', users_data.values())

        print("Inserting hotels...")
        hotels_insert = [
            (hid, None, data['first_date'], data['last_date'], data['count'])
            for hid, data in hotels_data.items()
        ]
        cursor.executemany('''
            INSERT OR IGNORE INTO hotels
            VALUES (?, ?, ?, ?, ?)
        ''', hotels_insert)

        print("Inserting reviews...")
        cursor.executemany('''
            INSERT OR IGNORE INTO reviews
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', reviews_data)

        self.conn.commit()
        print("Data import completed!")

        return len(reviews_data)

    def create_sample_database(self, sample_size=5000, output_path='data/reviews_sample.db'):
        """Create a smaller sample database for GitHub."""
        print(f"Creating sample database with {sample_size} reviews...")

        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)

        sample_conn = sqlite3.connect(output_path)
        self.conn.backup(sample_conn)

        # Random sampling
        cursor = sample_conn.cursor()
        cursor.execute(f'''
            DELETE FROM reviews
            WHERE review_id NOT IN (
                SELECT review_id FROM reviews
                ORDER BY RANDOM()
                LIMIT {sample_size}
            )
        ''')

        # Clean up orphaned users and hotels
        cursor.execute('''
            DELETE FROM users
            WHERE user_id NOT IN (SELECT DISTINCT user_id FROM reviews)
        ''')
        cursor.execute('''
            DELETE FROM hotels
            WHERE hotel_id NOT IN (SELECT DISTINCT hotel_id FROM reviews)
        ''')

        sample_conn.commit()

        # Reclaim disk space after bulk deletions
        cursor.execute('VACUUM')

        sample_conn.close()

        print(f"Sample database created: {output_path}")

    def get_stats(self):
        """Get database statistics."""
        cursor = self.conn.cursor()

        stats = {}
        stats['total_reviews'] = cursor.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
        stats['total_hotels'] = cursor.execute('SELECT COUNT(*) FROM hotels').fetchone()[0]
        stats['total_users'] = cursor.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        stats['date_range'] = cursor.execute('''
            SELECT MIN(review_date), MAX(review_date) FROM reviews
        ''').fetchone()

        return stats

    def close(self):
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    processor = HotelReviewDataProcessor('data/reviews.db')

    # 1. Create database
    processor.create_database()

    # 2. Import 2008-2012 data
    num_reviews = processor.load_json_data(
        'data/reviews.json',
        year_start=2008,
        year_end=2012,
        sample_rate=0.1
    )

    # Check count
    print(f"\nTarget: 50K-80K reviews")
    print(f"Actual: {num_reviews:,} reviews")

    if num_reviews < 50000:
        print("Less than 50K, consider expanding year range to 2007-2012")
    elif num_reviews > 100000:
        print("Too many reviews, consider 2009-2012 or add sampling")
    else:
        print("Perfect range!")

    # 3. Create indexes
    processor.create_indexes()

    # 4. Show statistics
    stats = processor.get_stats()
    print("\nDatabase Statistics:")
    print(f"   Reviews: {stats['total_reviews']:,}")
    print(f"   Hotels: {stats['total_hotels']:,}")
    print(f"   Users: {stats['total_users']:,}")
    print(f"   Date Range: {stats['date_range']}")

    # 5. Create sample database (for GitHub)
    processor.create_sample_database(sample_size=5000)

    processor.close()
