-- hotels table
CREATE TABLE hotels (
    hotel_id INTEGER PRIMARY KEY,
    hotel_name TEXT,
    first_review_date DATE,
    last_review_date DATE,
    total_reviews INTEGER
);

-- users table
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    username TEXT,
    location TEXT,
    num_reviews INTEGER,
    num_cities INTEGER,
    num_helpful_votes INTEGER,
    num_type_reviews INTEGER
);

-- reviews table
CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY,
    hotel_id INTEGER,
    user_id TEXT,
    review_date DATE,
    date_stayed TEXT,  -- keep original as may only have month/year
    title TEXT,
    text TEXT,
    via_mobile BOOLEAN,
    num_helpful_votes INTEGER,
    
    -- ratings
    rating_overall REAL,
    rating_service REAL,
    rating_cleanliness REAL,
    rating_value REAL,
    rating_location REAL,
    rating_sleep_quality REAL,
    rating_rooms REAL,
    
    FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- indexes for performance optimization
CREATE INDEX idx_hotel_id ON reviews(hotel_id);
CREATE INDEX idx_user_id ON reviews(user_id);
CREATE INDEX idx_review_date ON reviews(review_date);
CREATE INDEX idx_rating_overall ON reviews(rating_overall);
CREATE INDEX idx_date_stayed ON reviews(date_stayed);

-- composite indexes for common query patterns
CREATE INDEX idx_hotel_date ON reviews(hotel_id, review_date);
CREATE INDEX idx_hotel_rating ON reviews(hotel_id, rating_overall);