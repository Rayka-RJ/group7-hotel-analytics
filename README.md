# group7-hotel-analytics

IS5126 Assignment 1 & 2 for Group 7 — Hotel review analytics platform for HospitalityTech Solutions.

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

**Data:** The repo includes `data/reviews_sample.db` (5,000+ reviews) for immediate use. For the full dataset (50K–80K+ reviews), run `notebooks/01_data_preparation.ipynb` — this requires `data/reviews.json` (raw data, not in repo).

---

## Usage

### Dashboard (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

The dashboard uses `reviews.db` if present, otherwise `reviews_sample.db`.

### Notebooks

Run in order for full pipeline:

| Notebook | Purpose |
|----------|---------|
| `01_data_preparation.ipynb` | Build DB from raw JSON (needs `reviews.json`) |
| `02_exploratory_analysis.ipynb` | EDA and insights |
| `03_competitive_benchmarking.ipynb` | Clustering, comparable hotels, recommendations |
| `04_performance_profiling.ipynb` | Query and code profiling |

**Note:** Notebooks 02–04 use `reviews.db` by default. If only `reviews_sample.db` exists, they automatically fall back to it.

---

## Dashboard Features

1. **Overview** — Total reviews, hotels, users, date range  
2. **Hotel Performance Explorer** — Rank hotels, view top/bottom performers  
3. **Competitive Benchmarking** — Find comparable hotels, get improvement recommendations  
4. **Rating Trends** — Monthly review volume over time  
5. **Best Practices** — Compare top vs bottom performers  

---

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── reviews_sample.db    # 5K+ sample (included)
│   └── data_schema.sql
├── notebooks/               # 01–04 analysis pipeline
├── src/                     # data_processing, benchmarking, utils
├── app/streamlit_app.py     # Dashboard
├── profiling/               # Query & code profiling outputs
└── reports/
```

---

## Assignment Link

https://prakashsukhwal.github.io/IS5126/ASSIGNMENT_STUDENT_FINAL-v2.html
