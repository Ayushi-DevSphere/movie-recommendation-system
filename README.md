# 🎬 Content-Based Movie Recommendation System

A **Content-Based Movie Recommendation System** with a Netflix-style web interface, built using Python, NLP, and Machine Learning. The system analyses movie metadata genres, keywords, cast, crew, and plot overview to recommend similar movies using **TF-IDF Vectorization** and **Cosine Similarity**.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Problem Statement

Given a movie title, recommend the **top-N most similar movies** based on textual metadata. This project demonstrates a complete NLP pipeline from raw data ingestion and feature engineering to vectorization and similarity computation without relying on user ratings or collaborative filtering.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| NLP | NLTK (PorterStemmer, Stopwords) |
| Vectorization | TF-IDF (Scikit-Learn) |
| Similarity | Cosine Similarity |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Dataset | [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |

---

## 🏗️ Architecture

```
┌──────────────┐      ┌─────────────────┐      ┌──────────────────┐
│   Raw CSVs   │─────▶│  Preprocessing  │─────▶│ Feature Engineer │
│  (movies +   │      │  (load, merge,  │      │  (parse JSON,    │
│   credits)   │      │   select cols)  │      │   build tags)    │
└──────────────┘      └─────────────────┘      └────────┬─────────┘
                                                        │
                                                        ▼
                      ┌─────────────────┐      ┌──────────────────┐
                      │ Recommendations │◀─────│  NLP Pipeline    │
                      │ (cosine sim,    │      │  (normalize,     │
                      │  top-N lookup)  │      │   stem, TF-IDF)  │
                      └────────┬────────┘      └──────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │  Netflix-Style  │
                      │  Web Frontend   │
                      │  (Flask + JS)   │
                      └─────────────────┘
```

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── src/
│   ├── __init__.py
│   ├── utils.py           # JSON parsing, text normalization, stemming
│   ├── preprocess.py      # Data loading, merging, feature engineering
│   └── recommender.py     # TF-IDF vectorization, cosine similarity
│
├── static/
│   ├── index.html         # Netflix-style frontend
│   ├── style.css          # Dark theme styling
│   └── script.js          # Client-side logic
│
├── notebook/
│   └── exploration.ipynb  # EDA and pipeline walkthrough
│
├── app.py                 # Flask web server
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔍 How It Works

### 1. Data Preprocessing
- Load **TMDB 5000 Movies** and **Credits** datasets
- Merge on `movie_id`
- Retain: `title`, `overview`, `genres`, `keywords`, `cast` (top 3), `crew` (director)

### 2. Feature Engineering
- Parse JSON-encoded columns (genres, keywords, cast, crew)
- Extract meaningful text values with spaces removed
- Combine all features into a single `tags` column per movie

### 3. NLP Pipeline
- **Normalize**: lowercase, remove punctuation, collapse whitespace
- **Stem**: PorterStemmer to reduce words to root form
- **Stopword Removal**: filter out common English words

### 4. Vectorization & Similarity
- **TF-IDF Vectorizer** (`max_features=5000`, `stop_words='english'`)
- **Cosine Similarity** matrix across all movies
- Given a movie → sort by similarity → return top-N matches

### 5. Web Interface
- **Flask API** serves movie data and recommendations as JSON
- **Netflix-style frontend** with hero banner, genre carousels, search, and detail modals

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Ayushi-DevSphere/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Create Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 💡 Example

### Web Interface
Search for any movie → click on it → get AI-powered recommendations with similarity scores.

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Netflix-style web UI |
| `GET /api/movies/trending` | Top 20 highest-rated movies |
| `GET /api/genres` | All available genres |
| `GET /api/movies/genre/<genre>` | Movies by genre |
| `GET /api/search?q=<query>` | Search movies by title |
| `GET /api/recommend/<title>` | Get similar movie recommendations |
| `GET /api/movie/<id>` | Single movie details |

### Sample API Response

```
GET /api/recommend/The Dark Knight

{
  "movie": "The Dark Knight",
  "recommendations": [
    {"title": "The Dark Knight Rises", "similarity": 0.4775},
    {"title": "Batman Returns", "similarity": 0.3736},
    {"title": "Batman Begins", "similarity": 0.3582},
    {"title": "Batman Forever", "similarity": 0.3286},
    {"title": "Batman: The Dark Knight Returns, Part 2", "similarity": 0.2823}
  ]
}
```

---

## 📊 Dataset

- **Source**: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) on Kaggle
- **Size**: ~4,800 movies
- **Files**: `tmdb_5000_movies.csv` (metadata) + `tmdb_5000_credits.csv` (cast & crew)

> **Note**: Download the datasets from Kaggle and place them in the `data/` directory before running.

---

## 🧠 Key Features

- **Content-based filtering** using movie metadata (no user ratings needed)
- **Full NLP pipeline**: tokenization → stopword removal → stemming → TF-IDF
- **Netflix-style dark UI** with hero banner, genre carousels, and search
- **Case-insensitive** movie title matching
- **RESTful API** for programmatic access
- **Modular codebase** clean separation of concerns

---

## 📝 Resume Description

> Designed and built a **Content-Based Movie Recommendation System** using Python and NLP techniques. Implemented a full preprocessing pipeline to parse and merge TMDB movie metadata, engineered text features from genres, keywords, cast, and crew, and applied TF-IDF vectorization with cosine similarity to deliver accurate movie recommendations. Built a Netflix-style web interface using Flask and vanilla JavaScript. The project follows clean, modular software engineering practices.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Ayushi-DevSphere/movie-recommendation-system/issues).

---

## 👩‍💻 Author

**Ayushi Majumdar**
- GitHub: [@Ayushi-DevSphere](https://github.com/Ayushi-DevSphere)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
