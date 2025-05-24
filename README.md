# movie-kompis

# 🎬 Movie Kompis - ML Draft

Movie recommendation system tailored for kids using collaborative and content-based filtering with machine learning.

## 📁 1. Datasets Used

- `kids_ratings.csv`: User ratings dataset (sourced from [MovieLens.org](https://movielens.org))
- `kids_movies.csv`: Movie metadata (title, year, genres, summary, etc.)

## 🧹 2. Data Preprocessing

- Filtered for kid-friendly movies only
- Merged ratings with metadata
- Normalized and cleaned the data
- Removed low-activity users and less-rated movies

## 🤝 3. Collaborative Filtering Approach

- Constructed a user-item rating matrix
- Applied matrix factorization using SVD

## 🧠 4. Content-Based Filtering Approach

- Used metadata: genres, summary, year (ongoing improvement)
- Applied TF-IDF vectorization on summaries
- Used cosine similarity to compute movie closeness

## 🔄 5. Continuous Improvement

- Plan to regularly retrain the model as new ratings/movies are added
- Improve dataset scope and accuracy over time

## 🎯 6. Final Output

- Recommends top-N kid-friendly movies per user
- Offers explainable suggestions (e.g., similar genre or theme)

## 🚧 7. Future Improvements

- Add age rating and language filters
- Personalize using watch history and usage time
