import pandas as pd

# --- Load Data ---
movies = pd.read_csv('ml-25m/movies.csv')
ratings = pd.read_csv('ml-25m/ratings.csv')

# --- Filter movies that contain the exact genre "Children" ---
children_movies = movies[movies['genres'].str.contains(r'\bChildren\b', na=False)]

print(f"✅ Found {len(children_movies)} movies with 'Children' genre")

# --- Save filtered kids movies ---
children_movies.to_csv('ml-25m/kids_movies.csv', index=False)
print("✅ Saved: ml-25m/kids_movies.csv")

# --- Filter ratings only for those movies ---
kids_ratings = ratings[ratings['movieId'].isin(children_movies['movieId'])]
kids_ratings.to_csv('ml-25m/kids_ratings.csv', index=False)
print("✅ Saved: ml-25m/kids_ratings.csv")
