import pandas as pd

# Load the filtered kids movies
kids_movies = pd.read_csv('ml-25m/kids_movies.csv')

# Split genres and flatten the list
all_genres = kids_movies['genres'].dropna().str.split('|').sum()

# Get unique genres
unique_genres = sorted(set(all_genres))

# Print results
print("✅ Unique genres in kids_movies.csv:")
for genre in unique_genres:
    print(genre)
