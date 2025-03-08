import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load the data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Prepare data
n_users = ratings['user_id'].nunique()
n_movies = ratings['movie_id'].nunique()

# Map user_id and movie_id to indexes
user_map = {id: idx for idx, id in enumerate(ratings['user_id'].unique())}
movie_map = {id: idx for idx, id in enumerate(ratings['movie_id'].unique())}

ratings['user_id'] = ratings['user_id'].map(user_map)
ratings['movie_id'] = ratings['movie_id'].map(movie_map)

# Split the data
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Convert to tensors
train_users = torch.tensor(train_data['user_id'].values, dtype=torch.long)
train_movies = torch.tensor(train_data['movie_id'].values, dtype=torch.long)
train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float)

test_users = torch.tensor(test_data['user_id'].values, dtype=torch.long)
test_movies = torch.tensor(test_data['movie_id'].values, dtype=torch.long)
test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float)

# Define the model
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=8):
        super(MatrixFactorization, self).__init__()
        # Add +1 to n_users to handle new user embeddings
        self.user_embedding = nn.Embedding(n_users + 1, n_factors) 
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.bias_user = nn.Embedding(n_users + 1, 1)
        self.bias_movie = nn.Embedding(n_movies, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, movie):
        user_embedding = self.user_embedding(user)
        movie_embedding = self.movie_embedding(movie)
        pred = (user_embedding * movie_embedding).sum(1)
        pred += self.bias_user(user).squeeze() + self.bias_movie(movie).squeeze() + self.bias
        return pred

# Initialize model
model = MatrixFactorization(n_users, n_movies)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
loss_fn = nn.MSELoss()

# Train the model
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(train_users, train_movies)
    loss = loss_fn(predictions, train_ratings)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_predictions = model(test_users, test_movies)
    test_loss = loss_fn(test_predictions, test_ratings)
    print(f'Test Loss: {test_loss.item():.4f}')

# Recommend movies for a new user based on top 5 favorite movies
def recommend_movies(user_favorites, n=5):
    favorite_movie_ids = [movie_map[movie] for movie in user_favorites if movie in movie_map]
    
    scores = []
    with torch.no_grad():
        for movie_id in range(n_movies):
            if movie_id not in favorite_movie_ids:
                # Use `n_users` as index for the new user embedding
                predicted_rating = model(torch.tensor([n_users]), torch.tensor([movie_id]))
                scores.append((movie_id, predicted_rating.item()))
    
    # Sort by predicted rating
    scores.sort(key=lambda x: x[1], reverse=True)
    recommendations = [movies.loc[movies['movie_id'] == list(movie_map.keys())[x[0]], 'movie_title'].values[0] for x in scores[:n]]
    
    return recommendations

# Example: New user likes Inception (1), The Dark Knight (3), Interstellar (5)
user_favorites = [1, 3, 5]
suggestions = recommend_movies(user_favorites)
print("Recommended Movies:", suggestions)