import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

# ---- Load Data ----
movies = pd.read_csv('small-data-set/movies.csv')
ratings = pd.read_csv('small-data-set/ratings.csv')
genome_scores = pd.read_csv('small-data-set/genome-scores.csv')

# ---- Map IDs ----
n_users = ratings['user_id'].nunique()
n_movies = ratings['movie_id'].nunique()

user_map = {id: idx for idx, id in enumerate(ratings['user_id'].unique())}
movie_map = {id: idx for idx, id in enumerate(ratings['movie_id'].unique())}

ratings['user_id'] = ratings['user_id'].map(user_map)
ratings['movie_id'] = ratings['movie_id'].map(movie_map)

# ---- Split Data ----
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

train_users = torch.tensor(train_data['user_id'].values, dtype=torch.long)
train_movies = torch.tensor(train_data['movie_id'].values, dtype=torch.long)
train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float)

test_users = torch.tensor(test_data['user_id'].values, dtype=torch.long)
test_movies = torch.tensor(test_data['movie_id'].values, dtype=torch.long)
test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float)

# ---- GPU Support ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- DataLoader ----
batch_size = 1024
train_dataset = TensorDataset(train_users, train_movies, train_ratings)
test_dataset = TensorDataset(test_users, test_movies, test_ratings)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---- Model ----
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=64):
        super(MatrixFactorization, self).__init__()
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

# ---- Initialize Model ----
model = MatrixFactorization(n_users, n_movies).to(device)
model = torch.compile(model)  # For PyTorch 2.x
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
scaler = GradScaler()

# ---- Train Model ----
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    
    for users, movies, ratings in train_loader:
        users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
        
        optimizer.zero_grad()
        with autocast():
            predictions = model(users, movies)
            loss = loss_fn(predictions, ratings)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}: Loss = {avg_loss:.4f}')

# ---- Evaluate Model ----
model.eval()
with torch.no_grad():
    total_loss = 0
    for users, movies, ratings in test_loader:
        users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
        predictions = model(users, movies)
        loss = loss_fn(predictions, ratings)
        total_loss += loss.item()

    test_loss = total_loss / len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

# ---- Precompute Genre Weights ----
movie_genre_weights = genome_scores.groupby('movie_id').apply(
    lambda x: {row['tag_id']: row['relevance'] for _, row in x.iterrows()}
).to_dict()

def get_favorite_genre_weights(favorite_movies):
    genre_weights = {}
    for movie in favorite_movies:
        if movie in movie_map:
            genres = movie_genre_weights.get(movie, {})
            for tag, relevance in genres.items():
                genre_weights[tag] = genre_weights.get(tag, 0) + relevance

    total_weight = sum(genre_weights.values())
    if total_weight > 0:
        for tag in genre_weights:
            genre_weights[tag] /= total_weight

    return genre_weights

def recommend_movies(user_favorites, n=5):
    favorite_movie_ids = [movie_map[movie] for movie in user_favorites if movie in movie_map]
    genre_weights = get_favorite_genre_weights(user_favorites)

    scores = []
    with torch.no_grad():
        for movie_id in range(n_movies):
            if movie_id not in favorite_movie_ids:
                pred = model(torch.tensor([n_users], device=device), torch.tensor([movie_id], device=device)).item()
                genre_score = sum(genre_weights.get(tag, 0) * row['relevance'] for tag, row in movie_genre_weights.get(movie_id, {}).items())
                scores.append((movie_id, pred + genre_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [movies.loc[movies['movie_id'] == list(movie_map.keys())[x[0]], 'movie_title'].values[0] for x in scores[:n]]

# ---- Example Usage ----
user_favorites = [1, 3, 5]
suggestions = recommend_movies(user_favorites)
print("Recommended Movies:", suggestions)
