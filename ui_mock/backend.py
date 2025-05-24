import asyncio
import json
import csv
import uuid
import websockets
import random
from pathlib import Path

class MovieRecommendationBackend:
    def __init__(self, csv_file_path):
        self.films = {}  # Film catalog
        self.user_ratings = {}  # User ratings storage
        self.load_films_from_csv(csv_file_path)
        self.recommender_uri = "ws://localhost:8766"

    async def initialize_recommender(self):
        """Initialize the recommender with film data"""
        try:
            async with websockets.connect(self.recommender_uri) as websocket:
                # Send film data to recommender
                await websocket.send(json.dumps({
                    "type": "load_films",
                    "films": self.films
                }))

                # Wait for confirmation
                response = await websocket.recv()
                data = json.loads(response)
                if data["type"] == "films_loaded" and data["status"] == "success":
                    print("Recommender initialized successfully")
                else:
                    print("Failed to initialize recommender")
        except Exception as e:
            print(f"Error initializing recommender: {e}")

    def load_films_from_csv(self, csv_file_path):
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            films_list = [row for row in reader]  # Read all rows into a list

        random.shuffle(films_list)  # Shuffle the list of films

        for i, row in enumerate(films_list):
            film_id = f"film_{i}"  # Generate simple IDs
            self.films[film_id] = {
                "title": row["Title"],
                "poster_url": row["Poster"]
            }

        print(f"Loaded {len(self.films)} films from CSV")

    def get_films_for_rating(self, user_id, count=5):
        """Get films for the user to rate"""
        # Get films the user hasn't rated yet
        rated_film_ids = set()
        if user_id in self.user_ratings:
            rated_film_ids = {rating["film_id"] for rating in self.user_ratings[user_id]["rated_films"]}

        # Select films not yet rated
        available_films = [
            {"film_id": film_id, **film_data}
            for film_id, film_data in self.films.items()
            if film_id not in rated_film_ids
        ]

        # Return the first 'count' films
        return available_films[:count]

    def add_rating(self, user_id, film_id, rating):
        """Add a film rating for a user"""
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {
                "rated_films": [],
                "recommendations": []
            }

        # Skip if rating is "skip"
        if rating == "skip":
            return len(self.user_ratings[user_id]["rated_films"])

        # Add the rating
        if film_id in self.films:
            self.user_ratings[user_id]["rated_films"].append({
                "film_id": film_id,
                "title": self.films[film_id]["title"],
                "rating": rating
            })

        return len(self.user_ratings[user_id]["rated_films"])

    async def get_recommendations(self, user_id):
        """Get recommendations for a user who has rated 5 films"""
        if user_id not in self.user_ratings:
            return []

        # Check if user has rated 5 films
        if len(self.user_ratings[user_id]["rated_films"]) < 5:
            return []

        try:
            # Connect to recommender service
            async with websockets.connect(self.recommender_uri) as websocket:
                # Request recommendations
                await websocket.send(json.dumps({
                    "type": "get_recommendations",
                    "user_id": user_id,
                    "rated_films": self.user_ratings[user_id]["rated_films"],
                    "count": 3
                }))

                # Wait for recommendations
                response = await websocket.recv()
                data = json.loads(response)

                if data["type"] == "recommendations" and data["user_id"] == user_id:
                    # Store recommendations for the user
                    self.user_ratings[user_id]["recommendations"] = data["films"]
                    return data["films"]
                else:
                    print(f"Unexpected response from recommender: {data}")
                    return []
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            # Fallback to local recommendations if recommender service fails
            return self._fallback_recommendations(user_id)

    def _fallback_recommendations(self, user_id):
        """Fallback method if recommender service fails"""
        rated_film_ids = {rating["film_id"] for rating in self.user_ratings[user_id]["rated_films"]}

        # Get first 3 unrated films as recommendations
        recommendations = []
        for film_id, film_data in self.films.items():
            if film_id not in rated_film_ids:
                recommendations.append({
                    "film_id": film_id,
                    "title": film_data["title"],
                    "poster_url": film_data["poster_url"]
                })
                if len(recommendations) >= 3:
                    break

        return recommendations

# Create a global backend instance
backend = None

# WebSocket server handler
async def handler(websocket):
    global backend  # Use the global backend instance
    user_id = str(uuid.uuid4())  # Generate a unique user ID
    print(f"New connection: {user_id}")

    try:
        # Send initial films for rating
        initial_films = backend.get_films_for_rating(user_id)
        await websocket.send(json.dumps({
            "type": "films_to_rate",
            "films": initial_films
        }))

        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "rating":
                film_id = data["film_id"]
                rating = data["rating"]  # "thumbs_up", "thumbs_down", or "skip"

                # Add the rating
                ratings_count = backend.add_rating(user_id, film_id, rating)

                # If user skipped, send a new film
                if rating == "skip":
                    new_films = backend.get_films_for_rating(user_id, 1)
                    if new_films:
                        await websocket.send(json.dumps({
                            "type": "additional_film",
                            "film": new_films[0]
                        }))

                # Check if we have 5 ratings now
                if ratings_count == 5:
                    # Get recommendations
                    recommendations = await backend.get_recommendations(user_id)
                    await websocket.send(json.dumps({
                        "type": "recommendations",
                        "films": recommendations
                    }))

            elif data["type"] == "get_more_films":
                # User wants more films to rate
                more_films = backend.get_films_for_rating(user_id, 5)
                await websocket.send(json.dumps({
                    "type": "films_to_rate",
                    "films": more_films
                }))

    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed for user: {user_id}")

async def main():
    global backend
    # Initialize the backend
    backend = MovieRecommendationBackend("movies.csv")

    # Initialize the recommender
    await backend.initialize_recommender()

    # Start WebSocket server
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
