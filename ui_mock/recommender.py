import asyncio
import json
import websockets
from pathlib import Path

class MovieRecommender:
    def __init__(self):
        self.films = {}  # Film catalog

    def load_films(self, films_data):
        """Load films data received from backend"""
        self.films = films_data
        print(f"Recommender loaded {len(self.films)} films")

    def generate_recommendations(self, user_id, rated_films, count=3):
        """Generate recommendations based on user ratings"""
        # Extract film IDs that the user has already rated
        rated_film_ids = {rating["film_id"] for rating in rated_films}

        # Get first 'count' unrated films as recommendations
        recommendations = []
        for film_id, film_data in self.films.items():
            if film_id not in rated_film_ids:
                recommendations.append({
                    "film_id": film_id,
                    "title": film_data["title"],
                    "poster_url": film_data["poster_url"]
                })
                if len(recommendations) >= count:
                    break

        return recommendations

# Create a global recommender instance
recommender = MovieRecommender()

# WebSocket server handler for recommender
async def handler(websocket):
    global recommender

    try:
        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "load_films":
                # Load films data sent from backend
                recommender.load_films(data["films"])
                await websocket.send(json.dumps({
                    "type": "films_loaded",
                    "status": "success"
                }))

            elif data["type"] == "get_recommendations":
                user_id = data["user_id"]
                rated_films = data["rated_films"]
                count = data.get("count", 3)

                # Generate recommendations
                # This is where the ML recommender provides with the recommendations
                # based on the rated films.
                recommendations = recommender.generate_recommendations(user_id, rated_films, count)

                # Send recommendations back
                await websocket.send(json.dumps({
                    "type": "recommendations",
                    "user_id": user_id,
                    "films": recommendations
                }))

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")

async def main():
    # Start WebSocket server
    async with websockets.serve(handler, "0.0.0.0", 8766):
        print("Recommender WebSocket server started on ws://localhost:8766")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
