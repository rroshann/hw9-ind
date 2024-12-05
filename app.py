
import numpy as np
import pandas as pd
import streamlit as st

class MovieKNN:
    def __init__(self, k=4):
        self.k = k
    
    def prepare_features(self, df):
        # Convert genres if they're not already in list format
        if isinstance(df['genres'].iloc[0], str):
            df['genres'] = df['genres'].str.strip('[]').str.split(',')
            df['genres'] = df['genres'].apply(lambda x: [item.strip() for item in x])

        # Create feature matrix
        # 1. Year as numeric feature
        years = df['year'].values.reshape(-1, 1)

        # 2. Ratings as numeric feature
        ratings = df['average_rating'].values.reshape(-1, 1)

        # 3. Genre similarity (one-hot encoding)
        all_genres = set()
        for genres in df['genres']:
            all_genres.update(genres)

        genre_matrix = np.zeros((len(df), len(all_genres)))
        for i, genres in enumerate(df['genres']):
            for j, genre in enumerate(all_genres):
                if genre in genres:
                    genre_matrix[i, j] = 1

        # Combine features
        self.features = np.hstack([
            years / years.max(), 
            ratings / ratings.max(),  
            genre_matrix  
        ])

        self.movies = df
        return self
    
    def euclidean_distance(self, movie1, movie2):
        return np.sqrt(np.sum((movie1 - movie2) ** 2))
    
    def get_recommendations(self, movie_title):
        # Find the movie index
        movie_idx = self.movies[self.movies['title'] == movie_title].index[0]
        movie_features = self.features[movie_idx]
        
        # Calculate distances to all other movies
        distances = []
        for idx, features in enumerate(self.features):
            if idx != movie_idx:  # Skip the input movie
                dist = self.euclidean_distance(movie_features, features)
                distances.append((idx, dist))
        
        # Sort by distance and get top k
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.k]
        
        # Get the recommended movie titles
        recommendations = [self.movies.iloc[idx]['title'] for idx, _ in neighbors]
        
        return recommendations
    


# Load and prepare data
df = pd.read_csv('./movies.csv')
knn = MovieKNN(k=4)
knn.prepare_features(df)

st.title('Movie Recommendation System')

# Create text input for movie title
selected_movie = st.text_input('Enter a movie title you like:')

if st.button('Get Recommendations'):
    # Check if movie exists in database
    if selected_movie in df['title'].values:
        recommendations = knn.get_recommendations(selected_movie)
        
        st.write("### Based on your selection, we recommend:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.error("Sorry, this movie is not in our database. Please try another movie.")

