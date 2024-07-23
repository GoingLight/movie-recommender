import streamlit as st
import pandas as pd 
import zipfile
import requests
import io 
from sklearn.metrics.pairwise import cosine_similarity

#function to load the data

def load_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    ratings = pd.read_csv(zip_file.open('ml-latest-small/ratings.csv'))
    movies = pd.read_csv(zip_file.open('ml-latest-small/movies.csv'))

    data = pd.merge (ratings, movies, on ="movieId")
    return data, movies

#function to build matrices

def build_matrices(data):
    user_item_matrix = data.pivot_table (index='userId', columns ="title", values="rating").fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns= user_item_matrix.index)
    return user_item_matrix, user_similarity_df

#function for getting recommendations

def get_recommendation(selected_movies, user_item_matrix, num_recommendations=5):

    movie_ratings = user_item_matrix[selected_movies].mean(axis=1)
    similar_users = movie_ratings.sort_values(ascending=False).index
    similar_users_ratings = user_item_matrix.loc[similar_users]

    weighted_ratings = similar_users_ratings.T.dot(movie_ratings)
    weighted_ratings = weighted_ratings / movie_ratings.sum()

    recommendations = weighted_ratings.sort_values(ascending=False).head(num_recommendations)

    return recommendations

data, movies = load_data()
user_item_matrix, user_similarity_df = build_matrices(data)


#streamlit app 

st.title("Get your movie recommendation")

selected_movies = st.multiselect("Select your fav movies", movies['title'].unique())

if st.button("Give me"):
    if selected_movies:
        recommendations = get_recommendation(selected_movies, user_item_matrix)
        st.write("Top recommendations:")
        for movie, score in recommendations.items():
            st.write(f"{movie}: {score:.2f}")
    else: 
        st.write("Please select at leasty one movie")