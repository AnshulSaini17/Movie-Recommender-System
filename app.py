import pickle
import streamlit as st
import requests
import pandas as pd
from tmdbv3api import TMDb
from tmdbv3api import Movie
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv('movies_data.csv')
tmdb = TMDb()
tmdb.api_key = '4f482c58561643445d7340848e36b5e8'

def create_sim():
    data = pd.read_csv('movies_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb']).toarray()
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data,sim


def recommend(movie):
    index = data[data['movie_title']==movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names=[]
    for i in distances[1:6]:
        recommended_movie_names.append(data.iloc[i[0]].movie_title)
    return recommended_movie_names

def posters(movie):
    poster = []
    movie_title_list = []
    r = recommend(movie)
    movie = Movie()
    for movie_title in r:
        list_result = movie.search(movie_title)
        movie_id = list_result[0].id
        url = "https://api.themoviedb.org/3/movie/{}?api_key=4f482c58561643445d7340848e36b5e8&language=en-US".format(movie_id)
        #data = requests.get(url)
        response = requests.get(url)
        data = response.json()
       # st.text(data)
        #st.text(url)
        poster_path = data['poster_path']
        poster.append('https://image.tmdb.org/t/p/original{}'.format(data['poster_path']))
    return poster


st.header('Movie Recommender System')
similarity = pickle.load(open('similarity.pkl','rb'))
movie_list = data['movie_title'].values


selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)


if st.button('Show Recommendation'):
    recommended_movie_names= recommend(selected_movie)
    recommended_movie_posters = posters(selected_movie)

    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

