import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import operator

similarity = pickle.load(open('similarity.pkl', 'rb'))
df_dict = pickle.load(open('df_dict.pkl', 'rb'))
df = pd.DataFrame(df_dict)
ratings_dict = pickle.load(open('ratings_dict.pkl', 'rb'))
ratings = pd.DataFrame(ratings_dict)
avg_rating_dict = pickle.load(open('avg_rating_dict.pkl', 'rb'))
avg_rating = pd.DataFrame(avg_rating_dict)

# Adding direct movie ids and poster links because these both popularity_based and weighted_mean algorithms do not get
# affected due to the selected movie_name
# They return same values irrespective of movie selected
# This helps in reducing the number of API calls made

cache = {278: 'https://image.tmdb.org/t/p/w500//q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg',
         550: 'https://image.tmdb.org/t/p/w500//pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg',
         155: 'https://image.tmdb.org/t/p/w500//qJ2tW6WMUDux911r6m7haRef0WH.jpg',
         680: 'https://image.tmdb.org/t/p/w500//fIE3lAGcZDV1G6XM5KmuWnNsPp1.jpg',
         27205: 'https://image.tmdb.org/t/p/w500//edv5CZvWj09upOsy2Y6IwDhK8bt.jpg',
         238: 'https://image.tmdb.org/t/p/w500//3bhkrj58Vtu7enYsRolD1fZdja1.jpg',
         157336: 'https://image.tmdb.org/t/p/w500//gEU2QniE6E77NI6lCU6MxlNBvIx.jpg',
         13: 'https://image.tmdb.org/t/p/w500//saHP97rTPS5eLmrLQEcANmKrsFl.jpg',
         122: 'https://image.tmdb.org/t/p/w500//rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg',
         1891: 'https://image.tmdb.org/t/p/w500//2l05cFWJacyIsTpsqSgH0wQXe4V.jpg',
         11: 'https://image.tmdb.org/t/p/w500//6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg',
         807: 'https://image.tmdb.org/t/p/w500//69Sns8WoET6CfaYlIkHbla4l7nC.jpg'}


def extract_poster(movie_id):
    response = requests.get(
        "https://api.themoviedb.org/3/movie/{}?api_key=79cbc6cd8826ed1f865287854351ab80".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


def fetch_poster(movie_id):
    # Making an API call if poster is not present in the dictionary and adding it to dictionary
    if movie_id not in cache:
        poster = extract_poster(movie_id)
        cache[movie_id] = poster
    return cache[movie_id]


def content_filtering(movie):
    n = 10
    # Getting index of 'movie'
    movie_index = df[df['title'] == movie].index[0]
    # Getting the similarity of 'movie' using it's index
    distances = similarity[movie_index]
    # Sorting it in descending order
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n+1]
    content_movies = []
    content_posters = []
    for j in movie_list:
        movie_id = df.iloc[j[0]].id
        content_movies.append(df.iloc[j[0]].title)
        content_posters.append(fetch_poster(movie_id))
    return content_movies, content_posters


# Creating a new dataframe so as to groupby using ids
testdf = ratings[['id', 'userId', 'rating']]
testdf = testdf[['userId', 'rating']].groupby(testdf['id'])

listOfDictionaries = []
indexMap = {}
reverseIndexMap = {}
ptr = 0

for groupKey in testdf.groups.keys():
    tempDict = {}
    groupDF = testdf.get_group(groupKey)
    for i in range(0, len(groupDF)):
        tempDict[groupDF.iloc[i, 0]] = groupDF.iloc[i, 1]
    indexMap[ptr] = groupKey
    reverseIndexMap[groupKey] = ptr
    ptr = ptr+1
    listOfDictionaries.append(tempDict)

# using DictVectorizer and cosine_similarity
dictVectorizer = DictVectorizer(sparse=True)
vector = dictVectorizer.fit_transform(listOfDictionaries)
pairwiseSimilarity = cosine_similarity(vector)


def collaborative_filtering(movie):
    n = 10
    k = list(df['title'])
    m = list(df['id'])
    movie_index = m[k.index(movie)]
    collaborative_movies = []
    collaborative_posters = []
    row = reverseIndexMap[movie_index]
    counter = 0
    similar = []
    similar.append(df[df['id'] == movie_index]['title'].values[0])
    for j in np.argsort(pairwiseSimilarity[row])[:-2][::-1]:
        if ratings[ratings['id'] == indexMap[j]]['title'].values[0] not in similar and counter < n:
            counter += 1
            similar.append(ratings[ratings['id'] == indexMap[j]]['title'].values[0])
            collaborative_movies.append(ratings[ratings['id'] == indexMap[j]]['title'].values[0])
            collaborative_posters.append(fetch_poster(indexMap[j]))
    return collaborative_movies, collaborative_posters


def hybrid_model(movie):
    n = 10
    z = []
    k = float(1 / n)
    for x in range(n):
        z.append(1 - k * x)
    content, x = content_filtering(movie)
    collab, y = collaborative_filtering(movie)
    dictid = {}
    for x in collab:
        dictid[x] = z[collab.index(x)]
    for x in content:
        if x not in dictid:
            dictid[x] = z[content.index(x)]
        else:
            dictid[x] += z[content.index(x)]
    # Sorting dictid with z in descending order
    id_n = dict(sorted(dictid.items(), key=operator.itemgetter(1), reverse=True))
    counter = 0
    hybrid_movies = []
    hybrid_posters = []
    for x in id_n.keys():
        if counter < n:
            id_m = df.loc[df['title'] == x].reset_index(drop=True).iloc[0]['id']
            hybrid_movies.append(x)
            hybrid_posters.append(fetch_poster(id_m))
            counter += 1
    return hybrid_movies, hybrid_posters


# Creating a pivot_table with id, userId and ratings
matrix = ratings.pivot_table(index='userId', columns='id', values='rating').fillna(0)


def correlation_model(movie):
    n = 10
    # Get id of 'movie'
    id_m = df.loc[df['title'] == movie].reset_index(drop=True).iloc[0]['id']
    row = matrix[id_m]
    # Get pairwise correlation of 'movie'
    correlation = pd.DataFrame(matrix.corrwith(row), columns=['Pearson Corr'])
    corr = correlation.join(avg_rating['ratingCount'])
    # Sorting by 'Pearson Corr' in descending order
    res = corr.sort_values('Pearson Corr', ascending=False).head(n + 1)[1:].index
    correlation_movies = []
    correlation_posters = []
    for j in res:
        correlation_movies.append(df.loc[df['id'] == j].reset_index(drop=True).iloc[0]['title'])
        correlation_posters.append(fetch_poster(j))
    return correlation_movies, correlation_posters


# Creating a new dataframe by using groupby id with id and count as columns
data_model = (ratings.groupby(by=['id'])['rating'].count().reset_index().rename(columns={'rating': 'count'})[['id', 'count']])
# Merging data_model and ratings
result = pd.merge(data_model, ratings, on='id')
# Creating a pivot_table with result dataframe
matrixn = result.pivot_table(index='id', columns='userId', values='rating').fillna(0)
# Creating a csr_matrix
up_matrix = csr_matrix(matrixn)


def nearest_neighbours(movie):
    n = 10
    # Getting 'id' of 'movie'
    id_no = df.loc[df['title'] == movie].reset_index(drop=True).iloc[0]['id']
    # Using NearestNeighbors with cosine metric and brute force algorithm
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(up_matrix)
    # Getting distances and indices of n+1 neighbors as the first one will be itself
    distances, indices = model.kneighbors(matrixn.loc[id_no].values.reshape(1, -1), n_neighbors=n+1)
    neighbours_movies = []
    neighbours_posters = []
    for j in range(0, len(distances.flatten())):
        if j > 0:
            required_id = matrixn.index[indices.flatten()[j]]
            name = df.loc[df['id'] == required_id].reset_index(drop=True).iloc[0]['title']
            neighbours_movies.append(name)
            neighbours_posters.append(fetch_poster(required_id))
    return neighbours_movies, neighbours_posters


def popularity_year(movie):
    # Getting index of 'movie'
    movie_index = df[df['title'] == movie].index[0]
    # Getting release year of the 'movie'
    need_year = df['year'][movie_index]

    if need_year > 1961:
        n = 5
        # Getting movies released in the same year
        res = df[df['year'] == need_year]
        # Sorting by vote_average in descending order
        data = res.sort_values(by='vote_average', ascending=False)
        year_movies = []
        year_posters = []
        counter = 0
        for ind in data.index:
            if counter < n:
                year_movies.append(data["title"][ind])
                year_posters.append(fetch_poster(data["id"][ind]))
                counter += 1
    else:
        year_movies = []
        year_posters = []
    return need_year, year_movies, year_posters


def popularity_based(n):
    # Getting movies whose vote_count are more than 5000
    res = df[df['vote_count'] > 5000]
    # Sorting by vote_average in descending order
    data = res.sort_values(by='vote_average', ascending=False)
    total_movies = []
    total_posters = []
    counter = 0
    for ind in data.index:
        if counter < n:
            total_movies.append(data["title"][ind])
            total_posters.append(fetch_poster(data["id"][ind]))
            counter += 1
    return total_movies, total_posters


def weighted_mean(n):
    v = df['vote_count']                  # number of ratings
    r = df['vote_average']                # rating of the movie
    c = df['vote_average'].mean()         # mean rating
    m = df['vote_count'].quantile(0.90)   # minimum votes required
    # adding a new column weighted_mean whose values are calculated using formula (v*r/(v+m)) + (m*c/(m+v))
    df['weighted_mean'] = (v*r/(v+m)) + (m*c/(m+v))
    # Sorting by weighted_mean in descending order
    data = df.sort_values('weighted_mean', ascending=False)
    weighted_movies = []
    weighted_posters = []
    counter = 0
    for ind in data.index:
        if counter < n:
            weighted_movies.append(data["title"][ind])
            weighted_posters.append(fetch_poster(data["id"][ind]))
            counter += 1
    return weighted_movies, weighted_posters


st.title('Movie Recommendation System')
movies_list = df['title'].values
selected_movie_name = st.selectbox('Type or select a movie from the dropdown', movies_list)

if st.button('Recommend'):
    rec_content_movies, rec_content_posters = content_filtering(selected_movie_name)
    rec_collaborative_movies, rec_collaborative_posters = collaborative_filtering(selected_movie_name)
    rec_hybrid_movies, rec_hybrid_posters = hybrid_model(selected_movie_name)
    rec_correlation_movies, rec_correlation_posters = correlation_model(selected_movie_name)
    rec_neighbour_movies, rec_neighbour_posters = nearest_neighbours(selected_movie_name)
    needed_year, rec_year_movies, rec_year_posters = popularity_year(selected_movie_name)
    rec_total_movies, rec_total_posters = popularity_based(10)
    rec_weight_movies, rec_weight_posters = weighted_mean(10)

    st.subheader("Recommendations for {}".format(selected_movie_name))
    st.subheader("# Content Filtering Method")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_content_posters[0])
        st.text(rec_content_movies[0])
    with col2:
        st.image(rec_content_posters[1])
        st.text(rec_content_movies[1])
    with col3:
        st.image(rec_content_posters[2])
        st.text(rec_content_movies[2])
    with col4:
        st.image(rec_content_posters[3])
        st.text(rec_content_movies[3])
    with col5:
        st.image(rec_content_posters[4])
        st.text(rec_content_movies[4])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_content_posters[5])
        st.text(rec_content_movies[5])
    with col2:
        st.image(rec_content_posters[6])
        st.text(rec_content_movies[6])
    with col3:
        st.image(rec_content_posters[7])
        st.text(rec_content_movies[7])
    with col4:
        st.image(rec_content_posters[8])
        st.text(rec_content_movies[8])
    with col5:
        st.image(rec_content_posters[9])
        st.text(rec_content_movies[9])

    st.subheader("# Collaborative Filtering Method (Item-Item Filtering)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_collaborative_posters[0])
        st.text(rec_collaborative_movies[0])
    with col2:
        st.image(rec_collaborative_posters[1])
        st.text(rec_collaborative_movies[1])
    with col3:
        st.image(rec_collaborative_posters[2])
        st.text(rec_collaborative_movies[2])
    with col4:
        st.image(rec_collaborative_posters[3])
        st.text(rec_collaborative_movies[3])
    with col5:
        st.image(rec_collaborative_posters[4])
        st.text(rec_collaborative_movies[4])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_collaborative_posters[5])
        st.text(rec_collaborative_movies[5])
    with col2:
        st.image(rec_collaborative_posters[6])
        st.text(rec_collaborative_movies[6])
    with col3:
        st.image(rec_collaborative_posters[7])
        st.text(rec_collaborative_movies[7])
    with col4:
        st.image(rec_collaborative_posters[8])
        st.text(rec_collaborative_movies[8])
    with col5:
        st.image(rec_collaborative_posters[9])
        st.text(rec_collaborative_movies[9])

    st.subheader("# Hybrid Method (Content + Collaborative)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_hybrid_posters[0])
        st.text(rec_hybrid_movies[0])
    with col2:
        st.image(rec_hybrid_posters[1])
        st.text(rec_hybrid_movies[1])
    with col3:
        st.image(rec_hybrid_posters[2])
        st.text(rec_hybrid_movies[2])
    with col4:
        st.image(rec_hybrid_posters[3])
        st.text(rec_hybrid_movies[3])
    with col5:
        st.image(rec_hybrid_posters[4])
        st.text(rec_hybrid_movies[4])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_hybrid_posters[5])
        st.text(rec_hybrid_movies[5])
    with col2:
        st.image(rec_hybrid_posters[6])
        st.text(rec_hybrid_movies[6])
    with col3:
        st.image(rec_hybrid_posters[7])
        st.text(rec_hybrid_movies[7])
    with col4:
        st.image(rec_hybrid_posters[8])
        st.text(rec_hybrid_movies[8])
    with col5:
        st.image(rec_hybrid_posters[9])
        st.text(rec_hybrid_movies[9])

    st.subheader("# Correlation Method")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_correlation_posters[0])
        st.text(rec_correlation_movies[0])
    with col2:
        st.image(rec_correlation_posters[1])
        st.text(rec_correlation_movies[1])
    with col3:
        st.image(rec_correlation_posters[2])
        st.text(rec_correlation_movies[2])
    with col4:
        st.image(rec_correlation_posters[3])
        st.text(rec_correlation_movies[3])
    with col5:
        st.image(rec_correlation_posters[4])
        st.text(rec_correlation_movies[4])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_correlation_posters[5])
        st.text(rec_correlation_movies[5])
    with col2:
        st.image(rec_correlation_posters[6])
        st.text(rec_correlation_movies[6])
    with col3:
        st.image(rec_correlation_posters[7])
        st.text(rec_correlation_movies[7])
    with col4:
        st.image(rec_correlation_posters[8])
        st.text(rec_correlation_movies[8])
    with col5:
        st.image(rec_correlation_posters[9])
        st.text(rec_correlation_movies[9])

    st.subheader("# Nearest Neighbours Method")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_neighbour_posters[0])
        st.text(rec_neighbour_movies[0])
    with col2:
        st.image(rec_neighbour_posters[1])
        st.text(rec_neighbour_movies[1])
    with col3:
        st.image(rec_neighbour_posters[2])
        st.text(rec_neighbour_movies[2])
    with col4:
        st.image(rec_neighbour_posters[3])
        st.text(rec_neighbour_movies[3])
    with col5:
        st.image(rec_neighbour_posters[4])
        st.text(rec_neighbour_movies[4])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_neighbour_posters[5])
        st.text(rec_neighbour_movies[5])
    with col2:
        st.image(rec_neighbour_posters[6])
        st.text(rec_neighbour_movies[6])
    with col3:
        st.image(rec_neighbour_posters[7])
        st.text(rec_neighbour_movies[7])
    with col4:
        st.image(rec_neighbour_posters[8])
        st.text(rec_neighbour_movies[8])
    with col5:
        st.image(rec_neighbour_posters[9])
        st.text(rec_neighbour_movies[9])

    if needed_year > 1961:
        st.subheader("Popular In {}".format(needed_year))
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(rec_year_posters[0])
            st.text(rec_year_movies[0])
        with col2:
            st.image(rec_year_posters[1])
            st.text(rec_year_movies[1])
        with col3:
            st.image(rec_year_posters[2])
            st.text(rec_year_movies[2])
        with col4:
            st.image(rec_year_posters[3])
            st.text(rec_year_movies[3])
        with col5:
            st.image(rec_year_posters[4])
            st.text(rec_year_movies[4])

    st.subheader("All Time Favourites")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_total_posters[0])
        st.text(rec_total_movies[0])
    with col2:
        st.image(rec_total_posters[1])
        st.text(rec_total_movies[1])
    with col3:
        st.image(rec_total_posters[2])
        st.text(rec_total_movies[2])
    with col4:
        st.image(rec_total_posters[3])
        st.text(rec_total_movies[3])
    with col5:
        st.image(rec_total_posters[4])
        st.text(rec_total_movies[4])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_total_posters[5])
        st.text(rec_total_movies[5])
    with col2:
        st.image(rec_total_posters[6])
        st.text(rec_total_movies[6])
    with col3:
        st.image(rec_total_posters[7])
        st.text(rec_total_movies[7])
    with col4:
        st.image(rec_total_posters[8])
        st.text(rec_total_movies[8])
    with col5:
        st.image(rec_total_posters[9])
        st.text(rec_total_movies[9])

    st.subheader("# Weighted Mean Method")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_weight_posters[0])
        st.text(rec_weight_movies[0])
    with col2:
        st.image(rec_weight_posters[1])
        st.text(rec_weight_movies[1])
    with col3:
        st.image(rec_weight_posters[2])
        st.text(rec_weight_movies[2])
    with col4:
        st.image(rec_weight_posters[3])
        st.text(rec_weight_movies[3])
    with col5:
        st.image(rec_weight_posters[4])
        st.text(rec_weight_movies[4])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(rec_weight_posters[5])
        st.text(rec_weight_movies[5])
    with col2:
        st.image(rec_weight_posters[6])
        st.text(rec_weight_movies[6])
    with col3:
        st.image(rec_weight_posters[7])
        st.text(rec_weight_movies[7])
    with col4:
        st.image(rec_weight_posters[8])
        st.text(rec_weight_movies[8])
    with col5:
        st.image(rec_weight_posters[9])
        st.text(rec_weight_movies[9])
