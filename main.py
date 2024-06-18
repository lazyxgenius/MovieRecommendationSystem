import numpy as np
import pandas as pd
import difflib  # used to get clossest match of some value
from sklearn.feature_extraction.text import TfidfVectorizer  #used to convert text data into numerical value
from sklearn.metrics.pairwise import cosine_similarity  #gives similarity score

# data collection
md = pd.read_csv(r"C:\Users\Aditya PC\PycharmProjects\movie_recommendation\dataset\movies.csv")
md.head()

# md.shape
md.info()

# selecting relevant features -- this recommendation system will be content based and popularity based system
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director', 'overview']
print(selected_features)

# replacing the null values with null string
for feature in selected_features:
    md[feature] = md[feature].fillna('')

# combining all the selected feature
combined_features = md['genres'] + ' ' + md['keywords'] + ' ' + md['tagline'] + ' ' + md['cast'] + ' ' + md[
    'director'] + ' ' + md['overview']
print(combined_features)

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()

feature_vector = vectorizer.fit_transform(combined_features)
print(feature_vector)

# cosine similarity - getting similarity score - similarity confidence value
similarity = cosine_similarity(feature_vector)
print(similarity)

print(similarity.shape)

# gettning the movie name from the user
movie_name = input('Enter your favourite movie name :')

# creating a list with all the movies given in the dataset
list_of_all_titles = md['title'].tolist()
print(list_of_all_titles)

# finding the close match for the movie name given by the user
find_close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_matches)

close_match = find_close_matches[0]
print(close_match)

# finding the index of the movie with title

index_of_the_movie = md[md.title == close_match]['index'].values[0]
print(index_of_the_movie)

# getting a list of similarity scores with all the movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

len(similarity_score)

# sorting this list based on similarity score
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
print(sorted_similar_movies)

# suggest top 10 similar movies to user
print('Movies suggested for you: \n')

i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = md[md.index == index]['title'].values[0]
    if (i < 11):
        print(i, '.', title_from_index)
        i = i + 1
