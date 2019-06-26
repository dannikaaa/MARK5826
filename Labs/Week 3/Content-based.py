from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from utils import *


def tfidf_converter(df):
    # Change the NaN Values into Empty Strings
    df['tagline'] = df['tagline'].fillna('')
    # We only Use the "Overview" and "TagLine" features to Represent Each Movie
    df['description'] = df['overview'] + df['tagline']
    df['description'] = df['description'].fillna('')

    # TF-IDF Matrix
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['description'])

    print("TF-IDF Matrix Shape: " + str(tfidf_matrix.shape))
    print("=" * 120)
    return tfidf_matrix


def cosSimilarity(matrix):
    similarity_matrix = cosine_similarity(matrix)
    print("Cosine Similarity Matrix Sample:")
    print(similarity_matrix[:5, :5])
    print("=" * 120)
    return similarity_matrix


def recForOneItem(movie_list_df, similarity_matrix, title, num_rec):
    # Get the Indices for Each Title
    idx = movie_list_df[title]

    # Get a List of Tuples, Where Each Tuple Contains A Movie Index and The Similarity Score
    similarity_score = list(enumerate(similarity_matrix[idx]))

    # Sort the Above List of Tuples by The Similarity Score in Descendant Order
    similarity_score = heapq.nlargest(num_rec+1, similarity_score, key=lambda tup: tup[1])

    # Get the Titles of the Recommended Movies
    movie_indices = [item[0] for item in similarity_score[1:]]

    return movie_list_df.iloc[movie_indices]

if __name__ == '__main__':
    data, movie_list = cb_loadData()
    tfidf_matrix = tfidf_converter(data)
    similarity = cosSimilarity(tfidf_matrix)
    recommendation = recForOneItem(movie_list, similarity, 'The Godfather', 10)
    print("Recommendation List:")
    print(recommendation)
