from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def tf_idf(processed_text_data):

    # Combine the list of list to a list of strings
    sentence_list = [' '.join(doc) for doc in processed_text_data]

    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(min_df=1)

    # Fit the vectorizer and transform the data
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence_list)

    # Get the feature names (vocabulary)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names
