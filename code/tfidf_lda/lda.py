from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import np
from collections import Counter


def perform_lda(tfidf_matrix, feature_names, processed_text_data, topic_num):

    # Initialize LDA with the specified number of topics
    lda_model = LatentDirichletAllocation(
        n_components=topic_num, random_state=42)

    # Fit LDA model to the TF-IDF matrix
    lda_top = lda_model.fit(tfidf_matrix)

    # Calculate the matrix of topic-term probabilities
    topic_term_dists = lda_model.components_ / \
        lda_model.components_.sum(axis=1)[:, np.newaxis]

    # Calculate the matrix of document-topic probabilities
    doc_topic_dists = lda_model.transform(tfidf_matrix)

    # Calculate the length of each document
    doc_lengths = [len(document) for document in processed_text_data]

    # Initialize the list of all the words in the corpus used to train the model
    vocab = feature_names

    # Calculate the count of each particular term over the entire corpus
    term_frequency = Counter(
        word for document in processed_text_data for word in document)
    term_frequency = [term_frequency[word] for word in vocab]

    # Transform and prepare a LDA model’s data for visualization
    vis = pyLDAvis.prepare(
        topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency, R=50)

    # Save the visualization as an HTML file
    pyLDAvis.save_html(vis, 'tfidf_lda/lda_visualization.html')
