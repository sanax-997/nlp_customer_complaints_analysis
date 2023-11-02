import pandas as pd
from data_preprocessing import preprocess_text
from tfidf_lda.tfidf import tf_idf
from tfidf_lda.lda import perform_lda
from BERT.bert_topic import bertopic_pipeline

if __name__ == "__main__":
    # Read the text data from the csv file
    text_data = pd.read_csv('Data/complaints_data.csv')['text'].values.tolist()

    # Turn the unstructured text data into structured clean text
    processed_text_data = preprocess_text(text_data)

    # Choose to either perform tfidf and lda or BERTopic
    nlp_method = "Bertopic"

    # Perform nlp via tfidf, lda and visualize the results
    if nlp_method == "tfidf_lda":
        tfidf_matrix, feature_names = tf_idf(
            processed_text_data)
        lda_model = perform_lda(tfidf_matrix, feature_names,
                                processed_text_data, topic_num=4)

    # Perform nlp via the BERTopic pipeline
    if nlp_method == "Bertopic":
        bertopic_pipeline(processed_text_data)
