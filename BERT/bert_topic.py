from bertopic import BERTopic


def bertopic_pipeline(processed_text_data):

    # Create a BERTopic model instance
    topic_model = BERTopic()

    # Prepare the input data as a list of joined sentences
    sentence_list = [' '.join(doc) for doc in processed_text_data]

    # Fit the model on the processed text data
    topic_model.fit_transform(sentence_list)

    # Print information about the topics generated by the model
    print(topic_model.get_topic_info())
    
    # Visualize the topics and save the visualization to an HTML file
    fig = topic_model.visualize_topics(width=750, height=750)
    fig.write_html("BERT/bert_visualization.html")
