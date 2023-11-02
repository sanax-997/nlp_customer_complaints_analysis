import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import enchant
import re

from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS


def preprocess_text(text_data):
    # Download NLTK resources (stopwords and lemmatizer)
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Initialize stopwords, lemmatizer, and spell checker
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    spell_checker = enchant.Dict("en_US")

    # Tokenize the text
    documents = [text.split(' ') if isinstance(text, str) else []
                 for text in text_data]
    documents = [text for text in documents if text]

    document_list = []

    # Apply text processing on the tokens
    for document in documents:
        world_list = []
        for word in document:
            lemmatizer.lemmatize(word)
            word = word.lower()
            word = re.sub(r'[^a-zA-Z\s]', '', word)
            if word != '' and word not in stop_words and spell_checker.check(word):
                world_list.append(word)
        document_list.append(world_list)

    # Create phrases of n-grams
    phrases = Phrases(document_list, min_count=1, threshold=0.1,
                      connector_words=ENGLISH_CONNECTOR_WORDS)

    # Replace the individual tokens with the phrases
    for i, document in enumerate(document_list):
        document_list[i] = list(phrases[document])

    return document_list
