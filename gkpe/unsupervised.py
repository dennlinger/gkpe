"""
Contains several unsupervised KPE methods.
"""
import spacy
from typing import Tuple
from functools import lru_cache
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


@lru_cache(4)
def get_spacy(model: str = "de_core_news_sm", disable: Tuple[str] = ()):
    return spacy.load(model, disable=disable)


class sentence_tf_idf:
    """
    Baseline that computes sentence-level TF-IDF scores,
    which is useful to compute tags for a single document only.
    """

    def __init__(self):
        self.nlp = get_spacy(disable=("ner",))

    def extract_keywords(self, text: str, max_n_gram: int = 3, number_keyphrases: int = 5, lemmatize: bool = True):
        """

        :param text: Content of a single document, in raw form.
        :param max_n_gram: Maximum n-gram length of a single keyphrase.
        :param number_keyphrases: Number of keyphrases that get returned.
        :param lemmatize: If enabled, will compute scores on lemmatized tokens.
        :return: Top keyphrases extracted from text.
        """

        tfidf = TfidfVectorizer(ngram_range=(1, max_n_gram))

        doc = self.nlp(text)

        # Adjust corpus input for sklearn by splitting into sentences.
        if lemmatize:
            # TODO: Change output such that only lemmatized tokens are respected
            raise NotImplementedError("Functionality with lemmatization not yet available")
        else:
            corpus = [sentence.text for sentence in doc.sents]

        doc_matrix = tfidf.fit_transform(corpus)

        print("Test")

        return None



