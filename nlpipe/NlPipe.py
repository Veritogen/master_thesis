import spacy
import multiprocessing
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV
import pyLDAvis.sklearn
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt


class NlPipe:
    def __init__(self, list_of_docs, document_ids=None, language_model="en_core_web_lg", tagger=False, parser=False,
                 ner=False, categorization=False, remove_stopwords=True, remove_punctuation=True, set_lower=True,
                 remove_num=True, expand_stopwords=True):
        """
        :param list_of_docs: List of strings where every document is one string.
        :param document_ids: The ids of the documents, matching the order of the list_of_docs
        :param language_model: Spacy language model to be used for text preprocessing
        :param tagger: Use spacy part-of-speech tagger.
        :param parser: Use spacy to annotate syntactic dependencies in documents.
        :param ner: Use spacy for entity recognition and annotation.
        :param categorization: Use spacy to assign document labels
        :param remove_stopwords: Remove stop words during text preprocessing.
        :param remove_punctuation: Remove punctuation during text prssing.
        :param set_lower: Convert all strings to lowercase during text preprocessing.
        :param remove_num: Remove numeric characters during text preprocessing.
        :param expand_stopwords: Remove non-alpha-characters in stop words and add them to the stop words.
        """

        self.pipe_disable = []
        if not tagger:
            self.pipe_disable.append("tagger")
        if not parser:
            self.pipe_disable.append("parser")
        if not ner:
            self.pipe_disable.append("ner")
        if not categorization:
            self.pipe_disable.append("textcat")
        self.remove_punctuation = remove_punctuation
        self.remove_stop_words = remove_stopwords
        self.remove_num = remove_num
        self.set_lower = set_lower
        self.input_docs = list_of_docs
        self.document_ids = document_ids
        self.nlp = spacy.load(language_model)
        if expand_stopwords:
            stops = [stop for stop in self.nlp.Defaults.stop_words]
            for stop in stops:
                self.nlp.Defaults.stop_words.add(re.sub(r"[\W]", "", stop))
        self.spacy_docs = None
        self.processed_docs = None
        self.vectorizer = None
        self.bag_of_words = None
        self.tf_idf = None
        self.preprocessing_batch_size = 50
        self.processes = multiprocessing.cpu_count()-2
        self.lda_model = None
        self.lda_output = None
        self.grid_search = None
        self.evaluation_output = None
        self.evaluation_df = None

    def enable_pipe_component(self, component):
        if component in self.pipe_disable:
            self.pipe_disable.remove(component)
            #todo: add info if not in list from beginning or if successfully enable

    def disable_pipe_component(self, component):
        if component not in self.pipe_disable:
            self.pipe_disable.append(component)
            # todo: add info if not in list from beginning or if successfully enable

    def preprocess_spacy(self):
        self.spacy_docs = [doc for doc in tqdm(self.nlp.pipe(self.input_docs, disable=self.pipe_disable,
                                                             n_process=self.processes,
                                                             batch_size=self.preprocessing_batch_size),
                                               desc="Preprocessing text with spacy: ")]

    def preprocess(self):
        self.processed_docs = []
        if not self.spacy_docs:
            self.preprocess_spacy()
        for spacy_doc in tqdm(self.spacy_docs, desc="Removing stop words/punctuation/numeric chars: "):
            doc = []
            for token in spacy_doc:
                if not self.remove_stop_words and token.is_stop:
                    word = token.text
                elif token.is_stop:
                    continue
                else:
                    word = token.text
                if self.set_lower:
                    word = word.lower()
                if self.remove_num:
                    word = re.sub(r"[\d]", "", word)
                if self.remove_punctuation:
                    word = re.sub(r"[\W]", "", word)
                if len(word) >= 2:
                    doc.append(word)
            self.processed_docs.append(doc)

    def create_bag_of_words(self, n_grams=(1, 1), min_df=0.01, max_df=0.6):
        self.preprocess_spacy()
        self.preprocess()
        joined_docs = []
        for doc in self.processed_docs:
            joined_docs.append(" ".join(doc))
        self.vectorizer = CountVectorizer(lowercase=False, ngram_range=n_grams, min_df=min_df,
                                     max_df=max_df)
        self.bag_of_words = self.vectorizer.fit_transform(joined_docs)

    def create_tf_idf(self, n_grams=(1, 1), min_df=0.01, max_df=0.6):
        self.preprocess_spacy()
        self.preprocess()
        joined_docs = []
        for doc in self.processed_docs:
            joined_docs.append(" ".join(doc))
        self.vectorizer = TfidfVectorizer(lowercase=False, ngram_range=n_grams, min_df=min_df, max_df=max_df)
        self.tf_idf = self.vectorizer.fit_transform(joined_docs)

    def create_lda_model(self, no_topics=10, input_type="bag"):
        self.lda_model = LDA(n_jobs=self.processes, n_components=no_topics)
        if input_type == "bag":
            self.create_bag_of_words()
            self.lda_output = self.lda_model.fit_transform(self.bag_of_words)
        else:
            self.create_tf_idf()
            self.lda_output = self.lda_model.fit_transform(self.tf_idf)

    def search_best_model(self, n_components=[2, 3, 4, 5, 10, 15, 20, 25], learning_decay=[.5, .7, .9], input_type="bag"):
        lda_model = LDA()
        self.grid_search = GridSearchCV(lda_model, {"n_components": n_components, "learning_decay": learning_decay})
        if input_type == "bag":
            if self.bag_of_words is None:
                self.create_bag_of_words()
            self.grid_search.fit(self.bag_of_words)
        else:
            if self.tf_idf is None:
                self.create_tf_idf()
            self.grid_search.fit(self.tf_idf)

    def create_document_topic_df(self, model=None, no_topics=10, input_type="bag", input_matrix=None):
        if model is None:
            self.create_lda_model(no_topics=no_topics, input_type=input_type)
        else:
            self.lda_model = model
        if input_matrix is not None:
            self.evaluation_output = self.lda_model.fit_transform(input_matrix)
        elif input_type == "bag":
            self.evaluation_output = self.lda_model.fit_transform(self.bag_of_words)
        else:
            self.evaluation_output = self.lda_model.fit_transform(self.tf_idf)
        self.evaluation_df = pd.DataFrame(self.evaluation_output)
        if self.document_ids is not None:
            self.evaluation_df.index = self.document_ids
        dominant_topic = np.argmax(self.evaluation_df.values, axis=1)
        self.evaluation_df['dominant_topic'] = dominant_topic

    def plot_document_topic_distribution(self):
        counter = Counter(self.evaluation_df.dominant_topic)
        topic_dict = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
        sns.barplot(x=list(topic_dict.values()), y=list(topic_dict.keys()), order=list(topic_dict.keys()),orient='h')
        plt.show()

    def evaluate_lda_model(self):
        panel = pyLDAvis.sklearn.prepare(self.lda_model, self.bag_of_words, self.vectorizer)
        pyLDAvis.show(panel)
