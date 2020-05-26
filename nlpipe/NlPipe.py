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
from langdetect import detect

class NlPipe:
    def __init__(self, list_of_docs, document_ids=None, language_model="en_core_web_lg", tagger=False, parser=False,
                 ner=False, categorization=False, remove_stopwords=True, remove_punctuation=True, set_lower=True,
                 remove_num=True, expand_stopwords=True, language_detection=False, allowed_languages=frozenset({'en'})):
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
        :param language_detection: Detect language of docs.
        :param allowed_languages: Allowed language for the documents.
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
        self.preprocessing_batch_size = 500
        self.processes = multiprocessing.cpu_count()-2
        self.lda_model = None
        self.lda_output = None
        self.grid_search = None
        self.evaluation_output = None
        self.result_df = None
        self.word_topic_df = None
        self.word_topic_intersection = None
        self.allowed_languages = allowed_languages
        self.language_detection = language_detection

    def enable_pipe_component(self, component):
        if component in self.pipe_disable:
            self.pipe_disable.remove(component)
            #todo: add info if not in list from beginning or if successfully enable

    def disable_pipe_component(self, component):
        if component not in self.pipe_disable:
            self.pipe_disable.append(component)
            # todo: add info if not in list from beginning or if successfully enable

    def preprocess_spacy(self):
        # todo: add language check
        if self.language_detection:
            self.spacy_docs = [doc for doc in tqdm(self.nlp.pipe(self.input_docs, disable=self.pipe_disable,
                                                                 n_process=self.processes,
                                                                 batch_size=self.preprocessing_batch_size),
                                                   desc="Preprocessing text with spacy: ")
                               if detect(doc.text) in self.allowed_languages]
        else:
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
        self.result_df = pd.DataFrame(self.evaluation_output)
        if self.document_ids is not None and not self.language_detection:
            self.result_df.index = self.document_ids
        elif self.document_ids is not None and self.language_detection:
            raise Warning("Using document ids and language detection together is not implemented (yet).")
        dominant_topic = np.argmax(self.result_df.values, axis=1)
        self.result_df['dominant_topic'] = dominant_topic

    def plot_document_topic_distribution(self):
        #todo: log normalize
        counter = Counter(self.result_df.dominant_topic)
        topic_dict = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
        sns.barplot(x=list(topic_dict.values()), y=list(topic_dict.keys()), order=list(topic_dict.keys()), orient='h')
        plt.show()

    def evaluate_model(self, no_words=30):
        keywords = np.array(self.vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in self.lda_model.components_:
            top_keyword_locations = (-topic_weights).argsort()[:no_words]
            topic_keywords.append(keywords.take(top_keyword_locations))
        self.word_topic_df = pd.DataFrame(topic_keywords, columns=[f"word_{x}" for x in range(no_words)])

    def evaluate_pyldavis(self):
        panel = pyLDAvis.sklearn.prepare(self.lda_model, self.bag_of_words, self.vectorizer)
        pyLDAvis.show(panel)

    def get_word_topic_intersection(self, no_words=30):
        if not isinstance(self.word_topic_df, pd.DataFrame):
            self.evaluate_model(no_words=no_words)
        elif isinstance(self.word_topic_df, pd.DataFrame) and self.word_topic_df.shape[1] != no_words:
            self.evaluate_model(no_words=no_words)
        intersection_list = []
        for x in range(10):
            temp_list = []
            for y in range(10):
                if x != y:
                    temp_list.append(len(set(self.word_topic_df[self.word_topic_df.index == x].values[0]).intersection(
                        self.word_topic_df[self.word_topic_df.index == y].values[0]))/no_words)
                else:
                    temp_list.append(1)
            intersection_list.append(temp_list)
        self.word_topic_intersection = pd.DataFrame(intersection_list)

    def get_topic_coherence_scores(self):
        # todo: sum of distance between words in topic derived from word embedding
        # todo: sum of sum of distances divided by no topics
        pass

