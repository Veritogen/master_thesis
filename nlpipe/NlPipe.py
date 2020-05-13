# todo: 1. tokenize
# todo: 2. remove stopwords + puctuation
# todo: 3. create bag of words
# todo: 3.1 bag of words
# todo: 3.2 TfIdf
# todo: 3.3. Bi-/trigram model
# todo: create lda model
# todo: get accuracy of model
import spacy
import multiprocessing
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation as lda
from sklearn.model_selection import GridSearchCV

class NlPipe:
    def __init__(self, list_of_docs,language_model="en_core_web_lg", tagger=False, parser=False, ner=False,
                 categorization=False, remove_stopwords=True, remove_punctuation=True, set_lower=True,
                 remove_num=True, expand_stopwords=True):
        '''
        :param list_of_docs: List of strings where every document is one string
        '''
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
        self.nlp = spacy.load(language_model)
        if expand_stopwords:
            stops = [stop for stop in self.nlp.Defaults.stop_words]
            for stop in stops:
                self.nlp.Defaults.stop_words.add(re.sub(r"[\W]", "", stop))
        self.spacy_docs = None
        self.processed_docs = None
        self.bag_of_words = None
        self.tf_idf = None
        self.preprocessing_batch_size = 50
        self.processes = multiprocessing.cpu_count()-2
        self.lda_model = None
        self.lda_output = None
        self.grid_search = None

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
                                                             batch_size=self.preprocessing_batch_size))]

    def preprocess(self):
        self.processed_docs = []
        if not self.spacy_docs:
            self.preprocess_spacy()
        for spacy_doc in tqdm(self.spacy_docs):
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
        vectorizer = CountVectorizer(lowercase=False, ngram_range=n_grams, min_df=min_df,
                                     max_df=max_df)
        self.bag_of_words = vectorizer.fit_transform(joined_docs)

    def create_tf_idf(self, n_grams=(1, 1), min_df=0.01, max_df=0.6):
        self.preprocess_spacy()
        self.preprocess()
        joined_docs = []
        for doc in self.processed_docs:
            joined_docs.append(" ".join(doc))
        vectorizer = TfidfVectorizer(lowercase=False, ngram_range=n_grams, min_df=min_df,
                                     max_df=max_df)
        self.tf_idf = vectorizer.fit_transform(joined_docs)

    def create_lda_model(self, no_topics=10, input_type="bag"):
        self.lda_model = lda(n_jobs=self.processes, n_components=no_topics)
        if input_type == "bag":
            self.create_bag_of_words()
            self.lda_output = self.lda_model.fit_transform(self.bag_of_words)
        else:
            self.create_tf_idf()
            self.lda_output = self.lda_model.fit_transform(self.tf_idf)

    def evaluate_lda(self, n_components=[2, 3, 4, 5, 10, 15, 20, 25], learning_decay=[.5, .7, .9], input_type="bag"):
        lda_model = lda()
        self.grid_search = GridSearchCV(lda_model, {"n_components": n_components, "learning_decay": learning_decay})
        if input_type == "bag":
            if self.bag_of_words is None:
                self.create_bag_of_words()
            self.grid_search.fit(self.bag_of_words)
        else:
            self.grid_search.fit(self.tf_idf)

    def return_best_model_result(self):
        pass