import spacy
import re
from tqdm.auto import tqdm
import gensim.corpora as corpora
from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel, TfidfModel
import pyLDAvis
import pyLDAvis.gensim
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
from langdetect import detect
import psutil
import os
from itertools import compress
import pickle
import time


class NlPipe:
    def __init__(self, list_of_docs, path, document_ids=None, language_model="en_core_web_lg", tagger=False,
                 parser=False, ner=False, categorization=False, remove_stopwords=True, remove_punctuation=True,
                 set_lower=True, remove_num=True, expand_stopwords=True, language_detection=False,
                 allowed_languages=frozenset({'en'}), no_processes=None):
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
        self.path = path
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
        self.document_ids = np.array(document_ids)
        self.use_gpu = spacy.prefer_gpu()
        self.nlp = spacy.load(language_model)
        if expand_stopwords:
            stops = [stop for stop in self.nlp.Defaults.stop_words]
            for stop in stops:
                self.nlp.Defaults.stop_words.add(re.sub(r"[\W]", "", stop))
        self.spacy_docs = None
        self.preprocessed_docs = None
        self.bag_of_words = None
        self.preprocessing_batch_size = 50000
        if no_processes is None:
            self.processes = psutil.cpu_count(logical=False) - 1
        else:
            self.processes = no_processes
        self.lda_model = None
        self.result_df = None
        self.word_topic_df = None
        self.allowed_languages = allowed_languages
        self.language_detection = language_detection
        self.id2word = None
        if os.path.exists(f"{self.path}coherence_results"):
            print("coherence results found")
            with open(f"{self.path}coherence_results", "rb") as f:
                self.coherence_dict = pickle.load(f)
        else:
            self.coherence_dict = {}
        self.max_df = None
        self.min_df = None
        self.use_phrases = None
        self.filter_extremes_value = None
        self.keep_n = None
        self.keep_tokens = None

    def enable_pipe_component(self, component):
        """
        Method to enable components of the spacy pipeline after initialization of the class.
        :param component: Component to enable (see https://spacy.io/usage/processing-pipelines/ for available
        components).
        """
        if component in self.pipe_disable:
            self.pipe_disable.remove(component)

    def disable_pipe_component(self, component):
        """
        Method to disable components of the spacy pipeline after initialization of the class.
        :param component: Component to disable (see https://spacy.io/usage/processing-pipelines/ for available
        components).
        """
        if component not in self.pipe_disable:
            self.pipe_disable.append(component)

    def preprocess_spacy(self, load_existing=True, save_data=True, filter_loaded=None):
        """
        Method to preprocess the documents using spacy with the enabled pipeline components.
        """
        if os.path.exists(f"{self.path}text_df_preprocessed_spacy") and load_existing:
            preprocessed_df = pd.read_pickle(f"{self.path}text_df_preprocessed_spacy")
            if filter_loaded is None:
                self.spacy_docs = preprocessed_df['preprocessed_text'].to_list()
            else:
                self.spacy_docs = preprocessed_df['preprocessed_text'].loc[filter_loaded].to_list()
        else:
            if self.language_detection:
                self.spacy_docs = [doc for doc in tqdm(self.nlp.pipe(self.input_docs, disable=self.pipe_disable,
                                                                     n_process=self.processes,
                                                                     batch_size=self.preprocessing_batch_size),
                                                       desc="Preprocessing text with spacy: ")
                                   if detect(doc.text) in self.allowed_languages]
            else:
                self.spacy_docs = []
                for doc in tqdm(self.nlp.pipe(self.input_docs, disable=self.pipe_disable, n_process=self.processes,
                                              batch_size=self.preprocessing_batch_size), desc="Preprocessing spacy"):
                    self.spacy_docs.append(doc)
            if save_data:
                temp_df = pd.DataFrame([self.document_ids, self.spacy_docs]).transpose()
                temp_df.columns = ['thread_id', 'preprocessed_text']
                temp_df.to_pickle(f"{self.path}text_df_preprocessed_spacy")

    def preprocess(self, load_existing=True, filter_loaded=None):
        """
        Remove stop words, numbers and punctation as well as lower case all of the tokens, depending on the settings
        passed to the class during initialization.
        """
        if os.path.exists(f"{self.path}/text_df_preprocessed") and load_existing:
            print("Found preprocessed data. Loading")
            preprocessed_df = pd.read_pickle(f"{self.path}/text_df_preprocessed")
            if filter_loaded is None:
                self.preprocessed_docs = preprocessed_df['preprocessed_text'].to_list()
                print('Preprocessed data loaded.')
            else:
                self.preprocessed_docs = preprocessed_df['preprocessed_text'].loc[filter_loaded].to_list()
                if isinstance(self.document_ids, np.ndarray):
                    self.document_ids = self.document_ids[filter_loaded]
            print('Preprocessed data loaded.')
        else:
            self.preprocessed_docs = []
            if not self.spacy_docs:
                self.preprocess_spacy()
            for spacy_doc in tqdm(self.spacy_docs, desc="Removing stop words/punctuation/numeric chars: "):
                doc = []
                for token in spacy_doc:
                    # todo: check if useful condition
                    if not self.remove_stop_words and token.is_stop:
                        word = token.text
                    elif token.is_stop:
                        continue
                    else:
                        word = token.lemma_
                    if self.set_lower:
                        word = word.lower()
                    if self.remove_num:
                        word = re.sub(r"[\d]", "", word)
                    if self.remove_punctuation:
                        word = re.sub(r"[\W]", "", word)
                    if len(word) >= 2 and word != "wbr":
                        doc.append(word)
                self.preprocessed_docs.append(doc)
            temp_df = pd.DataFrame([self.document_ids, self.preprocessed_docs]).\
                transpose()
            temp_df.columns = ['thread_id', 'preprocessed_text']
            temp_df.to_pickle(f"{self.path}/text_df_preprocessed")

    def create_bag_of_words(self, filter_extremes=True, min_df=5, max_df=0.5, keep_n=100000, keep_tokens=None,
                            use_phrases=None, bigram_min_count=1000, bigram_threshold=100, trigram_threshold=100,
                            load_existing=True):
        """
        :param filter_extremes: En-/Disable filtering of tokens that occur too frequent/not frequent enough
        (https://radimrehurek.com/gensim/corpora/dictionary.html)
        :param min_df: Keep only tokens that appear in at least n documents (see link above)
        :param max_df: Keep only tokens that appear in less than the fraction of documents (see link above)
        :param keep_n: Keep only n most frequent tokens (see link above)
        :param keep_tokens: Iterable of tokens not to be remove (see link above)
        :param use_phrases: Set to bigram or trigram if the use of Gensmin Phrases
        (https://radimrehurek.com/gensim/models/phrases.html) is wanted. Will create bigrams/trigrams of frequently
        co-occuring words (e.g. "new", "york" => "new_yor)k").
        :param bigram_min_count: Minimum occurrence of bigrams to be considered by Gensmin Phrases.
        :param bigram_threshold: Threshold for Gensim Phrases bigram settings.
        :param trigram_threshold: Threshold for Gensim Phrases trigram settings.
        """
        if use_phrases not in {None, "bigram", "trigram"}:
            raise Exception("Please use valid option (None, 'bigram' or 'trigram) to make use of this function.")
        #todo: check logic
        else:
            if use_phrases == "bigram" and not isinstance(bigram_threshold, int) and not isinstance(bigram_min_count,
                                                                                                    int):
                raise Exception("Thresholds or minimum count for bigrams/trigrams not integer. Please provide "
                                "threshold and minimum count for bigrams (and trigrams) as integer.")
            elif use_phrases == "trigram" and not isinstance(bigram_threshold, int) \
                    or not isinstance(trigram_threshold, int) or not isinstance(bigram_min_count, int):
                raise Exception("Thresholds or minimum count for bigrams/trigrams not integer. Please provide "
                                "threshold and minimum count for bigrams (and trigrams) as integer.")

        if not self.preprocessed_docs:
            self.preprocess()
        if os.path.exists(f"{self.path}/gensim_dict_{filter_extremes}_{min_df}_{max_df}_{use_phrases}") \
                and load_existing:
            self.load_dict(path=f"{self.path}/gensim_dict_{filter_extremes}_{min_df}_{max_df}_{use_phrases}")
            self.filter_extremes_value = filter_extremes
            self.min_df = min_df
            self.max_df = max_df
            self.use_phrases = use_phrases
        else:
            #todo: add auto check for existing dictionary here.
            if use_phrases == "bigram" or use_phrases == "trigram":
                self.create_bigrams(bigram_min_count=bigram_min_count, bigram_threshold=bigram_threshold)
            if use_phrases == "trigram":
                self.create_bigrams(bigram_min_count=bigram_min_count, bigram_threshold=bigram_threshold)
                self.create_trigrams(trigram_threshold=trigram_threshold)
            self.create_dictionary(filter_extremes=filter_extremes, min_df=min_df, max_df=max_df, keep_n=keep_n,
                                   keep_tokens=keep_tokens, use_phrases=use_phrases)
            self.create_bag_of_words_matrix()

    def create_bigrams(self, bigram_min_count, bigram_threshold):
        self.bigram_phrases = Phrases(self.preprocessed_docs, min_count=bigram_min_count,
                                      threshold=bigram_threshold)
        self.bigram_phraser = Phraser(self.bigram_phrases)
        self.preprocessed_docs = [self.bigram_phraser[doc] for doc in
                                  tqdm(self.preprocessed_docs, desc="Extracting bigrams")]

    def create_trigrams(self, trigram_threshold):
        trigram_phrases = Phrases(self.bigram_phrases[self.preprocessed_docs], threshold=trigram_threshold)
        trigram_phraser = Phraser(trigram_phrases)
        self.preprocessed_docs = [trigram_phraser[self.bigram_phraser[doc]]
                                  for doc in tqdm(self.preprocessed_docs, desc="Extracting trigrams")]

    def create_bag_of_words_matrix(self):
        self.bag_of_words = [self.id2word.doc2bow(doc)
                             for doc in tqdm(self.preprocessed_docs, desc='Creating bag of words')]

    def create_dictionary(self, filter_extremes, min_df, max_df, keep_n, keep_tokens, use_phrases):
        print('Creating dictionary.')
        self.id2word = corpora.Dictionary(self.preprocessed_docs)
        # todo: add autosave of dictionary here
        self.max_df = max_df
        self.min_df = min_df
        self.use_phrases = use_phrases
        self.filter_extremes_value = filter_extremes
        self.keep_n = keep_n
        self.keep_tokens = keep_tokens
        if filter_extremes:
            self.filter_extremes(min_df = self.min_df, max_df=self.max_df, keep_n=self.keep_n,
                                 keep_tokens=self.keep_tokens)
        self.save_dict(path=f"{self.path}/gensim_dict_{filter_extremes}_{min_df}_{max_df}_{use_phrases}")

    def filter_extremes(self, min_df, max_df, keep_n, keep_tokens=[]):
        self.filter_extremes_value = True
        self.max_df = max_df
        self.min_df = min_df
        self.keep_n = keep_n
        self.keep_tokens = keep_tokens
        self.id2word.filter_extremes(no_below=self.min_df, no_above=self.max_df, keep_n=keep_n,
                                     keep_tokens=keep_tokens)

    def create_tfidf(self):
        pass

    def create_lda_model(self, no_topics=10, random_state=42, passes=5, alpha='auto', eta='auto', workers=None,
                         chunksize=2000):
        """
        :param no_topics: Number of topics that are to be explored by lda model
        :param random_state: Random state for reproducible results (default 42, gensim default is None)
        :param passes: Number of times the whole corpus is processed.
        :param alpha: set topic-document distribution prior alpha to "symmetric" or "asymmetric"
        (gensim default is "symmetric")
        :param eta: Word-topic distribution prior eta (beta)
        :param workers: number of workers to use. Defaulting to one as there seems to be a bug in gensim. 1 already
        uses all available cores. Higher number of workers results in a load bigger than the number of cores.
        :param chunksize: chunsize parameter of gensim
        """
        if workers is None:
            workers = self.processes
        if self.bag_of_words is None:
            self.create_bag_of_words()
        self.lda_model = LdaMulticore(corpus=self.bag_of_words, id2word=self.id2word, num_topics=no_topics, eta=eta,
                                      workers=workers, random_state=random_state, alpha=alpha, passes=passes,
                                      chunksize=chunksize)

    def calculate_coherence(self, model=None, coherence_score='c_v', workers=None):
        """
        Method to calculate the coherence score of a given lda model. The model can either be provided or will be taken
        from the class.
        :param model: Model to use instead of the model saved within the class.
        :param coherence_score: Coherence score to calculate
        :return: Return coherence model, which also contains the coherence score of a model.
        """
        if workers is None:
            workers = self.processes
        if model is None:
            model = self.lda_model
        else:
            model = model
        if coherence_score != 'u_mass':
            coherence_model = CoherenceModel(model=model, texts=self.preprocessed_docs, dictionary=self.id2word,
                                             coherence=coherence_score, processes=workers)
        else:
            coherence_model = CoherenceModel(model=model, corpus=self.bag_of_words, dictionary=self.id2word,
                                             coherence=coherence_score, processes=workers)
        return coherence_model

    def search_best_model(self, topic_list=frozenset({2, 3, 4, 5, 10, 15, 20, 25}), alphas=[0.9, 0.5, 0.1],
                          etas=['auto', 0.9, 0.5, 0.1], save_best_model=True, save_models=False,
                          return_best_model=False, passes = 1, coherence_scores=['c_v'], chunksize=2000, workers=None,
                          coherence_suffix=None):
        #todo: save best model within class.
        """
        Method to search for the best lda model for a given number of topics. The best model will be determined by its
        coherence score.
        :param topic_list: Iterable of integers of topics to test the coherence score for.
        :param alphas: Iterable of floats between 0 and 1 for determining the dirichlet prior of the lda model.
        :param save_best_model: Set to true if the best model has to be saved within the class.
        :param save_models: If set to false (default) only the coherence score for each combination of numbers of topics
        and alphas will be saved. If set to true, the lda model, the coherence score and the coherence model will be
        saved.
        :param return_best_model: If true, the method will return the best found model and the number of topics of this
        model.
        :return: Number of topics for the best result and the model with the best result of the coherence score
        """
        if workers is None:
            workers = self.processes
        if return_best_model and not save_best_model:
            raise Exception("To return the best model, the parameter save_best_model has to be set to True.")
        if self.coherence_dict and save_best_model:
            try:
                best_score = self.coherence_dict['best_score']
            except:
                best_score = 0
        else:
            best_score = 0
        for no_topics in tqdm(topic_list, desc="Calculating topic coherences: "):
            for alpha in tqdm(alphas, desc='Alphas'):
                for eta in tqdm(etas, desc='Etas'):
                    coherence_key = f"no={no_topics}-a={alpha}-e={eta}-filter={self.filter_extremes_value}" \
                                    f"-min_df={self.min_df}-max_df={self.max_df}-phrases={self.use_phrases}" \
                                    f"-k_n={self.keep_n}-k_t={self.keep_tokens}"
                    if coherence_key in self.coherence_dict.keys():
                        print("coherence value found, skipping")
                        continue
                    else:
                        self.create_lda_model(no_topics=no_topics, alpha=alpha, eta=eta, passes=passes,
                                              chunksize=chunksize, workers=workers)
                        self.coherence_dict[coherence_key] = {}
                        if save_models:
                            self.coherence_dict[coherence_key]["lda_model"] = self.lda_model
                        for coherence_score in coherence_scores:
                            coherence_model = self.calculate_coherence(coherence_score=coherence_score, workers=workers)
                            coherence_result = coherence_model.get_coherence()
                            if save_models:
                                self.coherence_dict[coherence_key]["coherence_model"] = coherence_model
                            self.coherence_dict[coherence_key][coherence_score] = coherence_result
                            if save_best_model and coherence_result > best_score:
                                self.coherence_dict["best_score"] = coherence_result
                                self.coherence_dict["best_model"] = self.lda_model
                                self.coherence_dict["best_topic_no"] = no_topics
                                self.coherence_dict["best_alpha"] = alpha
                                self.coherence_dict["best_eta"] = eta
                            if coherence_result > best_score:
                                best_score = coherence_result
                        if coherence_suffix is None:
                            with open(f"{self.path}coherence_results", "wb") as f:
                                pickle.dump(self.coherence_dict, f)
                        else:
                            with open(f"{self.path}coherence_results_{coherence_suffix}", "wb") as f:
                                pickle.dump(self.coherence_dict, f)
        if return_best_model:
            #returns number of topics and the lda_model
            return self.coherence_dict["best_topic_no"], self.coherence_dict["best_model"]

    def create_document_topic_df(self, model=None, no_topics=10):
        """
        Creates a dataframe containing the the result of the LDA model for each document. Will set the topic with the
        highest share within the document as the dominant topic.
        :param model: LDA model to use for the calculation of the topic distribution of each document.
        :param no_topics: Number of topics in case no LDA model is provided.
        """
        if model is None:
            self.create_lda_model(no_topics=no_topics)
        else:
            self.lda_model = model
        topic_result_list = []
        for doc in self.lda_model.get_document_topics(bow=self.bag_of_words):
            temp_dict = {}
            for topic, probability in doc:
                temp_dict[topic] = probability
            topic_result_list.append(temp_dict)
        self.result_df = pd.DataFrame(data=topic_result_list, columns=range(no_topics))
        self.result_df = self.result_df.fillna(0)
        if self.document_ids is not None and not self.language_detection:
            self.result_df.index = self.document_ids
        elif self.document_ids is not None and self.language_detection:
            raise Warning("Using document ids and language detection together is not implemented (yet).")
        dominant_topic = np.argmax(self.result_df.values, axis=1)
        self.result_df['dominant_topic'] = dominant_topic

    def plot_document_topic_distribution(self):
        #todo: log normalize
        if self.result_df is None:
            raise Exception("Please create the topic distribution dataframe using the 'create_document_topic_df' "
                            "method")
        counter = Counter(self.result_df.dominant_topic)
        topic_dict = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
        sns.barplot(x=list(topic_dict.values()), y=list(topic_dict.keys()), order=list(topic_dict.keys()), orient='h')
        plt.show()

    def evaluate_model(self, no_words=30):
        #todo: update 4 gensim
        keywords = np.array(self.vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in self.lda_model.components_:
            top_keyword_locations = (-topic_weights).argsort()[:no_words]
            topic_keywords.append(keywords.take(top_keyword_locations))
        self.word_topic_df = pd.DataFrame(topic_keywords, columns=[f"word_{x}" for x in range(no_words)])

    def evaluate_pyldavis(self, model=None):
        """
        Method for a visual evaluation of the LDA topic model using pyldavis.
        :param model: LDA model that is to be evaluated. If 'None', it will use the last model that has been saved
        within the class.
        :return:
        """
        try:
            is_jupyter = os.environ['_'].split("/")[-1] == "jupyter-notebook"
            if is_jupyter:
                pyLDAvis.enable_notebook()
        except KeyError:
            is_jupyter = False
        if model is None:
            if self.lda_model is None:
                raise Exception("Please create a LDA model for evaluation before running this method.")
            model = self.lda_model
        panel = pyLDAvis.gensim.prepare(model, self.bag_of_words, self.id2word)
        if is_jupyter:
            pyLDAvis.display(panel)
        else:
            pyLDAvis.show(panel)

    def print_bow(self, doc_positions):
        print([[(self.id2word[token_id], freq) for token_id, freq in doc]
               for doc in compress(self.bag_of_words, doc_positions)])

    def save_model(self, path):
        self.lda_model.save(path)

    def load_model(self, path):
        self.lda_model = LdaMulticore.load(path)

    def save_dict(self, path):
        self.id2word.save(path)
        print("dict saved")

    def load_dict(self, path):
        self.id2word = corpora.Dictionary.load(path)