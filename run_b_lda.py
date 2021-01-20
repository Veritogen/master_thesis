import os
import pandas as pd
from nlpipe import NlPipe
import numpy as np
import os
from tqdm.auto import tqdm
import logging
from threadpoolctl import threadpool_limits

path = "b_collection_extracted/"
logging.basicConfig(filename=f"{path}lda.log", format='%(asctime)s : %(levelname)s : %(processName)s : %(message)s',
                    level=logging.DEBUG)

stat_df = pd.read_pickle(f"{path}stat_df")
if os.path.exists(f"{path}text_df"):
    print("text df found. loading.")
    text_df = pd.read_pickle(f"{path}text_df")
    texts = text_df.full_text.to_list()
    thread_ids = text_df.thread_id.to_list()
else:
    thread_ids = stat_df.thread_id.to_list()
    post_df = pd.read_pickle(f"{path}post_df_extracted")
    thread_id_of_posts = np.array(post_df.thread_id, dtype=np.uint32)
    texts = [" ".join(post_df.full_string[thread_id_of_posts == thread_id].tolist()) for thread_id in thread_ids]
    post_df = None
    text_df = pd.DataFrame([thread_ids, texts]).transpose()
    text_df.columns = ['thread_id', 'full_text']
    text_df.to_pickle(f"{path}text_df")

nlp = NlPipe.NlPipe(texts, path=path, document_ids=thread_ids, no_processes=20)
#filter_array = np.logical_and(stat_df.thread_id.isin(text_df.sample(frac=0.1, weights=stat_df.replies).thread_id),
                             # stat_df.replies > 10)
#filter_array = np.logical_and(filter_array, stat_df.language == 'en')
nlp.preprocess(load_existing=True)
nlp.create_bag_of_words(filter_extremes=False, use_phrases='bigram')
with threadpool_limits(limits=1, user_api='blas'):
    for max_df in tqdm([0.3, 0.2, 0.1], desc="max df"):
        nlp.filter_extremes(min_df=1, max_df=max_df, keep_n=nlp.keep_n, keep_tokens=nlp.keep_tokens)
        nlp.create_bag_of_words_matrix()
        nlp.search_best_model(topic_list=[50, 100, 150, 200], passes=2,
                              alphas=['asymmetric', 0.01, 0.1, 0.3, 0.5, 0.7], etas=['auto', 0.01, 0.1, 0.3],
                              chunksize=2000)
