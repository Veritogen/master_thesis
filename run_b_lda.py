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
                    level=logging.INFO)

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

nlp = NlPipe.NlPipe(texts, path=path, document_ids=thread_ids, no_processes=11)
filter_array = np.logical_and(stat_df.thread_id.isin(text_df.sample(frac=0.5, weights=stat_df.replies).thread_id),
                              stat_df.replies > 10)
filter_array = np.logical_and(filter_array, stat_df.language == 'en')
print(f"{len(filter_array)} is limiting to {sum(filter_array)}")
nlp.preprocess(load_existing=True, filter_loaded=filter_array)
nlp.create_bag_of_words(filter_extremes=False, min_df=None, max_df=None)
with threadpool_limits(limits=1, user_api='blas'):
    for max_df in tqdm([0.5, 0.4, 0.3, 0.2, 0.1], desc="max df"):
        for min_df in tqdm([10, 25]):
            nlp.filter_extremes(min_df=min_df, max_df=max_df, keep_n=nlp.keep_n, keep_tokens=nlp.keep_tokens)
            nlp.filter_extremes(min_df=25, max_df=max_df, keep_n=nlp.keep_n, keep_tokens=nlp.keep_tokens)
            nlp.create_bag_of_words_matrix()
            nlp.search_best_model(topic_list=[25, 50, 75, 100], passes=2,
                                  alphas=['asymmetric', 0.01, 0.1, 0.3], etas=['auto', 0.01, 0.1, 0.3, 0.5],
                                  chunksize=1000, coherence_suffix=1)
