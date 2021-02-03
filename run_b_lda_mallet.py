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

nlp = NlPipe.NlPipe(texts, path=path, document_ids=thread_ids, no_processes=10)
filter_array = np.logical_and(stat_df.language == 'en',
                              stat_df.replies > 10)

print(f"{len(filter_array)} is limiting to {sum(filter_array)}")
nlp.preprocess(load_existing=True, filter_loaded=filter_array)
nlp.create_bag_of_words(filter_extremes=True, min_df=(0.001*sum(filter_array)), max_df=0.5)
nlp.search_best_model_mallet(topic_list=[range(5, 50, 5)])
