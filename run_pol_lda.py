import pandas as pd
from nlpipe import NlPipe
import numpy as np
import os
import pickle
from tqdm.auto import tqdm

path = "pol_extracted/"
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

nlp = NlPipe.NlPipe(texts, path=path, document_ids=thread_ids, no_processes=36)
filter_array = np.logical_and(stat_df.thread_id.isin(text_df.sample(frac=0.3, weights=stat_df.replies).thread_id),
                              stat_df.replies > 10)
for max_df in tqdm([0.3, 0.2, 0.1], desc="max df"):
    nlp.preprocess(load_existing=True, filter_loaded=filter_array)
    nlp.create_bag_of_words(max_df=max_df)
    nlp.search_best_model(topic_list=[50, 100, 150, 200, 250, 300])

try:
    with open(f"{path}coherence_results", "wb") as f:
        pickle.dump(nlp.coherence_dict, f)
except:
    pass
