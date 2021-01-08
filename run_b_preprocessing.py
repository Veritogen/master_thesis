import pandas as pd
from nlpipe import NlPipe
import numpy as np
import os
import pickle
from tqdm.auto import tqdm

path = "b_collection_extracted/"
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

nlp = NlPipe.NlPipe(texts,path=path, document_ids=thread_ids, no_processes=10)
nlp.preprocess()
