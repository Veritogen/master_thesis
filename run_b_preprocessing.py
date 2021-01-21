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
    text_df = None
else:
    thread_ids = stat_df.thread_id.to_list()
    post_df = pd.read_pickle(f"{path}post_df_extracted")
    text_dict = {thread_id: "" for thread_id in stat_df.thread_id}
    for iter_tup in tqdm(post_df.itertuples(), desc="merging posts to a doc for each thread"):
        if isinstance(iter_tup.full_string, str):
            text_dict[iter_tup.thread_id] = text_dict[iter_tup.thread_id] + iter_tup.full_string
    post_df = None
    texts = [text_dict[thread_id] for thread_id in thread_ids]
    text_df = pd.DataFrame([thread_ids, texts]).transpose()
    text_df.columns = ['thread_id', 'full_text']
    text_df.to_pickle(f"{path}text_df")
    text_df = None

nlp = NlPipe.NlPipe(texts, path=path, document_ids=thread_ids, no_processes=48)
nlp.preprocess()
