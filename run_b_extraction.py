from extractor import extractor
import pandas as pd
from graph_pipeline.main import *
import os
from langdetect import detect
import logging
import multiprocessing as mp


def extract_graphs():
    path_list = []
    for index, row in e.stat_df.iterrows():
        path_list.append(f"{e.out_path}/gexfs/{row.board}/{row.thread_id}.gexf")
    graph_parameters = return_from_list(path_list, no_processes=no_processes)
    graph_features_all = pd.DataFrame(graph_parameters).transpose()
    graph_features_all['board'] = e.stat_df.set_index('thread_id').board
    graph_features_all.to_pickle(f"{e.out_path}/graph_features_all")


def lang_detect_wrapper(thread_id, text):
    try:
        return thread_id, detect(text)
    except:
        logging.exception(f"Couldn't detect language for thread no: {thread_id}")
        return thread_id, None


e = extractor.Extractor()
e.load(mode='legacy', in_path='/home/hengelhardt/masterarbeit/b_collection', out_path="b_collection_extracted")
logging.basicConfig(filename=f'{e.out_path}/extraction_log.log', filemode='a',
                            format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
# extract data from raw jsons
e.extract(save_com=False, save_full_text=True, batch_size=10000000)

# create dict with text of all threads for detection of languages
text_dict = {thread_id: "" for thread_id in e.stat_df.thread_id}
for iter_tup in tqdm(e.post_df.itertuples(), desc="merging posts to a doc for each thread"):
    if isinstance(iter_tup.full_string, str):
        text_dict[iter_tup.thread_id] = text_dict[iter_tup.thread_id] + iter_tup.full_string
no_processes = 40
pool = mp.Pool(processes=no_processes)
results = [pool.apply_async(lang_detect_wrapper, (item[0], item[1],))
           for item in tqdm(text_dict.items(),
                            desc="Adding language detection tasks to multiprocessing pool")]
results = [result.get() for result in tqdm(results, desc="Retrieving language detection "
                                                         "results from multiprocessing pool")]
lang_dict = {result[0]: result[1] for result in tqdm(results, desc="Creating dictionary of features")}
languages = [lang_dict[thread_id] for thread_id in e.stat_df.thread_id]
e.stat_df['language'] = languages

# extract gexf files for calculation of graph features, will also save the previously detected languages
e.create_gexfs()

# extract graph featues
if not os.path.exists(f"{e.out_path}/graph_features_all"):
    extract_graphs()
else:
    try:
        graph_features = pd.read_pickle(f"{e.out_path}/graph_features_all")
    except:
        extract_graphs()
