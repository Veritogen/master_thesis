from extractor import extractor
import pandas as pd
from graph_pipeline.main import *
import multiprocessing as mp
from tqdm.auto import tqdm
import logging
from langdetect import detect
from lxml import html


def lang_detect_wrapper(thread_id, text):
    try:
        return thread_id, detect(text)
    except:
        logging.exception(f"Couldn't detect language for thread no: {thread_id}")
        return thread_id, None


e = extractor.Extractor()
e.load(mode='pol_set', file_name='pol_062016-112019_labeled.ndjson', out_path="pol_extracted")
#e.load(mode='pol_set', file_name='pol_dataset/pol_shuffle_100k.ndjson', out_path="pol_extracted_sample")

logging.basicConfig(filename=f'{e.out_path}/extraction_log.log', filemode='a',
                            format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
e.extract(save_com=False, save_full_text=True, batch_size=10000000)
e.extract_text(no_chunks=4)

text_dict = {thread_id: "" for thread_id in e.stat_df.thread_id}
for iter_tup in e.post_df.itertuples():
    if isinstance(iter_tup.full_string, str):
        text_dict[iter_tup.thread_id] = text_dict[iter_tup.thread_id] + iter_tup.full_string

no_processes = 24
pool = mp.Pool(processes=no_processes)
results = [pool.apply_async(lang_detect_wrapper, (item[0], item[1],))
           for item in tqdm(text_dict.items(),
                            desc="Adding language detection tasks to multiprocessing pool")]
results = [result.get() for result in tqdm(results, desc="Retrieving language detection "
                                                         "results from multiprocessing pool")]
lang_dict = {result[0]: result[1] for result in tqdm(results, desc="Creating dictionary of features")}
languages = [lang_dict[thread_id] for thread_id in e.stat_df.thread_id]
e.stat_df['language'] = languages
text_df = pd.DataFrame([text_dict.keys(), text_dict.values()]).transpose()
text_df.columns = ['thread_id', 'full_text']
text_df.to_pickle(f"{e.out_path}text_df")
e.create_gexfs()

path_list = []
for index, row in e.stat_df.iterrows():
    path_list.append(f"{e.out_path}/gexfs/{row.board}/{row.thread_id}.gexf")
graph_parameters = return_from_list(path_list, no_processes=no_processes)
graph_features_all = pd.DataFrame(graph_parameters).transpose()
graph_features_all['board'] = e.stat_df.set_index('thread_id').board
graph_features_all.to_pickle(f"{e.out_path}/graph_features_all")

