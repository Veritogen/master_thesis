from extractor import extractor
import pandas as pd
from graph_pipeline.main import *
import os


def extract_graphs():
    path_list = []
    for index, row in e.stat_df.iterrows():
        path_list.append(f"{e.out_path}/gexfs/{row.board}/{row.thread_id}.gexf")
    graph_parameters = return_from_list(path_list, no_processes=24)
    graph_features_all = pd.DataFrame(graph_parameters).transpose()
    graph_features_all['board'] = e.stat_df.set_index('thread_id').board
    graph_features_all.to_pickle(f"{e.out_path}/graph_features_all")


e = extractor.Extractor()
e.load(mode='legacy', in_path='/home/hengelhardt/masterarbeit/collection_archives/archive_collection',
       out_path="archive_collection_extracted")
e.extract(save_com=False, save_full_text=True, batch_size=10000000)
#e.detect_lang() not used because data is already extracted. The rest should be loaded differently anyway.
e.create_gexfs()

if not os.path.exists(f"{e.out_path}/graph_features_all"):
    extract_graphs()
else:
    try:
        graph_features = pd.read_pickle(f"{e.out_path}/graph_features_all")
    except:
        extract_graphs()
