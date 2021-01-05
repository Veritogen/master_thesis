from extractor import extractor
import numpy as np
import pandas as pd
from graph_pipeline.main import *


e = extractor.Extractor()
e.load(mode='legacy', in_path='/home/hengelhardt/masterarbeit/collection_archives/archive_collection', out_path="archive_collection_extracted")
e.extract(save_com=False, save_full_text=True, batch_size=10000000)
e.extract_text()
e.create_gexfs()

path_list = []
filter_array = np.logical_and((np.array(e.stat_df.replies) >= 150), (np.array(e.stat_df.is_acyclic) == True))
for index, row in e.stat_df.iterrows():
    path_list.append(f"{e.out_path}/gexfs/{row.board}/{row.thread_id}.gexf")
graph_parameters = return_from_list(path_list)
graph_features_all = pd.DataFrame(graph_parameters).transpose()
graph_features_all['board'] = e.stat_df.set_index('thread_id').board
graph_features_all = graph_features_all.drop(columns='max_degree')
graph_features_all.to_pickle(f"{e.out_path}/graph_features_all")