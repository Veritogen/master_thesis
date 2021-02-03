from graph_pipeline.main import *
import pandas as pd


def extract_graphs(no_processes=25):
    path_list = []
    for index, row in stat_df.iterrows():
        path_list.append(f"{out_path}/gexfs/{row.board}/{row.thread_id}.gexf")
    graph_parameters = return_from_list(path_list, no_processes=no_processes)
    graph_features_all = pd.DataFrame(graph_parameters).transpose()
    graph_features_all['board'] = stat_df.set_index('thread_id').board
    for column in graph_features_all.columns:
        try:
            graph_features_all[column] = graph_features_all[column].astype(float)
        except:
            print(f"Couldn't convert {column} to float ([{out_path})")
    graph_features_all.to_pickle(f"{out_path}/graph_features_all")


for out_path in ["pol_extracted", "b_collection_extracted", "archive_collection_extracted"]:
    stat_df = pd.read_pickle(f"{out_path}/stat_df")
    extract_graphs(no_processes=20)

