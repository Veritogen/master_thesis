from extractor import extractor
import numpy as np
import pandas as pd
from graph_pipeline.main import *
from nlpipe import NlPipe
import pickle

e = extractor.Extractor()
e.load(mode='pol_set', file_name='pol_dataset/pol_shuffle_20.ndjson', out_path="pol_extracted_sample")
e.extract(save_com=False, save_full_text=True, batch_size=10000000)
e.extract_text()
e.detect_lang()
e.create_gexfs()

"""path_list = []
filter_array = np.logical_and((np.array(e.stat_df.replies) >= 150), (np.array(e.stat_df.is_acyclic) == True))
for index, row in e.stat_df.iterrows():
    path_list.append(f"{e.out_path}/gexfs/{row.board}/{row.thread_id}.gexf")
graph_parameters = return_from_list(path_list)
graph_features_all = pd.DataFrame(graph_parameters).transpose()
graph_features_all['board'] = e.stat_df.set_index('thread_id').board
graph_features_all = graph_features_all.drop(columns='max_degree')
graph_features_all.to_pickle(f"{e.out_path}/graph_features_all")"""

thread_ids = e.stat_df.thread_id.to_list()
texts = [e.get_document_text(thread_id=thread_id) for thread_id in thread_ids]
nlp = NlPipe.NlPipe(texts, document_ids=thread_ids)
nlp.create_bag_of_words(max_df=0.3)
topic_numbers = [i for i in range(5, 305,5)]
#no_topics, lda_model = nlp.search_best_model(topic_list=topic_numbers, alphas=[0.1,0.2,0.3,0.4,0.5])
nlp.search_best_model(topic_list=topic_numbers, alphas=[0.1])
#nlp.create_document_topic_df(model=lda_model, no_topics=no_topics)
try:
    with open(f"{path}coherence_results_max_df-03.pkl", "wb") as f:
        pickle.dump(nlp.coherence_dict, f)
except:
    pass
#nlp.result_df.to_pickle(f"{path}lda_topic_result.pkl")
#topic_result_df = nlp.result_df
#nlp = None
#topic_result_df = pd.read_pickle(f'{path}lda_topic_result.pkl')
#graph_features = return_all(f"{path}gexfs/275-325/")
#graph_features = pd.DataFrame(graph_features)
#graph_features = graph_features.transpose()
#graph_features = graph_features.join(topic_result_df['dominant_topic'])
#graph_features.to_pickle(f"{path}features_with_topic.pkl")
