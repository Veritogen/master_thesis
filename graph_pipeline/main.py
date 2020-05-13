import glob
from graph_pipeline import GraphFeatures as gf
import multiprocessing as mp
import networkx as nx
from tqdm import tqdm
import traceback


def return_one(path):
    graph = nx.read_gexf(path)
    try:
        graph_features = gf.GraphFeatures(graph).return_features()
    except:
        print(path, traceback.format_exc())
    thread_id = int(path.split("/")[-1].strip(".gexf"))
    return thread_id, graph_features


def return_from_list(id_list, path):
    path_list = [f"{path}{thread_id}.gexf" for thread_id in id_list]
    pool = mp.Pool(processes=11)
    results = [pool.apply_async(return_one, (path_to_file,)) for path_to_file in tqdm(path_list)]
    results = [result.get() for result in tqdm(results)]
    feature_dict = {result[0]: result[1] for result in tqdm(results)}
    return feature_dict


def return_all(path):
    path_list = glob.glob(f"{path}*.gexf")
    pool = mp.Pool(processes=11)
    results = [pool.apply_async(return_one, (path_to_file,)) for path_to_file in tqdm(path_list)]
    results = [result.get() for result in tqdm(results)]
    feature_dict = {result[0]: result[1] for result in tqdm(results)}
    return feature_dict

