import glob
from graph_pipeline import GraphFeatures as gf
import multiprocessing as mp
import networkx as nx
from tqdm import tqdm
import traceback

"""
Script to execute the extraction of graph measures for either a single graph or multiple/all graphs using 
multiprocessing to speed up the extraction.
"""


def return_one(path):
    #todo: add log instead of print
    #todo: add Exception/Warning if failed to extract data from a graph.
    """
    Function to extract the features of a given graph (provided with the path to a .gexf-file.)
    :param path: Path of the graph (saved as .gexf file) to extract the graph features from.
    :return:
    """
    graph = nx.read_gexf(path)
    try:
        graph_features = gf.GraphFeatures(graph).return_features()
    except:
        #print(path, traceback.format_exc())
        graph_features = None
    thread_id = int(path.split("/")[-1].strip(".gexf"))
    return thread_id, graph_features


def return_from_list(path_list):
    """
    Function to extract the features of certain graphs, where the id's are provided.
    :param id_list: List of thread ids of the .gexf-files in which the graphs are saved.
    :param path: Path to the directory, in which the .gexf-files are stored.
    :return: Dictionary with the thread ids as keys and the graph features as values.
    """
    path_list = path_list
    pool = mp.Pool(processes=mp.cpu_count())
    results = [pool.apply_async(return_one, (path_to_file,)) for path_to_file in tqdm(path_list)]
    results = [result.get() for result in tqdm(results)]
    feature_dict = {result[0]: result[1] for result in tqdm(results)}
    return feature_dict


def return_all(path):
    """
    Function to return the features of all graphs that are stored within a directory.
    :param path: Path to the directory, in which the .gexf-files are stored
    :return: Dictionary with the thread ids as keys and the graph features as values.
    """
    path_list = glob.glob(f"{path}*.gexf")
    pool = mp.Pool(processes=mp.cpu_count())
    results = [pool.apply_async(return_one, (path_to_file,)) for path_to_file in tqdm(path_list)]
    results = [result.get() for result in tqdm(results)]
    feature_dict = {result[0]: result[1] for result in tqdm(results)}
    return feature_dict

