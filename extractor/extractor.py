import os
from collections import defaultdict
import json
from tqdm import tqdm
import pandas as pd
import swifter
from bs4 import BeautifulSoup as bs
import networkx as nx
import warnings
import logging
from langdetect import detect
from collections import namedtuple, defaultdict
from lxml import html
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

#todo: setup logger
#todo: check for proper extraction (see chris project)
#todo: Data Classes (data classes json)


class Extractor:
    """
    Class for the extraction/mining of the data contained in json files, either collected from the API by myself or
    from the dataset of /pol/ thread, published with paper "Raiders of the Lost Kek".
    """
    def __init__(self):
        self.in_path = None
        self.out_path = None
        self.file_name = None
        self.mode = None
        self.file_dict = None
        self.stat_list = None
        self.post_list = None
        self.relevant_stats = None
        self.ignore_keys = None
        self.post_keys = None
        self.stat_df = None
        self.post_df = None
        self.board = None
        self.json_file = None
        self.filter_cyclic = None
        self.loaded = False
        self.save_com = None
        self.counter = 0
        self.post_df_columns = None
        self.ThreadTuple = namedtuple('ThreadTuple', ['board', 'thread_id', 'thread_dict'])
        self.extract_from_post = ['quoted_list']
        self.PostTuple = namedtuple('PostTuple', ['full_string', 'quoted_list', 'own_text', 'quote_string',
                                                  'dead_link_list'])
        self.tag_collection = defaultdict(list)

    def load(self, in_path=None, out_path=None, mode="legacy", file_name=None, debug=False):
        """
        :param in_path: Path of the files to be processed.
        :param out_path: Path where to save the extracted data.
        :param mode: Mode of the extraction, either "legacy" for the normal json files or "pol_set" if data from the
        "Raiders of the Lost Kek" paper is to be extracted.
        :param file_name: Name of the file if set to "pol_set". If not provided, the file name of the extracted data set
        will be used.
        :param debug: Can be used to allow debugging, disabled by default.
        """
        if in_path:
            self.in_path = in_path
        else:
            self.in_path = os.getcwd()
        if out_path is None:
            self.out_path = self.in_path
        else:
            self.out_path = out_path
            os.makedirs(f"{self.out_path}", exist_ok=True)
        self.file_name = file_name
        if mode not in ['pol_set', 'legacy']:
            logging.error("No valid information for mode was provided. Please specify either 'legacy' or 'pol_set', "
                          "depending on the input data.")
            raise Exception("No valid mode for extraction was provided.")
        self.mode = mode
        if self.mode == "pol_set" and self.file_name is None:
            logging.exception("Couldn't load data because no file was specified for pol extraction.")
            raise Exception("File name of dataset not provided.")
        logging.basicConfig(filename=f'{out_path}/extraction_log.log', filemode='a',
                            format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
        if not debug:
            logging.getLogger().setLevel(logging.INFO)
        self.loaded = True
        logging.info("Files loaded.")

    def create_file_dict(self):
        """
        Method to create a dictionary with information on all json files/threads in the input folder while also
        considering the board of the thread.
        """
        self.file_dict = defaultdict(list)
        for item in os.listdir(self.in_path):
            if not os.path.isfile(f"{self.in_path}/{item}"):
                for file in os.listdir(f"{self.in_path}/{item}"):
                    if file.endswith(".json"):
                        self.file_dict[item].append(file)
        logging.debug("File dict created.")

    def extract(self, filter_cyclic=True, complete_extraction=False, save_com=False,
                save_full_text=True, save_quote_text=False, save_dead_links=False, batch_size=10000):
        """
        Method to load the json files and set the according thread id/board within the class.
        :param filter_cyclic: If true, filter only threads where the post network in the thread is acyclic.
        :param complete_extraction: If true, all information provided by the 4chan API will be extracted, else just
        a limited set is used
        """
        if not self.loaded:
            logging.exception("Can't extract. No files loaded yet. Please use 'load' method to load files")
            raise Exception("Can't extract. No files loaded yet. Please use 'load' method to load files")

        self.relevant_stats = ['thread_id', 'board', 'semantic_url', 'time', 'archived_on', 'replies', 'images',
                               'bumplimit', 'imagelimit']
        self.ignore_keys = {'semantic_url', 'archived_on', 'replies', 'images', 'bumplimit',
                            'imagelimit', 'closed', 'archived'}
        if complete_extraction:
            self.post_keys = ['no', 'now', 'name', 'sub', 'com', 'filename', 'ext', 'w', 'h', 'tn_w', 'tn_h', 'tim',
                              'time', 'md5', 'fsize', 'resto', 'trip', 'filedeleted', 'capcode', 'since4pass',
                              'country', 'country_name', 'tail_size', 'troll_country', 'm_img', 'custom_spoiler',
                              'spoiler']
            logging.debug("Extracting all possible information from posts.")
        else:
            self.post_keys = ['no', 'com', 'time', 'resto']
            logging.debug("Extracting limited set of information from posts.")
        self.save_com = save_com
        self.filter_cyclic = filter_cyclic
        self.stat_list = []
        self.post_list = []
        post_columns = self.post_keys[:]
        if save_full_text:
            post_columns.append('full_string')
            self.extract_from_post.append('full_string')
        if save_quote_text:
            post_columns.append('quote_string')
            self.extract_from_post.append('quote_string')
        if save_dead_links:
            post_columns.append('dead_links')
            self.extract_from_post.append('dead_links')
        for column in ['thread_id']:
            post_columns.append(column)
        self.post_df_columns = post_columns
        self.post_df = pd.DataFrame(columns=post_columns)
        if not self.save_com:
            self.post_df = self.post_df.drop(columns='com')
        self.stat_df = pd.DataFrame(columns=self.relevant_stats)
        if self.mode == 'legacy':
            logging.debug("Extracting information from files collected from 4chan API.")
        if self.mode == 'pol_set':
            logging.debug("Extracting information from pol dataset.")
        for thread_tuple in tqdm(self.thread_generator()):
            self.extract_json(thread_tuple)
            if self.counter > batch_size:
                temp_post_df = pd.DataFrame(columns=post_columns, data=self.post_list)
                temp_post_df[self.extract_from_post] = temp_post_df.swifter.apply(lambda x:
                                                                                  self.strip_text(input_text=x['com'],
                                                                                                  post_id=x['no']),
                                                                                  result_type='expand', axis=1)
                if not self.save_com:
                    temp_post_df = temp_post_df.drop(columns='com')
                self.post_df = pd.concat([self.post_df, temp_post_df], ignore_index=True, copy= False)
                self.post_list = []
                temp_stat_df = pd.DataFrame(columns=self.relevant_stats, data=self.stat_list)
                self.stat_df = pd.concat([self.stat_df, temp_stat_df], ignore_index=True, copy= False)
                self.stat_list = []
                self.counter = 0
        temp_post_df = pd.DataFrame(columns=post_columns, data=self.post_list)
        temp_post_df[self.extract_from_post] = temp_post_df.swifter.apply(lambda x: self.strip_text(input_text=x['com'],
                                                                                                    post_id=x['no']),
                                                                          result_type='expand', axis=1)
        if not self.save_com:
            temp_post_df = temp_post_df.drop(columns='com')
        self.post_df = pd.concat([self.post_df, temp_post_df], ignore_index=True, copy= False)
        temp_stat_df = pd.DataFrame(columns=self.relevant_stats, data=self.stat_list)
        self.stat_df = pd.concat([self.stat_df, temp_stat_df], ignore_index=True, copy= False)


    def thread_generator(self):
        if self.mode == 'pol_set':
            self.post_keys.append('extracted_poster_id')
            board = 'pol'
            with open(f"{self.in_path}/pol_062016-112019_labeled.ndjson" if self.file_name is None
                      else f"{self.in_path}/{self.file_name}") as f:
                for line in f:
                    json_file = json.loads(line)
                    thread_id = int(json_file['posts'][0]['no'])
                    yield self.ThreadTuple(board=board, thread_id=thread_id, thread_dict=json_file)
        elif self.mode == 'legacy':
            if not self.file_dict:
                self.create_file_dict()
            for board in self.file_dict.keys():
                for thread_file in self.file_dict[board]:
                    json_file = json.load(open(f"{self.in_path}/{board}/{thread_file}"))
                    thread_id = int(thread_file.split('.')[0])
                    yield self.ThreadTuple(board=board, thread_id=thread_id, thread_dict=json_file)

    def extract_json(self, thread_tuple):
        #todo: add log instead of print
        """
        Method to extract the information that is contained in the json files, loaded by the method "extract". Will
        save information on the statistics per thread and extract data from the text of each post. Will also save all
        information provided per post by the API.
        """
        for post in thread_tuple.thread_dict['posts']:
            if post['no'] == thread_tuple.thread_id:
                stat_temp = {'board': thread_tuple.board, 'thread_id': thread_tuple.thread_id}
                for rel_stat in self.relevant_stats:
                    if rel_stat not in ['board', 'thread_id']:
                        try:
                            stat_temp[rel_stat] = post[rel_stat]
                        except KeyError:
                            logging.debug(f"Missing information for {rel_stat} in thread no {thread_tuple.thread_id}")
                            stat_temp[rel_stat] = 'MISSING'
                self.stat_list.append(stat_temp)
            post_dict = {'thread_id': thread_tuple.thread_id,
                         'board': thread_tuple.board}
            for key in self.post_keys:
                if key in post.keys():
                    post_dict[key] = post[key]
                else:
                    post_dict[key] = None
                if 'com' in post.keys():
                    post_dict['com'] = post['com']
                else:
                    post_dict['com'] = ""
            temp_post_dict = {}
            for key in self.post_df_columns:
                if key not in {'full_string', 'quoted_list', 'own_text', 'quote_string', 'dead_links'}:
                    temp_post_dict[key] = post_dict[key]
            self.post_list.append(temp_post_dict)
            self.counter += 1

    def save_json(self):
        """
        Method to save the extracted information to a json file.
        """
        if not self.stat_list or not self.post_list:
            self.extract()
        with open(f"{self.out_path}/stats.json", "w") as outfile:
            json.dump(self.stat_list, outfile)
        with open(f"{self.out_path}/posts.json", "w") as outfile:
            json.dump(self.post_list, outfile)

    def strip_text(self, input_text, post_id):
        """
        Method to extract data from the text of a post, like the thread id, the quoted/green text and the text that is
        written by the poster.
        :param input_text: String of post to extract the data from.
        :param post_id: Id (integer) of post in order to trace exceptions during extractions.
        :return: Returns the full post, mainly for readability, a list of the quotes within the post, the (post) ids
        that are quoted, the text written by the user and a list of dead links.
        """
        full_string = ''
        quote_list = []
        quote_string = ''
        dead_links = []
        if input_text:
            try:
                doc = html.fromstring(input_text)
            except Exception as e:
                #todo: change to logging
                print(f"Error creating lxml doc from given text. Post no: {post_id}. Text is: {input_text}. "
                      f"Exception is {e}")
                return_dict = {'full_string': full_string,
                               'quoted_list': quote_list,
                               'quote_string': quote_string,
                               'dead_links_list': dead_links
                               }
                return [return_dict[post_info] for post_info in self.extract_from_post]
            try:
                for text in doc.itertext():
                    full_string = f"{full_string}{text} "
                for element in doc.iter():
                    if element.tag == 'a':
                        if element.attrib:
                            if 'class' in element.attrib.keys():
                                if element.attrib['class'] == 'quotelink':
                                    if element.text is not None:
                                        quote_id = element.text.strip('>>')
                                        if quote_id.isdigit():
                                            quote_list.append(int(quote_id))
                    elif element.tag == 'span':
                        if element.attrib:
                            if 'class' in element.attrib.keys():
                                if element.attrib['class'] == 'quote':
                                    quote_string = f"{quote_string} {element.text}"
                                elif element.attrib['class'] == 'deadlink':
                                    if element.text is not None:
                                        dead_id = element.text.strip('>>')
                                        if dead_id.isdigit():
                                            dead_links.append(int(dead_id))
                    elif element.tag == 'img':
                        full_string = f"{full_string}{element.attrib['alt']} "
            except Exception as e:
                #todo: change to logging
                print(f"Error extracting text from post no {post_id}. Text is: {input_text}. Exception is {e}")
        return_dict = {'full_string': full_string,
                       'quoted_list': quote_list,
                       'quote_string': quote_string,
                       'dead_links_list': dead_links
                       }
        return [return_dict[post_info] for post_info in self.extract_from_post]

    def generate_network(self, thread_id):
        #todo: change from using at to use of row.resto/row.quoted_list
        """
        Method to create a network graph for the provided thread_id
        :param thread_id: Id of the thread to create the network from.
        :return: Returns a nx.DiGraph
        """
        # initialize graph
        graph = nx.DiGraph()
        # iterate over DF filtered by the ID of the thread for which the graph is to be created
        for index, row in self.post_df[self.post_df.thread_id == thread_id].iterrows():
            graph.add_node(index)
            quote_list = self.post_df.at[index, 'quoted_list']
            # check if post ids have been quoted inside the post (visible as links on 4chan)
            if len(quote_list) != 0:
                # iterate over list of posts quoted/referred to
                for quote in quote_list:
                    graph.add_edge(index, int(quote))
            else:
                # as no ids have been quoted inside the post, a edge to the op will be created if the post isn't referring to a different thread
                resto = self.post_df.at[index, 'resto']
                if resto != 0:
                    graph.add_edge(index, resto)
        return graph

    def save_gexf(self, thread_id, path, save_cyclic=False):
        #todo: remove print, set up log and exception.
        """
        Save the nx.DiGraph to a gexf file for further processing.
        :param thread_id: Id of the thread to save, will be used to name the file.
        :param path: Path to save the file to.
        :param save_cyclic: Will determine if cyclic graphs are to be saved or not.
        """
        if thread_id in list(self.stat_df.index):
            g = self.generate_network(thread_id)
            is_acyclic = nx.algorithms.dag.is_directed_acyclic_graph(g)
            if is_acyclic:
                nx.write_gexf(g, f"{path}{thread_id}.gexf")
                self.stat_df.at[thread_id, 'is_acyclic'] = True
            elif not is_acyclic and save_cyclic:
                nx.write_gexf(g, f"{path}{thread_id}.gexf")
                self.stat_df.at[thread_id, 'is_acyclic'] = False
            else:
                self.stat_df.at[thread_id, 'is_acyclic'] = False
        else:
            #todo: change to exception
            print('Thread id not found. Please check if the provided thread id is correct.')

    def generate_edge_list(self, thread_id=None):
        """
        Method to create a list of edges of a graph network of a given thread.
        :param thread_id: Id of the thread to create the edge list for.
        :return: List of all edges of the thread network.
        """
        edge_list = []
        for index, row in self.post_df[self.post_df.thread_id == thread_id].iterrows():
            quotes_list = self.post_df.at[index, 'quoted_list']
            resto = self.post_df.at[index, 'resto']
            if len(quotes_list) != 0:
                for quote in quotes_list:
                    edge_list.append([int(index), int(quote)])
            elif resto != 0:
                edge_list.append([int(index), int(resto)])
        return edge_list

    def create_gexfs(self, min_replies=275, max_replies=325, language='en'):
        #todo: add option to not consider the language
        # todo create gexf (b-mode or id-mode)
        """
        Method to save thread networks of all threads within the range of replies to a .gexf file. Will use the output
        path of the class to save the files there.
        :param min_replies: Minimum number of replies within a thread to be saved.
        :param max_replies: Maximum number of replies within a thread to be saved.
        :param language: Consider only threads with the given language.
        """
        path = f"{self.out_path}/gexfs/{min_replies}-{max_replies}/"
        os.makedirs(path, exist_ok=True)
        thread_list = self.stat_df[(self.stat_df['replies'] >= min_replies) &
                                   (self.stat_df['replies'] <= max_replies) &
                                   (self.stat_df['language'] == language)].index
        if self.filter_cyclic:
            for thread_id in tqdm(thread_list, desc="Saving gexfs: "):
                self.save_gexf(thread_id, path)
        else:
            for thread_id in tqdm(thread_list, desc="Saving gexfs: "):
                self.save_gexf(thread_id, path, save_cyclic=True)

    def return_documents(self, text_column="own_text", min_replies=275, max_replies=325, language='en'):
        #todo: add option to not consider the language
        """
        Method to return a list of strings where each string is the text posted in a thread. Can consider different
        texts, e.g. only the quotes, only the text written by a user or the full posts.
        :param text_column: Column of text (quoted text, text written by the user or full post).
        :param min_replies: Minimum number of replies within a thread for a thread to be considered.
        :param max_replies: Maximum number of replies within a thread for a thread to be considered.
        :param language: Dominant language of a thread to be considered.
        :return: Returns a list of strings where each string is the text of a thread and a list of the thread ids,
        matching the documents.
        """
        if not isinstance(self.post_df, pd.DataFrame) or not isinstance(self.stat_df, pd.DataFrame):
            self.create_dfs()
        text_list = []
        if self.filter_cyclic:
            thread_list = self.stat_df[(self.stat_df['replies'] >= min_replies) &
                                       (self.stat_df['replies'] <= max_replies) &
                                       (self.stat_df['language'] == language) &
                                       (self.stat_df['is_acyclic'] == True)
                                       ].index
        else:
            thread_list = self.stat_df[(self.stat_df['replies'] >= min_replies) &
                                       (self.stat_df['replies'] <= max_replies) &
                                       (self.stat_df['language'] == language)
                                       ].index
        for thread_id in tqdm(thread_list, desc="Assembling text list."):
            text_list.append(" ".join(self.post_df[text_column][self.post_df.thread_id == thread_id].tolist()))
        return text_list, thread_list

    def detect_lang(self):
        """
        Method to detect the languages of each thread in the collection.
        """
        languages = []
        for thread_id in tqdm(self.stat_df.index, desc='Detecting languages: '):
            try:
                languages.append(detect(" ".join(self.post_df['full_string']
                                                 [self.post_df.thread_id == thread_id].tolist())))
            except:
                languages.append(None)
        self.stat_df['language'] = languages

    def save_df_pickles(self, path=None):
        """
        Method to save the pandas dataframes with the extracted information to a pickle file.
        :param path: Path to save the pickels to. If none, the output path of the class will be used.
        """
        if path is None:
            path = self.out_path
        self.stat_df.to_pickle(f"{path}stat_df.pkl")
        self.post_df.to_pickle(f"{path}post_df.pkl")

    def save_df_csvs(self, path=None):
        """
        Method to save the pandas dataframes with the extracted information to a csv file.
        :param path: Path to save the csv to. If none, the output path of the class will be used.
        """
        if path is None:
            path = self.out_path
        self.stat_df.to_csv(f"{path}stat_df.csv")
        self.post_df.to_csv(f"{path}post_df.csv")
