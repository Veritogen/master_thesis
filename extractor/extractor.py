import os
from collections import defaultdict
import json
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup as bs
import networkx as nx
import warnings
import logging
from langdetect import detect

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


#todo: setup logger
#todo: check for proper extraction (see chris project)

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
        self.stat_dict = None
        self.post_list = None
        self.relevant_stats = None
        self.ignore_keys = None
        self.post_keys = None
        self.stat_df = None
        self.post_df = None
        self.board = None
        self.thread_id = None
        self.json_file = None
        self.filter_cyclic = None
        self.loaded = False
        self.save_com = None
        self.counter = 0

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
        self.mode = mode
        if self.mode == "pol_set" and self.file_name is None:
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
                save_full_text=True, save_own_text=False, save_quote_text=False,):
        """
        Method to load the json files and set the according thread id/board within the class.
        :param filter_cyclic: If true, filter only threads where the post network in the thread is acyclic.
        :param complete_extraction: If true, all information provided by the 4chan API will be extracted, else just
        a limited set is used
        """
        if not self.loaded:
            logging.exception("Can't extract. No files loaded yet. Please use 'load' method to load files")
            raise Exception("Can't extract. No files loaded yet. Please use 'load' method to load files")

        self.relevant_stats = ['board', 'semantic_url', 'time', 'archived_on', 'replies', 'images', 'bumplimit',
                               'imagelimit']
        self.ignore_keys = {'semantic_url', 'archived_on', 'replies', 'images', 'bumplimit',
                            'imagelimit', 'closed', 'archived'}
        if complete_extraction:
            self.post_keys = ['no', 'now', 'name', 'sub', 'com', 'filename', 'ext', 'w', 'h', 'tn_w', 'tn_h', 'tim',
                              'time', 'md5', 'fsize', 'resto', 'trip', 'filedeleted', 'capcode', 'since4pass',
                              'country', 'country_name', 'tail_size', 'troll_country', 'm_img', 'custom_spoiler',
                              'spoiler']
            logging.debug("Extracting all possible information from posts.")
        else:
            self.post_keys = ['no', 'time', 'resto']
            logging.debug("Extracting limited set of information from posts.")
        self.save_com = save_com
        self.filter_cyclic = filter_cyclic
        self.stat_dict = {}
        self.post_list = []
        self.stat_df = pd.DataFrame(columns=self.relevant_stats)
        post_columns = self.post_keys[:]
        if save_full_text:
            post_columns.append('full_string')
        if save_own_text:
            post_columns.append('own_text')
        if save_quote_text:
            post_columns.append("quote_string")
        for column in ['thread_id', 'quoted_list']:
            post_columns.append(column)
        self.post_df = pd.DataFrame(columns=post_columns)
        if self.mode == 'legacy':
            logging.debug("Extracting information from files collected from 4chan API.")
            self.extract_legacy()
        if self.mode == 'pol_set':
            logging.debug("Extracting information from pol dataset.")
            self.extract_pol_set()

    def extract_legacy(self):
        if not self.file_dict:
            self.create_file_dict()
        for board in tqdm(self.file_dict.keys(), desc='Board'):
            self.board = board
            for thread_file in tqdm(self.file_dict[board], desc='Threads'):
                self.json_file = json.load(open(f"{self.in_path}/{board}/{thread_file}"))
                self.thread_id = int(thread_file.split('.')[0])
                self.extract_json()
                if self.counter > 10000:
                    temp_df = pd.DataFrame(columns=self.post_df.columns, data=self.post_list)
                    self.post_df = pd.concat([self.post_df, temp_df], ignore_index=True)
                    self.post_list = []
                    self.counter = 0
        temp_df = pd.DataFrame(columns=self.post_df.columns, data=self.post_list)
        self.post_df = pd.concat([self.post_df, temp_df], ignore_index=True)


    def extract_pol_set(self):
        self.post_keys.append('extracted_poster_id')
        self.board = 'pol'
        if self.file_name is None:
            with open(f"{self.in_path}pol_062016-112019_labeled.ndjson") as f:
                for line in tqdm(f, desc='Threads'):
                    self.json_file = json.loads(line)
                    self.thread_id = self.json_file['posts'][0]['no']
                    self.extract_json()
                    if self.counter > 10000:
                        temp_df = pd.DataFrame(columns=self.post_df.columns, data=self.post_list)
                        self.post_df = pd.concat([self.post_df, temp_df], ignore_index=True)
                        self.post_list = []
                        self.counter = 0
            temp_df = pd.DataFrame(columns=self.post_df.columns, data=self.post_list)
            self.post_df = pd.concat([self.post_df, temp_df], ignore_index=True)
        else:
            with open(f"{self.in_path}/{self.file_name}") as f:
                for line in tqdm(f, desc='Threads'):
                    self.json_file = json.loads(line)
                    self.thread_id = self.json_file['posts'][0]['no']
                    self.extract_json()
                    if self.counter > 10000:
                        temp_df = pd.DataFrame(columns=self.post_df.columns, data=self.post_list)
                        self.post_df = pd.concat([self.post_df, temp_df], ignore_index=True)
                        self.post_list = []
                        self.counter = 0
            temp_df = pd.DataFrame(columns=self.post_df.columns, data=self.post_list)
            self.post_df = pd.concat([self.post_df, temp_df], ignore_index=True)

    def extract_json(self):
        #todo: add log instead of print
        """
        Method to extract the information that is contained in the json files, loaded by the method "extract". Will
        save information on the statistics per thread and extract data from the text of each post. Will also save all
        information provided per post by the API.
        """
        for post in self.json_file['posts']:
            if post['no'] == self.thread_id:
                stat_temp = {'board': self.board}
                for rel_stat in self.relevant_stats:
                    if rel_stat != 'board':
                        try:
                            stat_temp[rel_stat] = post[rel_stat]
                        except KeyError:
                            logging.debug(f"Missing information for {rel_stat} in thread no {self.thread_id}")
                            stat_temp[rel_stat] = 'MISSING'
                self.stat_df.loc[self.thread_id] = stat_temp
            post_dict = {'thread_id': self.thread_id,
                         'board': self.board}
            for key in self.post_keys:
                if key in post.keys():
                    post_dict[key] = post[key]
                else:
                    post_dict[key] = None
                if 'com' in post.keys():
                    if self.save_com:
                        post_dict['com'] = post['com']
                    try:
                        full_string, quoted_list, quote_string, own_text, dead_links = self.strip_text(post['com'])
                    except Exception as e:
                        logging.error(f"Exception while extracting post {post['no']} in thread {self.thread_id}. {e}")
                else:
                    if self.save_com:
                        post_dict['com'] = ""
                    full_string, quoted_list, quote_string, own_text, dead_links = "", [], "", "", []
                post_dict['full_string'] = full_string
                post_dict['quoted_list'] = quoted_list
                post_dict['quote_string'] = quote_string
                post_dict['own_text'] = own_text
                post_dict['dead_links'] = dead_links
            try:
                temp_post_dict = {}
                for key in self.post_df.columns:
                    temp_post_dict[key] = post_dict[key]
                self.post_list.append(temp_post_dict)
                self.counter += 1
            except Exception as e:
                print(self.post_df.columns, post_dict.keys(), e)

    def save_json(self):
        """
        Method to save the extracted information to a json file.
        """
        if not self.stat_dict or not self.post_list:
            self.extract()
        with open(f"{self.out_path}/stats.json", "w") as outfile:
            json.dump(self.stat_dict, outfile)
        with open(f"{self.out_path}/posts.json", "w") as outfile:
            json.dump(self.post_list, outfile)

    def create_dfs(self):
        """
        Method to create a pandas dataframe for the extracted statistics and another one for information on the posts.
        Will also detect the (dominant) language in each thread and save it to the statistics dataframe.
        """
        if not self.stat_dict or not self.post_list:
            self.extract()
        self.stat_df = pd.DataFrame(data=self.stat_dict).transpose()
        self.stat_df.index = self.stat_df.no
        self.stat_df = self.stat_df.drop(columns='no')
        self.stat_df['is_acyclic'] = False
        post_columns = self.post_keys[:]
        for column in ['thread_id', 'full_string', 'quoted_list', 'quote_string', 'own_text']:
            post_columns.append(column)
        self.post_df = pd.DataFrame(data=self.post_list, columns=post_columns)
        self.post_df[['resto', 'time', 'thread_id']] = self.post_df[['resto', 'time', 'thread_id']].astype(int)
        self.post_df.index = self.post_df.no
        self.post_df = self.post_df.drop(columns='no')
        self.detect_lang()

    def strip_text(self, text):
        """
        Method to extract data from the text of a post, like the thread id, the quoted/green text and the text that is
        written by the poster.
        :param text: String of post to extract the data from.
        :return: Returns the full post, mainly for readability, a list of the quotes within the post, the (post) ids
        that are quoted, the text written by the user and a list of dead links.
        """
        # todo: handle dead link class
        soup = bs(text, 'html.parser')
        full_string = ''
        quoted_list = []
        quote_string = ''
        own_text = ''
        dead_link_list = []
        for item in soup.contents:
            if str(item).startswith('<a class="quotelink"'):
                quote_ids = soup.find_all(class_='quotelink')
                for quote_id in quote_ids:
                    full_string = full_string + quote_id.contents[0]
                    if quote_id.contents[0].strip('>>').isdigit():
                        quoted_list.append(int(quote_id.contents[0].strip('>>')))
            elif str(item).startswith('<span class="quote">'):
                quotes = soup.find_all(class_='quote')
                for quote in quotes:
                    full_string = full_string + '"' + str(quote.contents[0]) + '" '
                    quote_string = quote_string + ' ' + str(quote.contents[0])
            elif str(item).startswith('<br/>'):
                full_string = full_string + ' \n '
            elif str(item).startswith('<span class="deadlink'):
                dead_links = soup.find_all(class_='deadlink')
                for dead_link in dead_links:
                    if dead_link.contents[0].strip('>>').isdigit():
                        dead_link_list.append(int(dead_link.contents[0].strip('>>')))
                    else:
                        dead_link_list.append(dead_link.contents[0])
            else:
                full_string = full_string + str(item)
                own_text = own_text + ' ' + (str(item))
        return full_string, quoted_list, quote_string, own_text, dead_link_list

    def generate_network(self, thread_id):
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
