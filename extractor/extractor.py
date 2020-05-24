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

class Extractor:
    def __init__(self, path):
        self.path = path
        self.file_dict = None
        self.stat_dict = None
        self.post_list = None
        self.relevant_stats = ['no','semantic_url', 'time', 'archived_on',  'replies', 'images', 'bumplimit',
                               'imagelimit']
        self.ignore_keys = {'semantic_url', 'archived_on',  'replies', 'images', 'bumplimit',
                               'imagelimit', 'closed', 'archived'}
        ''' 
        List of all possible information, shortened in order to save memory.
        self.post_keys = ['no', 'now', 'name', 'sub', 'com', 'filename', 'ext', 'w', 'h', 'tn_w', 'tn_h', 'tim', 'time',
                          'md5', 'fsize', 'resto', 'trip', 'filedeleted', 'capcode', 'since4pass', 'country',
                          'country_name', 'tail_size', 'troll_country', 'm_img', 'custom_spoiler', 'spoiler'] 
                          '''
        self.post_keys = ['no', 'now', 'name', 'com', 'time', 'md5', 'resto', 'trip', 'filedeleted', 'country',
                          'country_name', 'troll_country']
        self.stat_df = None
        self.post_df = None

    def create_file_dict(self):
        self.file_dict = defaultdict(list)
        for item in os.listdir(self.path):
            if not os.path.isfile(f"{self.path}/{item}"):
                for file in os.listdir(f"{self.path}/{item}"):
                    if file.endswith(".json"):
                        self.file_dict[item].append(file)

    def extract(self):
        self.stat_dict = {}
        self.post_list = []
        if not self.file_dict:
            self.create_file_dict()
        for board in tqdm(self.file_dict.keys(), desc='Board'):
            for thread_file in tqdm(self.file_dict[board], desc='Threads'):
                json_file = json.load(open(f"{self.path}/{board}/{thread_file}"))
                thread_id = int(thread_file.split('.')[0])
                for post in json_file['posts']:
                    if post['no'] == thread_id:
                        self.stat_dict[post['no']] = {'board': board}
                        for rel_stat in self.relevant_stats:
                            self.stat_dict[post['no']][rel_stat] = post[rel_stat]
                    post_dict = {'thread_id': thread_id,
                                 'board': board}
                    for key in self.post_keys:
                        if key in post.keys():
                            post_dict[key] = post[key]
                        else:
                            post_dict[key] = None
                        if key == 'com':
                            try:
                                if key in post.keys():
                                    full_string, quoted_list, quote_string, own_text = self.strip_text(post[key])
                                # if no comment is written, add emtpy string
                                else:
                                    full_string, quoted_list, quote_string, own_text = self.strip_text("")
                                    #todo: log missing text
                                post_dict['full_string'] = full_string
                                post_dict['quoted_list'] = quoted_list
                                post_dict['quote_string'] = quote_string
                                post_dict['own_text'] = own_text
                            except:
                                print(f"board: {board}, Thread: {thread_id}, Post: {post['no']}"
                                      f" Post keys: {post.keys()}, Full Post: {post}")
                                #todo: add board and Id to log
                                #logging.ex(post, post_dict)
                    self.post_list.append(post_dict)

    def save_json(self):
        if not self.stat_dict or not self.post_list:
            self.extract()
        with open(f"{self.path}/stats.json", "w") as outfile:
            json.dump(self.stat_dict, outfile)
        with open(f"{self.path}/posts.json", "w") as outfile:
            json.dump(self.post_list, outfile)

    def create_dfs(self):
        if not self.stat_dict or not self.post_list:
            self.extract()
        self.stat_df = pd.DataFrame(data=self.stat_dict).transpose()
        self.stat_df.index = self.stat_df.no
        self.stat_df = self.stat_df.drop(columns='no')
        post_columns = self.post_keys[:]
        for column in ['thread_id', 'full_string', 'quoted_list', 'quote_string', 'own_text']:
            post_columns.append(column)
        self.post_df = pd.DataFrame(data=self.post_list, columns=post_columns)
        self.post_df[['resto', 'time', 'thread_id']] = self.post_df[['resto', 'time', 'thread_id']].astype(int)
        self.post_df.index = self.post_df.no
        self.post_df = self.post_df.drop(columns='no')
        self.detect_lang()

    def strip_text(self, text):
        #todo: handle dead link class
        soup = bs(text, 'html.parser')
        full_string = ''
        quoted_list = []
        quote_string = ''
        own_text = ''
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
            else:
                full_string = full_string + str(item)
                own_text = own_text + ' ' + (str(item))
        return full_string, quoted_list, quote_string, own_text

    def generate_network(self, thread_id):
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

    def save_gexf(self, thread_id=None):
        if thread_id in list(self.stat_df.index):
            nx.write_gexf(self.generate_network(thread_id), f"{self.path}/gexfs/{thread_id}.gexf")
        else:
            print(
                'Die angegebene Thread ID wurde im Datensatz nicht gefunden. Bitte überprüfe, ob die ID richtig ist.')

    def generate_edge_list(self, thread_id=None):
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

    def create_gexfs(self, min_replies=275, max_replies=325):
        os.makedirs(f"{self.path}/gexfs/", exist_ok=True)
        thread_list = self.stat_df[(self.stat_df['replies'] >= min_replies) & (self.stat_df['replies'] <= max_replies)]\
            .index
        for thread_id in tqdm(thread_list):
            self.save_gexf(thread_id)
        #todo create gexf (b-mode or id-mode)

    def return_document_list(self, text_column="own_text", min_replies=275, max_replies=325, language='en'):
        if not isinstance(self.post_df, pd.DataFrame) or not isinstance(self.stat_df, pd.DataFrame):
            self.create_dfs()
        text_list = []
        for thread_id in tqdm(self.stat_df[(self.stat_df['replies'] >= min_replies) &
                                           (self.stat_df['replies'] <= max_replies) &
                                           (self.stat_df['language'] == language)].index, desc="Assembling text list."):
            text_list.append(" ".join(self.post_df[text_column][self.post_df.thread_id == thread_id].tolist()))
        return text_list

    def detect_lang(self):
        languages = []
        for thread_id in tqdm(self.stat_df.index, desc='Detecting languages: '):
            try:
                languages.append(detect(" ".join(self.post_df['full_string']
                                                 [self.post_df.thread_id == thread_id].tolist())))
            except:
                languages.append(None)
        self.stat_df['language'] = languages
