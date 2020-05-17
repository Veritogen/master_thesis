import requests as rq
import time
import logging
import json
from Exceptions import *
import datetime
from copy import deepcopy
from queue import Queue, PriorityQueue
import os
import threading
#logging.basicConfig(filename='log.log', filemode='a', format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)


class Scrape4chan:
    #todo: init meta at beginning
    #todo: scrape id's from archive files

    def __init__(self, collection_type, boards, start_time, end_time, path, stat_table, meta_stats, telegram_bot=None,
                 telegram_target=None, proxies=None, use_real_ip=True):
        """
        :param collection_type: info if live or archived threads are to be collected
        :param board: board of 4chan that will be scraped
        :param start_time: start time of collection
        :param end_time: end time of collection
        :param path: path to the folder where to save the scraped files
        :param stat_table: peewee table model for the per thread statistics.
        :param meta_stats: peewee table model for the global statistics.
        :param telegram_bot: Bot to send messages of exceptions via telegram
        :param telegram_target: ID to send messages to using the telegram bot
        """
        logging.basicConfig(filename=f'{path}log.log', filemode='a', format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.getLogger('peewee').setLevel(logging.INFO)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        if telegram_bot:
            self.bot = telegram_bot
            self.target = telegram_target
        else:
            self.bot = None
        self.boards = boards
        self.collection_type = collection_type
        self.path = path
        self.finished = False
        self.collection_ended = False
        self.Stats = stat_table
        self.Meta = meta_stats
        self.thread_dict = {board: {} for board in self.boards}
        self.old_thread_dict = {board: {} for board in self.boards}
        self.last_threads = {board: {} for board in self.boards}
        self.boards_with_id = ['pol', 'bant', 'soc']
        self.thread_get_list = []
        self.download_queue = PriorityQueue()
        self.proxy_queue = Queue()
        self.proxies = proxies
        if proxies:
            self.max_thread_count = len(self.proxies)
        if self.proxies:
            for proxy in proxies:
                self.proxy_queue.put({'https': proxy})
        if use_real_ip:
            self.proxy_queue.put(None)
            self.max_thread_count += 1
        meta = self.Meta.get_or_none(self.Meta.id == 1)
        if meta:
            self.start_time = meta.start_time
            self.end_time = meta.end_time
            if time.time() > self.end_time:
                self.finished = True
            logging.info("Continuing collecting threads after restart.")
            if not meta.collection_ended:
                last_threads_from_db = self.Stats.select().where(self.Stats.finished == 0)
                for board in self.boards:
                    self.last_threads[board] = {}
                for thread in last_threads_from_db:
                    self.last_threads[thread.board][thread.thread_id] = thread.last_modified
                logging.info(f"Continuing collection of last threads after restart. {self.get_last_thread_no} "
                             f"threads remaining.")
                # load unfinished threads to last_threads
            else:
                self.collection_ended = True
        else:
            self.start_time = start_time
            self.end_time = end_time
            for board in self.boards:
                os.makedirs(f"{self.path}{board}/", exist_ok=True)
            self.Meta.create(id=1, no_threads=self.Stats.select().count(), iterations=0, start_time=self.start_time,
                             end_time=self.end_time, collection_end=int(time.time()),
                             collection_type=self.collection_type)
            logging.debug('Meta stats created')
        logging.info(f'Scrape4Chan set up to scrape boards {self.boards} for {self.end_time - self.start_time}, '
                     f'saving to {self.path}.')

    def send_msg(self, message):
        self.bot.send_message(self.target, message)

    def get_last_thread_no(self):
        no = 0
        for board in self.boards:
            if self.collection_type == 'live':
                no += len(self.last_threads[board].keys())
            else:
                no += len(self.last_threads[board])
        return no

    def get_link(self, link):
        """
        Function to retrieve links. Will try five times to get the link.
        If not possible, will raise a exception after the fifths time.
        :param link: link to get via requests.
        """
        proxy = self.proxy_queue.get()
        tries = 0
        while tries < 5:
            time.sleep(1)
            try:
                file = rq.get(link, timeout=10, proxies=proxy)
            except:
                logging.exception(f"Couldn't get link. Traceback: ", exc_info=True)
                tries += 1
                time.sleep(3)
                continue
            tries += 1
            if file.status_code == 200:
                logging.debug(f'Link {link} retrieved with http code 200.')
                self.proxy_queue.task_done()
                self.proxy_queue.put(proxy)
                return file
            elif file.status_code == 404:
                logging.debug(f'Link {link} not retrieved. Received http code 404.')
                self.proxy_queue.task_done()
                self.proxy_queue.put(proxy)
                raise ConnectionException(file.status_code)
            else:
                logging.info(f'Link {link} not retrieved. Received http code {file.status_code} Sleeping 5 seconds.')
                time.sleep(5)
        raise Exception(f"Couldn't retrieve link {link} after 5 retries. Aborting.")

    def setup_queue(self):
        """
        Function to get a dictionary of thread. Will also set up some stuff needed later
        (for statistical reasons mainly).
        """
        self.old_thread_dict = deepcopy(self.thread_dict)
        self.thread_dict = {}
        board_priority_counter = 0
        for board in self.boards:
            thread_priority_counter = 0
            self.thread_dict[board] = {}
            if self.collection_type == 'live':
                try:
                    threads_json = self.get_link(f'https://a.4cdn.org/{board}/threads.json')
                    logging.debug(f'Retrieved thread file for board {board}')
                except:
                    logging.exception(f"Couldn't retrieve link.", exc_info=True)
                    raise GetThreadException('Error retrieving thread list.')
                for page in reversed(threads_json.json()):
                    for thread in reversed(page['threads']):
                        last_modified = thread['last_modified']
                        thread_id = thread['no']
                        db_entry = self.Stats.get_or_none((self.Stats.thread_id == thread_id)
                                                          & (self.Stats.board == board))
                        if db_entry:
                            if last_modified - db_entry.last_modified > 10 and last_modified - db_entry.last_get > 10:
                                self.download_queue.put((thread_priority_counter, board_priority_counter, board,
                                                        thread_id))
                            logging.debug(f'Thread no {thread_id} exists in DB. Updating.')
                            db_entry.last_modified = last_modified
                            db_entry.seen = db_entry.seen + 1
                            db_entry.save()
                        else:
                            if not self.finished:
                                logging.debug(f'Thread no {thread_id} not yet in DB. Creating entry.')
                                self.Stats.create(thread_id=thread_id, board=board,
                                                  last_modified=last_modified, seen=1, collected=0,
                                                  finished=0, archived=0, last_get=0)
                                self.download_queue.put((thread_priority_counter, board_priority_counter, board,
                                                         thread_id))
                        self.thread_dict[board][thread_id] = last_modified
                        thread_priority_counter += 1
                logging.debug(f"Threads of board {board} added.")
            else:
                try:
                    threads_json = self.get_link(f"https://a.4cdn.org/{board}/archive.json")
                    logging.debug(f'Retrieved archive file for board {board}')
                except:
                    logging.exception(f"Couldn't retrieve archive list for board {board}.", exc_info=True)
                    continue
                self.thread_dict[board] = set(threads_json.json())
                for thread_id in reversed((threads_json.json())):
                    db_entry = self.Stats.get_or_none((self.Stats.thread_id == thread_id) & (self.Stats.board == board))
                    if not db_entry:
                        self.Stats.create(thread_id=thread_id, board=board, last_modified=0, seen=1, collected=0,
                                          finished=0, archived=0, last_get=0)
                        self.download_queue.put((thread_priority_counter, board_priority_counter, board, thread_id))
                    elif db_entry and db_entry.collected == 0:
                        if not db_entry.finished == 1:
                            self.download_queue.put((thread_priority_counter, board_priority_counter, board, thread_id))
                    thread_priority_counter += 1
            board_priority_counter += 1
        # todo: change back to debug
        logging.info(f"Thread queue updated. Contains {self.download_queue.qsize()} entries.")

    def is_finished(self):
        """
        Function to check if collection is finished.
        """
        if time.time() > self.end_time and not self.collection_ended:
            logging.info("Collection time over. Collecting already started threads.")
            self.finished = True
            if self.get_last_thread_no() == 0:
                self.create_last_threads()
        if self.finished and self.collection_ended:
            logging.info(f'Collection finished at {time.time()}.')
            return True
        else:
            if self.finished:
                logging.debug(f"{self.get_last_thread_no()} threads left to collect.")
            return False

    def update_collected_thread(self, thread_id):
        """
        Function to update the statistics of downloads per thread.
        :param thread_id: Id of thread to update.
        """
        collected = self.Stats.get(self.Stats.thread_id == thread_id)
        collected.collected = collected.collected + 1
        collected.archived = int(time.time())
        collected.save()
        logging.debug(f"Thread no {thread_id} updated after collection.")

    def check_integrity(self, posts):
        """
        Function to check if the json-file is ok. Sometimes the API hands over broken jsons.
        :param posts: Dictionary of posts contained in the json of the thread.
        """
        for post in posts:
            if 'no' in post.keys():
                logging.debug('Thread file integrity checked with success.')
                return True
            else:
                logging.debug('Thread file integrity checked without success.')
                return False

    def get_threads(self):
        """
        Function to iterate over the thread ids and pass them over to the downloader.
        """
        # reversed in order to collect the threads that will be dead soon first.
        thread_counter = 0
        while not self.download_queue.empty():
            if not self.proxies:
                board, thread_id = self.download_queue.get()[2:]
                try:
                    self.download_thread(board, thread_id)
                except:
                    logging.exception(f"get_threads failed to download thread no {thread_id} in board {board}")
                    continue
                self.download_queue.task_done()
            else:
                queue_entry = self.download_queue.get()
                board, thread_id = queue_entry[2:]
                self.download_queue.task_done()
                try:
                    worker = threading.Thread(target=self.download_thread, args=(board, thread_id,), daemon=True)
                    thread_counter += 1
                    worker.start()
                    if thread_counter == self.max_thread_count:
                        worker.join()
                        thread_counter = 0
                    #logging.info("worker started")
                except:
                    logging.exception(f"Error starting threading thread for thread no {thread_id} in board "
                                      f"{board}.")
                    self.download_queue.put(queue_entry)

    def create_last_threads(self):
        """
        Function to set the threads that have been started during the collection time. This way no incomplete threads
        will be in the collection.
        """
        self.last_threads = deepcopy(self.thread_dict)
        logging.info(f'last_threads created. Contains {self.get_last_thread_no()} entries.')

    def download_thread(self, board, thread_id):
        """
        Function for downloading the threads. Will pass files to function for saving files.
        :param board: Name of the board to pass to the download and the save function.
        :param thread_id: Thread id to pass to the download and the save function.
        """
        try:
            thread_file = self.get_link(f'https://a.4cdn.org/{board}/thread/{thread_id}.json')
            if board in self.boards_with_id:
                thread_html = self.get_link(f"https://boards.4chan.org/{board}/thread/{thread_id}")
                self.save_txt(board, thread_id, str(thread_html.content))
            if self.check_integrity(thread_file.json()['posts']):
                self.save_json(board, thread_id, thread_file.json())
                self.update_collected_thread(thread_id)
                logging.debug(f'Thread no {thread_id} in board {board} downloaded successfully.')
                db_entry = self.Stats.get_or_none((self.Stats.thread_id == thread_id)
                                                  & (self.Stats.board == board))
                db_entry.last_get = int(time.time())
                db_entry.save()
            else:
                logging.exception(f"Could not verify integrity of thread no {thread_id}. Thread put back into queue.")
                self.download_queue.put((3, 3, thread_id, board))
        except ConnectionException as e:
            if e.args[0] == 404:
                logging.info(f'Thread no {thread_id} in board {board} is dead (http status 404).')
                self.set_finished(board, thread_id)
            else:
                logging.exception(f'Error collecting thread no {thread_id} in board {board}. Has http status '
                                  f'{e.args[0]}. Thread put back into queue.')
                self.download_queue.put((3, 3, thread_id, board))
        except Exception:
            logging.exception(f"download_thread: Couldn't get thread no {thread_id} in board {board}.", exc_info=True)
        return None

    def save_json(self, board, thread_id, json_file):
        """
        Function to save the json files.
        :param board: Name of the board. Will be used to select folder.
        :param thread_id: Id of the thread to be saved. Will be used in the file name.
        :param json_file: File to save to disk.
        """
        with open(f'{self.path}{board}/{thread_id}.json', 'w') as outfile:
            json.dump(json_file, outfile)
            logging.debug(f"Json of thread no {thread_id} in board {board} saved successfully to disk.")

    def save_txt(self, board, thread_id, text):
        """
        Function to save the txt files.
        :param board: Name of the board. Will be used to select folder.
        :param thread_id: Id of the thread to be saved. Will be used in the file name.
        :param text: String to save to disk.
        """
        with open(f'{self.path}{board}/{thread_id}.txt', 'w') as outfile:
            outfile.write(text)
            logging.debug(f"Text of thread no {thread_id} in board {board} saved successfully to disk.")

    def update_meta(self):
        """
        Function to update statistics on the collection itself.
        """
        meta = self.Meta.get_or_none(self.Meta.id == 1)
        if meta:
            meta.iterations = meta.iterations + 1
            meta.no_threads = self.Stats.select().count()
            meta.collection_end = time.time()
            meta.save()
            logging.debug('Meta stats updated')

    def archive_threads(self):
        archive_list = []
        for board in self.boards:
            if self.collection_type == 'live':
                live_threads = set(self.thread_dict[board].keys())
                if not self.finished:
                    for thread_id in self.old_thread_dict[board].keys():
                        if thread_id not in live_threads:
                            self.set_finished(board, thread_id)
                else:
                    for thread_id in self.last_threads[board]:
                        if thread_id not in live_threads:
                            self.set_finished(board, thread_id)
                            archive_list.append((board, thread_id))
        if self.collection_type == 'live':
            for del_board, del_thread in archive_list:
                del self.last_threads[del_board][del_thread]
            if self.get_last_thread_no() == 0 and self.finished:
                self.collection_ended = True
        else:
            if self.finished:
                self.collection_ended = True

    def set_finished(self, board, thread_id):
        finished = self.Stats.get((self.Stats.thread_id == thread_id) & (self.Stats.board == board))
        finished.finished = 1
        finished.save()
        logging.debug(f'Link {thread_id} set to "finished".')

    def collect(self):
        if time.time() < self.start_time:
            time.sleep(self.start_time - time.time())
        self.send_msg(f"Collection of threads started.")
        last_message = time.time()
        while not self.is_finished():
            try:
                self.setup_queue()
            except:
                logging.exception('Failed to create thread dict. Sleeping 30s until retry.')
                time.sleep(30)
                continue
            self.get_threads()
            self.archive_threads()
            self.update_meta()
            meta = self.Meta.get_or_none(self.Meta.id == 1)
            if time.time() - last_message > 86400:
                if meta:
                    td = datetime.timedelta(seconds=self.end_time - time.time())
                    if self.target is not None and self.bot is not None:
                        try:
                            self.send_msg(f"Collection contains {meta.no_threads} threads after {meta.iterations} "
                                          f"iterations. Collecting last in {td.days} days, {td.seconds // 3600} hours "
                                          f"and {(td.seconds // 60) % 60} minutes.")
                            logging.info(f"Collection contains {meta.no_threads} threads after {meta.iterations} "
                                         f"iterations. Collecting last in {td.days} days, {td.seconds // 3600} hours "
                                         f"and {(td.seconds // 60) % 60} minutes.")
                        except:
                            logging.exception("Couldn't info message via bot. Appending to log.", exc_info=True)
                last_message = time.time()
        meta = self.Meta.get_or_none(self.Meta.id == 1)
        logging.info(f"Collection finished after {meta.iterations} iterations.")
        if self.target is not None and self.bot is not None:
            try:
                self.send_msg(f"Collection finished after {meta.iterations} iterations.")
            except:
                logging.exception("Couldn't info message via bot. Appending to log.", exc_info=True)
