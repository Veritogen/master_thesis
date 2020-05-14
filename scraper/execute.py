from Scrape4Chan import *
from peewee import *
from Exceptions import *
import requests as rq
import time
import logging
import json
import os
import sys
import argparse
from copy import deepcopy
import telegram
import traceback
import ast
# import your token and target id for the telegram bot


def setup_bot(token):
    if token is None:
        return None
    else:
        return telegram.Bot(token=token)


def send_msg(bot, message, target=None):
    if target is None:
        raise Warning("No target for Telegram message provided. Set variable 'telegram_target' and 'telegram_token' in "
                      "environment variables or provide it as argument in order to receive updates via Telegram.")
    else:
        bot.send_message(target, message)


# function to initiate the scraping using the parameters handed over at executing
def main(argv):
    parse = argparse.ArgumentParser(description='Execute Scrape4Chan, a scraper for 4chan.org')
    parse.add_argument(
        '--boards',
        default=['b'],
        help='Choose the board you want to scrape.'
    )
    parse.add_argument(
        '--amount',
        type=int,
        default=1,
        choices=range(1000),
        help='Select amount of time you want to scrape given the unit you specified.'
    )
    parse.add_argument(
        '--unit',
        default='mon',
        choices=['min', 'h', 'd', 'wk', 'mon', 'yr'],
        help='Select the time unit you want specify the amount for.'
    )
    parse.add_argument(
        '--path',
        default=f"{os.path.abspath(os.getcwd())}",
        help='Enter the path to the directory you want to save the db and files to scrape to.'
    )
    parse.add_argument(
        '--name',
        help='Enter the name of the collection (will create a sub folder matching the name).'
    )
    parse.add_argument(
        '--start_date',
        help='Enter the date (DD.MM.YYYY) you want the collection to start.'
    )
    parse.add_argument(
        '--start_time',
        help='Enter the time (HH:MM) you want your collection to start'
    )
    parse.add_argument(
        '--type',
        choices=['live', 'archive'],
        help='Enter the time (HH:MM) you want your collection to start'
    )
    parse.add_argument(
        '--proxies',
        choices=["True", "False"],
        default="False",
        help='Enter if proxies are to be used.'
    )
    parse.add_argument(
        '--real_ip',
        choices=["True", "False"],
        default="True",
        help='Enter if the real ip is to be used.'
    )
    args = parse.parse_args(sys.argv[1:])
    path = f"{getattr(args, 'path')}/{getattr(args, 'name')}/"
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(filename=f"{path}log.log", filemode='a',
                        format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    logging.debug(f"Using path {path}.")
    logging.info(f"Starting collection on {getattr(args,'start_date')} at {getattr(args,'start_time')}.")
    start_date = time.mktime(time.strptime(f"{getattr(args, 'start_date')}", "%d.%m.%Y"))
    start_time = int(getattr(args, 'start_time').split(":")[0])*60*60 \
                 + int(getattr(args, 'start_time').split(":")[1])*60
    start_time_s = start_date + start_time

    # create db and db models::
    db = SqliteDatabase(f"{path}db.sqlite")

    class Stats(Model):
        id = AutoField()
        thread_id = IntegerField()
        board = TextField()
        last_modified = IntegerField()
        last_get = IntegerField()
        seen = IntegerField()
        collected = IntegerField()
        archived = IntegerField()
        finished = IntegerField()
        class Meta:
            database = db
            table_name = 'stats'

    class MetaStats(Model):
        id = IntegerField()
        no_threads = IntegerField()
        iterations = IntegerField()
        start_time = IntegerField()
        end_time = IntegerField()
        collection_ended = BooleanField(default=False)
        collection_end = IntegerField()
        collection_type = TextField()

        class Meta:
            database = db
            table_name = 'metastats'

    db.connect()

    try:
        logging.debug('Creating table in the database.')
        Stats.create_table()
    except:
        logging.info('Table already present in database, using existing table.')

    try:
        logging.debug('Meta stats table in the database.')
        MetaStats.create_table()
    except:
        logging.info('Meta stats table already present in database, using existing table.')

    # initiate scraper
    run_unit_multiplicators = {
        'min': 60,
        'h': 60 * 60,
        'd': 60 * 60 * 24,
        'wk': 60 * 60 * 24 * 7,
        'mon': 60 * 60 * 24 * 30,
        'yr': 60 * 60 * 24 * 365
    }
    run_time = getattr(args, 'amount') * run_unit_multiplicators[getattr(args, 'unit')]
    end_time = start_time_s + run_time
    boards = [board.strip(' ') for board in getattr(args, 'boards').split(",")]
    if getattr(args, 'proxies') == 'True':
        try:
            import proxies
            proxy_config = proxies.proxies
        except:
            proxy_config = None
            raise Exception("Please provide proxy information in a file named 'proxies.py' as a list (proxies) or set"
                            " a environment variable namend 'proxy_list'.")
    if getattr(args, 'real_ip') == 'True':
        real_ip = True
    else:
        real_ip = False
    if proxy_config is None and not real_ip:
        raise Exception("Please provide either a proxy list or allow the use of the real ip.")
    scrape = Scrape4chan(getattr(args, 'type'), boards, start_time_s, end_time, path, Stats, MetaStats,
                         telegram_bot=tel_bot, telegram_target=telegram_target, proxies=proxy_config,
                         use_real_ip=real_ip)
    try:
        send_msg(tel_bot, f"Scraper started. Collection name: {getattr(args,'name')}, for board(s) {', '.join(boards)}"
                          f" for {getattr(args, 'amount')} {getattr(args, 'unit')} using path {path}")
        logging.info(f"Scraper started. Collectigit on name: {getattr(args,'name')}, for board(s) {', '.join(boards)} for"
                     f" {getattr(args, 'amount')} {getattr(args, 'unit')} using path {path}")
    except:
        logging.exception("Couldn't info message via bot. Appending to log.", exc_info=True)
    scrape.collect()

try:
    import telegram_config
    telegram_target = telegram_config.target
    telegram_token = telegram_config.token
except KeyError:
    telegram_target = None
    telegram_token = None
    raise Warning("Importing Telegram config failed. Updates via Telegram won't work.")


tel_bot = setup_bot(telegram_token)

try:
    if __name__ == '__main__':
        main(sys.argv)
except Exception as e:
    logging.exception(f"Main execution failed.", exc_info=True)
    if tel_bot is not None and telegram_target is not None:
        send_msg(tel_bot, f"A exception occurred at {time.strftime('%a %d.%m.%Y %H:%M:%S')}. Exception message: /n {e}."
                 f" Traceback: {traceback.format_exc()}")
