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
# import your token and target id for the telegram bot

try:
    telegram_target = os.environ['telegram_target']
except:
    raise Warning("Variable 'telegram_token' not found in environment variables.")


def setup_bot(token=None):
    if token is None:
        try:
            token = os.environ['telegram_token']
            return telegram.Bot(token=token)
        except KeyError:
            raise Exception("Telegram bot not set up. Set variable 'telegram_token' in environment variables or "
                            "provide it as argument.")
    else:
        return telegram.Bot(token=token)


def send_msg(bot, message, target=None):
    if target is None:
        try:
            target = os.environ['telegram_target']
            bot.send_message(target, message)
        except KeyError:
            raise Exception("No target for Telegram message provided. Set variable 'telegram_target' in environment "
                            "variables or provide it as argument.")
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
        import proxies
        proxy_config = proxies.proxies
    else:
        proxy_config = None
    if getattr(args, 'real_ip') == 'True':
        real_ip = True
    else:
        real_ip = False
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


tel_bot = setup_bot()

try:
    if __name__ == '__main__':
        main(sys.argv)
except Exception as e:
    logging.exception(f"Main execution failed.", exc_info=True)
    if tel_bot in locals() and 'telegram_target' in os.environ.keys():
        send_msg(tel_bot, f"A exception occurred at {time.strftime('%a %d.%m.%Y %H:%M:%S')}. Exception message: /n {e}."
                 f" Traceback: {traceback.format_exc()}")
