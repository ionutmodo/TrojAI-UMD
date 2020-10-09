import os
import sys
from datetime import datetime


class Logger:
    FILE = None
    _START_TIME = None

    @staticmethod
    def open(file, mode='w', log_time=True):
        basename = os.path.basename(file)
        folder = file.replace(basename, '')
        if not os.path.exists(folder):
            os.makedirs(folder)

        if os.path.isfile(file):
            Logger.FILE = open(file, 'a')
            Logger.log('START APPENDING TO EXISTING FILE')
        else:
            Logger.FILE = open(file, mode)

        if log_time:
            Logger._START_TIME = datetime.now()
            Logger.log(f'started at {Logger._START_TIME}')

    @staticmethod
    def close():
        if Logger._START_TIME is not None:
            Logger.log('script ended')
            Logger.log(f'elapsed {datetime.now() - Logger._START_TIME}')
        Logger.FILE.close()
        Logger.FILE = None
        Logger._START_TIME = None

    @staticmethod
    def log(message='', end='\n'):
        if Logger.FILE is not None:
            print(message, end=end)
            sys.stdout.flush()

            Logger.FILE.write(f'{message}{end}')
            Logger.FILE.flush()
