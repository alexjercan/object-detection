import os
import glob
import math
import logging
from utils.constants import NO_CHANNELS


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def count_channles(layers):
    return sum([NO_CHANNELS[chan] for chan in layers])


def check_file(file):
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)
        assert len(files), 'File Not Found: %s' % file
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)
        return files[0]
    

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)
