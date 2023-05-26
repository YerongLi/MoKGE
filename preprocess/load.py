import json
from tqdm import tqdm
import collections
import configparser
import sys

config = configparser.ConfigParser()
config.read("paths.cfg")


def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

read_json(DATA_PATH + '/test/{}hops_{}_directed_triple_filter.json')
