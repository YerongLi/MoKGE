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

dataset = 'eg'
DATA_PATH = config["paths"][dataset + "_dir"]
T = 2 
max_B = 100


triple=read_json(DATA_PATH + '/test/{}hops_{}_directed_triple_filter.json'.format(T, max_B))
print(dir(triple))