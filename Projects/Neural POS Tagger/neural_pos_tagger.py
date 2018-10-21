# PyTorch Implementation of a Neural POS Tagger

#### Libraries ####
import argparse
import random
import sys
import numpy as np
#
import torch
torch.manual_seed(0)

#### Variables & Constants & Paths ####
PAD = "__PAD__"
UNK = "__UNK__"
DIM_EMBEDDING = 100
LSTM_HIDDEN = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.015
LEARNING_DECAY_RATE = 0.05
EPOCHS = 100
KEEP_PROB = 0.5
WEIGHT_DECAY = 1e-8

GLOVE = './DATA/glove.6B.100d.txt'

#### Functions ####

# Data Reader
def read_data(filename):
    """ Example Input :
    Pierre|NNP Vinken|NNP ,|, 61|CD years|NNS old|JJ
    """
    content =[]
    with open(filename) as data_src:
        for line in data_src:
            t_p = [w.split('|') for w in line.strip().split()]
            tokens = [v[0] for v in t_p]
            tags = [v[1] for v in t_p]
            content.append((tokens, tags))
    return content


