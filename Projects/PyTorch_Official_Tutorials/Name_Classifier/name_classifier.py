# Begin!
import os
import glob

import unicodedata
import string
import torch

# Paths
DATA_GLOB_PATH = os.path.join('data', 'names', '*.txt')

all_letters = string.ascii_letters + " .,;'"
vocab_size = len(all_letters)


# Helper Function to convert Unicode to ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Helper Function to read data from some_filepath
def readData(some_filepath):
    language = os.path.basename(some_filepath).split('.')[0]
    names = []
    with open(some_filepath, 'r', encoding='utf-8') as f:
        names.extend([unicodeToAscii(line.strip()) for line in f.readlines()])
    return language, names


def char_2_ix(some_character):
    return all_letters.find(some_character)


def name_2_tensor(some_name):
    name_tensor = torch.zeros(len(some_name), 1, vocab_size)
    # print(name_tensor.size())
    for ix, charac in enumerate(some_name):
        name_tensor[ix][0][char_2_ix(charac)] = 1
    return name_tensor


# Reading the Data into a Dictionary
data = {k: v for (k, v) in [readData(curr_file_path) for curr_file_path in glob.glob(DATA_GLOB_PATH)]}

# # Print Some of the Data
# print(data.keys())
# print(data['Arabic'][:5])
# print(data['English'][:5])
# print(data['Chinese'][:5])

# example = 'Daniel'
# for ix, char in enumerate(example):
#     print(char, "->", char_2_ix(char))
#
# print(name_2_tensor('Daniel'))


