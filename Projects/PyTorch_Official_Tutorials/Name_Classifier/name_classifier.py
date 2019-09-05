# Begin!
import os
import glob

import unicodedata
import string
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from random import shuffle

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
categories = list(data.keys())

data_tuples = [(name, ctgry) for ctgry in categories for name in data[ctgry]]
random.shuffle(data_tuples)

training_tuples = data_tuples[:15000]
validation_tuples = data_tuples[15000:17500]
test_tuples = data_tuples[17500:]

print("Dataset Sizes are : Training({0}), Validation({1}), Test({2}) ".format(len(training_tuples),
                                                                              len(validation_tuples), len(test_tuples)))


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

###### Defining the Network ########


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def forward(self, char_embedding, hidden_state):
        combined_state = torch.cat((char_embedding, hidden_state), dim=1)
        next_hidden_state = self.i2h(combined_state)
        output_state = self.i2o(combined_state)
        output_state = self.LogSoftmax(output_state)
        return output_state, next_hidden_state


#
# sample_input = name_2_tensor('Daniel')
# our_rnn = RNN(input_size=vocab_size, hidden_size=128, output_size=len(categories))
# init_hidden = our_rnn.init_hidden()
#
# curr_out, curr_hidden = our_rnn(sample_input[0], init_hidden)
#
# print("Current Output is {0}".format(curr_out))
# print("Current Hidden is {0}".format(curr_hidden))

def get_random_data_pair():
    random_category = random.choice(categories)
    random_name = random.choice(data[random_category])
    random_name_tensor = name_2_tensor(random_name)
    random_category_tensor = torch.tensor([categories.index(random_category)], dtype=torch.long)
    return random_name, random_name_tensor, random_category, random_category_tensor


# for i in range(10):
#     curr_name, curr_name_tensor, curr_category, curr_category_tensor = get_random_data_pair()
#     print("Name : {0} \t Category : {1}".format(curr_name, curr_category))


char_rnn = RNN(input_size=vocab_size, hidden_size=128, output_size=len(categories))

learning_rate = 0.005


def train(data_tuple):
    curr_name, curr_ctgry = data_tuple
    curr_name_embedding = name_2_tensor(curr_name)

    curr_hidden = char_rnn.init_hidden()

    char_rnn.zero_grad()
    for curr_char_embedding in curr_name_embedding:
        curr_output, curr_hidden = char_rnn(curr_char_embedding, curr_hidden)
    curr_target = torch.tensor([categories.index(curr_ctgry)], dtype=torch.long)

    loss_func = nn.NLLLoss()

    curr_loss = loss_func(curr_output, curr_target)
    curr_loss.backward()
    #
    #
    for each_param in char_rnn.parameters():
        each_param.data.add_(-learning_rate, each_param.grad.data)

    return curr_loss.item()


def eval(data_tuple):
    curr_name, curr_ctgry = data_tuple
    curr_name_embedding = name_2_tensor(curr_name)

    curr_hidden = char_rnn.init_hidden()

    char_rnn.zero_grad()
    for curr_char_embedding in curr_name_embedding:
        curr_output, curr_hidden = char_rnn(curr_char_embedding, curr_hidden)
    curr_target = torch.tensor([categories.index(curr_ctgry)], dtype=torch.long)

    loss_func = nn.NLLLoss()

    curr_loss = loss_func(curr_output, curr_target)

    return curr_loss.item()


train(('Giannis', 'Greek'))

NUM_EPOCHS = 30

collected_train_loss = []
collected_valid_loss = []

for epoch in range(NUM_EPOCHS):
    epoch_train_loss = 0
    epoch_valid_loss = 0
    shuffle(training_tuples)
    for single_training_tuple in training_tuples:
        curr_train_loss = train(single_training_tuple)
        epoch_train_loss += curr_train_loss

    with torch.no_grad():
        for single_valid_tuple in validation_tuples:
            curr_valid_loss = eval(single_valid_tuple)
            epoch_valid_loss += curr_valid_loss

    epoch_train_loss = epoch_train_loss / len(training_tuples)
    epoch_valid_loss = epoch_valid_loss / len(validation_tuples)

    collected_train_loss.append(epoch_train_loss)
    collected_valid_loss.append(epoch_valid_loss)
    print("Epoch {0} \t Training Loss : {1} \t Valid Loss : {2}".format(epoch, epoch_train_loss, epoch_valid_loss))

plt.figure()
plt.plot(collected_train_loss)
plt.plot(collected_valid_loss)
plt.savefig('loss_graph.png')

confusion = torch.zeros(len(categories), len(categories))

def test_evaluate(curr_name):
    curr_name_embedding = name_2_tensor(curr_name)

    curr_hidden = char_rnn.init_hidden()

    char_rnn.zero_grad()
    for curr_char_embedding in curr_name_embedding:
        curr_output, curr_hidden = char_rnn(curr_char_embedding, curr_hidden)

    curr_pred_value, curr_pred_ix = curr_output.topk(1)
    return curr_pred_ix


with torch.no_grad():
    for single_test_tuple in test_tuples:
        curr_name, curr_ctgry = single_test_tuple
        pred_ix = test_evaluate(curr_name)
        confusion[categories.index(curr_ctgry), pred_ix] += 1

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + categories, rotation=90)
ax.set_yticklabels([''] + categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.savefig('test_confusion.png')
