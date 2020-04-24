# Begin!
import os
import glob
import torch
import torch.nn as nn
import random
import unicodedata
import string

DATA_GLOB_PATH = os.path.join('data', 'names', '*.txt')

all_letters = [c for c in string.ascii_letters + " .,:'"] + ['<EOS>']
vocab_size = len(all_letters)


## Helper Functions

# Convert Unicode to Ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read Data from a FilePath
def readData(some_filepath):
    language = os.path.basename(some_filepath).split('.')[0]
    names = []
    with open(some_filepath, 'r', encoding='utf-8') as f:
        names.extend([unicodeToAscii(line.strip()) for line in f.readlines()])
    return language, names


def char_2_ix(some_character):
    return all_letters.index(some_character)


def name_2_tensor(some_name):
    name_tensor = torch.zeros(len(some_name), 1, vocab_size)
    for ix, charac in enumerate(some_name):
        name_tensor[ix][0][char_2_ix(charac)] = 1
    return name_tensor


#
# readData('data/names/english.txt')
# print(name_2_tensor('Daniel'))

# Reading Data into a Dictionary
data = {k: v for k, v in [readData(curr_filepath) for curr_filepath in glob.glob(DATA_GLOB_PATH)]}
categories = list(data.keys())

data_tuples = [(name, category) for category in categories for name in data[category]]
random.shuffle(data_tuples)

training_tuples = [([list(name), list(name)[1:] + ['<EOS>'], category]) for name, category in data_tuples]
print("Length of Dataset is {0}".format(len(training_tuples)))

# print(training_tuples[:5])

class RNN(nn.Module):
    def __init__(self, category_size, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # Layers
        self.i2o = nn.Linear(category_size + input_size + hidden_size, output_size)
        self.i2h = nn.Linear(category_size + input_size + hidden_size, hidden_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)

        # Transformations
        self.Dropout = nn.Dropout(p=0.1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def forward(self, onehot_category, char_embedding, hidden_state):
        initial_combined = torch.cat((onehot_category, char_embedding, hidden_state), dim=1)
        initial_output_state = self.i2o(initial_combined)
        next_hidden_state = self.i2h(initial_combined)
        output_combined = torch.cat((initial_output_state, next_hidden_state), dim=1)
        final_output_state = self.o2o(output_combined)
        droppedout_final_output_state = self.Dropout(final_output_state)
        softmaxd_output = self.LogSoftmax(droppedout_final_output_state)
        return softmaxd_output, next_hidden_state


loss_func = nn.NLLLoss()
learning_rate = 0.005

char_rnn = RNN(len(categories), len(all_letters), 128, len(all_letters))
# highest_till_now = 0
for epoch in range(50):
    epoch_loss = 0



    for orig_chars, shift_chars, name_category in training_tuples:
        char_rnn.zero_grad()
        hidden_state = char_rnn.init_hidden()

        orig_chars_embedding = name_2_tensor(orig_chars)

        collected_loss = 0
        collected = []
        # print("*", ''.join(orig_chars))
        for input_char_embedding, output_char in zip(orig_chars_embedding, shift_chars):
            output_char_ix = torch.tensor([char_2_ix(output_char)])
            category_1hot = torch.zeros(1, len(categories)).scatter(1,
                                                                    torch.tensor([[categories.index(name_category)]]),
                                                                    1)
            # print(input_char_embedding.shape)
            # print(output_char_ix.shape)
            # print(category_1hot.shape)
            # print(hidden_state.shape)
            # print("\n")
            predicted_char_embedding, hidden_state = char_rnn(category_1hot, input_char_embedding, hidden_state)
            curr_loss = loss_func(predicted_char_embedding, output_char_ix)

            # print("*",curr_loss.item())
            collected_loss += curr_loss
            # input_char = all_letters[torch.argmax(input_char_embedding, dim=1).item()]
            # collected.append((input_char, output_char, curr_loss.item()))
        # print(''.join(orig_chars))
        # print(collected)
        # print("*",collected_loss.item(),epoch_loss)
        # print("\n")
        collected_loss = collected_loss/len(shift_chars)
        collected_loss.backward()

        for each_param in char_rnn.parameters():
            each_param.data.add_(-learning_rate, each_param.grad.data)

        epoch_loss+=collected_loss.item()
        # if collected_loss.item()>highest_till_now:
        #     highest_till_now = collected_loss.item()
        # print("\n")

    #     print(output_char, collected_loss.item())
    #

#
    print("Epoch {0}:\t Loss {1}".format(epoch, epoch_loss))
# print("Highest Loss is : {0}".format(highest_till_now))
#

    #
    #     epoch_loss += collected_loss.item()
    # print("Epoch {0}:\t Loss {1}".format(epoch, epoch_loss))
# #
# a = torch.tensor([[1,2,3,4],[33,44,55,66]])
# b = torch.tensor([[6,7,8,9]])
# print(a.shape)
# print(b.shape)
#
# print(torch.cat([a,b], dim=0))
#
# print(torch.zeros(1,5))
# print(torch.zeros(5))
#
# c = 10
# a = 3
#
# # print(torch.zeros(1,c))
# print(torch.zeros(1, c).scatter(1, torch.tensor([[a]]), 1))
