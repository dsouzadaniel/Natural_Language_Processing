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
#
# train_data = read_data('./DATA/train_lines.txt')
# print("Length of Train Data is : ", len(train_data))
#
# test_data = read_data('./DATA/test_lines.txt')
# print("Length of Test Data is : ", len(test_data))

def simplify_token(token):
    chars = []
    for char in token:
        if char.isdigit():
            chars.append('0')
        else:
            chars.append(char)
    return ''.join(chars)
#
# print(simplify_token(("My IM name is crazy4catz and I have 11 cats at home")))

def read_glove(glove_vector_path):
    pretrained_glove_vectors = {}
    for line in open(glove_vector_path):
        parts = line.strip().split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        pretrained_glove_vectors[word] = vector
    return pretrained_glove_vectors



def main():
    parser = argparse.ArgumentParser(description = 'POS Tagger')
    parser.add_argument('training_data')
    parser.add_argument('testing_data')
    args = parser.parse_args()

    train = read_data(args.training_data)
    test = read_data(args.testing_data)

    id_2_token = [PAD, UNK]
    token_2_id = {PAD:0, UNK:1}

    id_2_tag = [PAD]
    tag_2_id = {PAD:0}

    for tokens, tags in train:
        for token in tokens:
            if not token in token_2_id:
                token_2_id[token] = len(token_2_id)
                id_2_token.append(token)
        for tag in tags:
            if not tag in tag_2_idL
                tag_2_id[tag] = len(tag_2_id)
                id_2_tag.append(tag)

    num_of_words = len(token_2_id)
    num_of_tags = len(tag_2_id)

    # Load up the GloVe Vectors
    pretrained = read_glove(GLOVE)
    pretrained_list = []
    scale = np.sqrt(3.0/ DIM_EMBEDDING)

    for word in id_2_token:
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)

    model = TaggerModel(num_of_words, num_of_tags, pretrained_list, id_2_token)



class TaggerModel(torch.nn.Module):

    def __init__(self, nwords, ntags, pretrained_list, id_2_token):
        super.init()

        # Create Word Embeddings
        pretrained_tensor = torch.FloatTensor(pretrained_list)
        self.word_embedding = torch.nn.Embedding.from_pretrained(pretrained_tensor, freeze = False)

        self.word_dropout = torch.nn.Dropout(1- KEEP_PROB)

        self.lstm = torch.nn.LSTM(DIM_EMBEDDING,
                                  LSTM_HIDDEN,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)

        self.lstm_output_dropout = torch.nn.Dropout(1 - KEEP_PROB)

        self.hidden_to_tag = torch.nn.Linear(LSTM_HIDDEN*2, ntags)

    def forward(self, sentences, labels, lengths, curr_batch_size):
        max_length = sentences.size(1)

        word_vectors = self.word_embedding(sentences)

        dropped_word_vectors = self.word_dropout(word_vectors)

        packed_words = torch.nn.utils.rnn.pack_padded_sequence(dropped_word_vectors, lengths, True)

        lstm_out, _ = self.lstm(packed_words, None)

        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length = max_length)

        lstm_out_dropped = self.lstm_output_dropout(lstm_out)

        output_scores = self.hidden_to_tag(lstm_out_dropped)

        # Calculate Loss
        output_scores = output_scores.view(curr_batch_size*max_length, -1)

        flat_labels = labels.view(curr_batch_size*max_length)

        loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction = 'sum')

        loss = loss_function(output_scores, flat_labels)

        predicted_tags  = predicted_tags.view(curr_batch_size, max_length)

        return loss, predicted_tags








