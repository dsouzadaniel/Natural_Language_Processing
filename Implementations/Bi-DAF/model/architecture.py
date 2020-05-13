# External Libraries
import string
import torch
import torch.nn as nn
import spacy


# from typing import List
# from allennlp.modules.elmo import Elmo, batch_to_ids
#
# nlp = spacy.load('en_core_web_sm')


# Model Definition
class HighwayNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 ):
        super(HighwayNetwork, self).__init__()
        # Model Properties
        self.input_dim = input_dim
        self.num_layers = num_layers

        # Model Layers
        self.plain = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(self.num_layers)])
        self.gate = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(self.num_layers)])
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, highway_tensor: torch.Tensor) -> torch.Tensor:
        # Highway Networks provide a gated mechanism between a
        # non-linear affine transformation of your input and a
        # original version of your input (https://arxiv.org/pdf/1505.00387.pdf)
        # Output = Gated*NonLinearAffineTranformedInput + (1-Gated)*Input

        for layer_ix in range(self.num_layers):
            curr_gate_layer = self.sigmoid(self.gate[layer_ix](highway_tensor))
            curr_plain_layer = self.relu(self.plain[layer_ix](highway_tensor))
            highway_tensor = curr_gate_layer * curr_plain_layer + (1 - curr_gate_layer) * highway_tensor
        return highway_tensor

#
# hn = HighwayNetwork(input_dim=100, num_layers=2)
#
# input_tensor = torch.randn(32, 100)
# print("Input Tensor Shape is {0}".format(input_tensor.shape))
# output_tensor = hn(input_tensor)
# print("Output Tensor Shape is {0}".format(output_tensor.shape))
#
#
#
#
#
# class BiDAF(nn.Module):
#     def __init__(self,
#                  input_dim: int):
#         super(BiDAF, self).__init__()
#         # Model Properties
#         self.input_dim = input_dim
#
#         # Useful Constants
#         self.CHAR_SET_CNN = list(string.whitespace+string.punctuation+string.ascii_lowercase + string.ascii_uppercase + string.digits)
#         self.CNN_FILTERS = 100
#         self.CNN_WIDTH = 5
#         self.OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
#         self.WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
#
#         # Model Layers
#         self.cnn = nn.Conv1d(in_channels=self.input_dim,
#                              out_channels=self.CNN_FILTERS,
#                              kernel_size=self.CNN_WIDTH,
#                              padding=self.CNN_WIDTH // 2,
#                              stride=1
#                              )
#
#         self.elmo_embed = Elmo(self.OPTIONS_FILE, self.WEIGHTS_FILE, 1)
#
#     def forward(self, input_text_list: List) -> torch.Tensor:
#         for input_text in input_text_list:
#             input_text_doc = nlp(input_text)
#             sentences = [[token.text for token in input_text_doc]]
#             # Embed the Words with Elmo
#             character_ids = batch_to_ids(sentences)
#             sentences_embedded = self.elmo_embed(character_ids)['elmo_representations'][0]
#             print(sentences_embedded.shape)
#             print(self.CHAR_SET_CNN)
#             print(len(self.CHAR_SET_CNN))
#         pass
#         # output_tensor = self.cnn(input_tensor)
#         # max_output, _ = output_tensor.max(dim=2)
#         # max_output = max_output.squeeze()
#         # return max_output
#
#
# bidaf = BiDAF(input_dim=1024)
# # input = torch.randn(1, 1024, 20)
# # print(input.shape)
# # output = bidaf(input)
# # print(output.shape)
#
# input_to_send = ['Hello this is the best']
# bidaf(input_to_send)
