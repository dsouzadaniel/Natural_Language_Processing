# External Libraries
import string
import torch
import torch.nn as nn
import spacy
from typing import List
from allennlp.modules.elmo import Elmo, batch_to_ids

nlp = spacy.load('en_core_web_sm')


# Model Definition
class BiDAF(nn.Module):
    def __init__(self,
                 input_dim: int):
        super(BiDAF, self).__init__()
        # Model Properties
        self.input_dim = input_dim

        # Useful Constants
        self.CHAR_SET_CNN = list(string.whitespace+string.punctuation+string.ascii_lowercase + string.ascii_uppercase + string.digits)
        self.CNN_FILTERS = 100
        self.CNN_WIDTH = 5
        self.OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        self.WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

        # Model Layers
        self.cnn = nn.Conv1d(in_channels=self.input_dim,
                             out_channels=self.CNN_FILTERS,
                             kernel_size=self.CNN_WIDTH,
                             padding=self.CNN_WIDTH // 2,
                             stride=1
                             )

        self.elmo_embed = Elmo(self.OPTIONS_FILE, self.WEIGHTS_FILE, 1)

    def forward(self, input_text_list: List) -> torch.Tensor:
        for input_text in input_text_list:
            input_text_doc = nlp(input_text)
            sentences = [[token.text for token in input_text_doc]]
            # Embed the Words with Elmo
            character_ids = batch_to_ids(sentences)
            sentences_embedded = self.elmo_embed(character_ids)['elmo_representations'][0]
            print(sentences_embedded.shape)
            print(self.CHAR_SET_CNN)
            print(len(self.CHAR_SET_CNN))
        pass
        # output_tensor = self.cnn(input_tensor)
        # max_output, _ = output_tensor.max(dim=2)
        # max_output = max_output.squeeze()
        # return max_output


bidaf = BiDAF(input_dim=1024)
# input = torch.randn(1, 1024, 20)
# print(input.shape)
# output = bidaf(input)
# print(output.shape)

input_to_send = ['Hello this is the best']
bidaf(input_to_send)
