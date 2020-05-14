# External Libraries
import string
import torch
import torch.nn as nn
import spacy

from typing import List, Union
from allennlp.modules.elmo import Elmo, batch_to_ids

nlp = spacy.load('en_core_web_sm')


# Model Definition
class HighwayNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 2,
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
        # Output = Gated*NonLinearAffineTransformedInput + (1-Gated)*Input

        for layer_ix in range(self.num_layers):
            curr_gate_layer = self.sigmoid(self.gate[layer_ix](highway_tensor))
            curr_plain_layer = self.relu(self.plain[layer_ix](highway_tensor))
            highway_tensor = curr_gate_layer * curr_plain_layer + (1 - curr_gate_layer) * highway_tensor
        return highway_tensor


class BiDAF(nn.Module):
    def __init__(self, elmo_sent: bool = False):
        super(BiDAF, self).__init__()
        # Model Properties
        self.elmo_sent = elmo_sent

        # Useful Constants
        self.UNK_TOK_IX = 0
        self.CHAR_EMBED_DIM = 100
        self.NUM_OF_CHAR_FILTERS = 100
        self.CHAR_FILTER_WIDTH = 5

        self.ELMO_EMBED_DIM = 256  # This will change if the ELMO options/weights change
        self.OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        self.WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

        # Model Helper
        self.CHAR_SET_CNN = ["UNK"] + list(
            " " + string.punctuation + string.ascii_lowercase + string.ascii_uppercase + string.digits)
        self.CHAR_2_IX = {CHAR: IX for IX, CHAR in enumerate(self.CHAR_SET_CNN)}
        self.token_2_char_ixs = lambda token: torch.LongTensor(
            [self.CHAR_2_IX.get(char, self.UNK_TOK_IX) for char in list(token)])

        # Model Layers
        self.char_embedding = nn.Embedding(num_embeddings=len(self.CHAR_SET_CNN),
                                           embedding_dim=self.CHAR_EMBED_DIM)
        self.cnn = nn.Conv1d(in_channels=self.CHAR_EMBED_DIM,
                             out_channels=self.NUM_OF_CHAR_FILTERS,
                             kernel_size=self.CHAR_FILTER_WIDTH,
                             padding=self.CHAR_FILTER_WIDTH // 2,
                             stride=1
                             )

        self.elmo = Elmo(self.OPTIONS_FILE, self.WEIGHTS_FILE, 1)

        self.highway = HighwayNetwork(input_dim=(self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM))

    def _cnn_embed_doc(self, doc_tokens: List[List[str]]) -> torch.Tensor:
        cnn_doc_feats = []
        doc_tokens = [token for sent_tokens in doc_tokens for token in sent_tokens]
        for token in doc_tokens:
            token_char_ixs = self.token_2_char_ixs(token)
            token_char_embedded = self.char_embedding(token_char_ixs)
            token_char_embedded = torch.transpose(token_char_embedded, 0, 1).unsqueeze(dim=0)
            cnn_features = self.cnn(token_char_embedded)
            cnn_features_maxpool, _ = cnn_features.max(dim=2)
            cnn_doc_feats.append(cnn_features_maxpool)
        cnn_doc_feats = torch.cat(cnn_doc_feats, dim=0)
        return cnn_doc_feats

    def _elmo_embed_doc(self, doc_tokens: List[List[str]]) -> torch.Tensor:
        if not self.elmo_sent:
            doc_tokens = [[token for sent_tokens in doc_tokens for token in sent_tokens]]

        print(doc_tokens)
        doc_elmo_ids = batch_to_ids(doc_tokens)
        doc_elmo_embed = self.elmo(doc_elmo_ids)

        if self.elmo_sent:
            _elmo_doc_feats = []
            for sent_elmo_embed, sent_elmo_mask in zip(doc_elmo_embed['elmo_representations'][0],
                                                       doc_elmo_embed['mask']):
                _elmo_doc_feats.append(sent_elmo_embed[:sum(sent_elmo_mask)])
            elmo_doc_feats = torch.cat(_elmo_doc_feats, dim=0)
        else:
            elmo_doc_feats = doc_elmo_embed['elmo_representations'][0][0]
        return elmo_doc_feats

    def _embed_doc(self, doc: str) -> torch.Tensor:
        # Prep Doc
        doc = nlp(doc)
        doc_tokens = [[token.text for token in sent] for sent in doc.sents]
        # Embed the Doc with CNN
        doc_embedded_cnn = self._cnn_embed_doc(doc_tokens)
        # Embed the Doc with Elmo
        doc_embedded_elmo = self._elmo_embed_doc(doc_tokens)
        # Concat the Doc Embeddings
        doc_embedded = torch.cat((doc_embedded_cnn, doc_embedded_elmo), dim=1)
        # Pass Embedding through a Highway Network
        doc_embedded_highway = self.highway(doc_embedded)
        return doc_embedded_highway

    def forward(self, context: str, query: str) -> Union:
        context_embedding = self._embed_doc(doc=context)
        query_embedding = self._embed_doc(doc=query)
        return context_embedding, query_embedding


bidaf = BiDAF()

c = "There once was a dog. His name was Charlie. He was a very good boy."
q = "Who is a good boy?"

c_em, q_em = bidaf(context=c, query=q)

print(c)
print(c_em.shape)
print(q)
print(q_em.shape)

