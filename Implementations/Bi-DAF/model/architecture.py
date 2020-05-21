# External Libraries
import string
import torch
import torch.nn as nn
import spacy

from typing import List, Union
from allennlp.modules.elmo import Elmo, batch_to_ids

nlp = spacy.load('en_core_web_sm')


import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

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
        self.randomize_init_hidden = True

        # Useful Constants
        self.UNK_TOK_IX = 0
        self.CHAR_EMBED_DIM = 100
        self.NUM_OF_CHAR_FILTERS = 100
        self.CHAR_FILTER_WIDTH = 5

        self.ELMO_EMBED_DIM = 256  # This will change if the ELMO options/weights change
        self.OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        self.WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

        self.CONTEXTUAL_LSTM_LAYERS = 1
        self.MODELLING_LSTM_LAYERS = 2
        self.POSITIONAL_LSTM_LAYERS = 1
        self.randomize_init_hidden = True

        # Model Helper
        self.CHAR_SET_CNN = ["UNK"] + list(
            " " + string.punctuation + string.ascii_lowercase + string.ascii_uppercase + string.digits)
        self.CHAR_2_IX = {CHAR: IX for IX, CHAR in enumerate(self.CHAR_SET_CNN)}
        self.token_2_char_ixs = lambda token: torch.LongTensor(
            [self.CHAR_2_IX.get(char, self.UNK_TOK_IX) for char in list(token)])

        # Model Layers
        self.softmax_dim0 = nn.Softmax(dim=0)
        self.softmax_dim1 = nn.Softmax(dim=1)

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

        self.contextual_lstm = nn.LSTM(input_size=self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM,
                                       hidden_size=self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM,
                                       bidirectional=True,
                                       num_layers=self.CONTEXTUAL_LSTM_LAYERS)

        self.similarity_alpha = nn.Linear(6 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM), 1, bias=False)

        self.modelling_lstm = nn.LSTM(input_size=8 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM),
                                      hidden_size=self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM,
                                      bidirectional=True,
                                      num_layers=self.MODELLING_LSTM_LAYERS)

        self.positioning_lstm = nn.LSTM(input_size=2 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM),
                                        hidden_size=self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM,
                                        bidirectional=True,
                                        num_layers=self.POSITIONAL_LSTM_LAYERS)

        self.pos1_linear = nn.Linear(10 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM), 1, bias=False)
        self.pos2_linear = nn.Linear(10 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM), 1, bias=False)

    def _init_bi_hidden(self, batch_size: int = 1, num_layers: int = 1):
        if self.randomize_init_hidden:
            init_hidden = torch.randn(num_layers * 2, batch_size,
                                      self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM)
        else:
            init_hidden = torch.zeros(num_layers * 2, batch_size,
                                      self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM)
        return init_hidden, init_hidden

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

        # print(doc_tokens)
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
        # Pass Embedding through the LSTM
        doc_embedded_highway = doc_embedded_highway.unsqueeze(dim=1)
        doc_embedded_contextual = self._run_through_bilstm(input_tensor=doc_embedded_highway,
                                                           bilstm=self.contextual_lstm)

        return doc_embedded_contextual

    def _run_through_bilstm(self, input_tensor: torch.Tensor, bilstm: torch.nn.modules.rnn):
        init_bi_hidden = self._init_bi_hidden(num_layers=bilstm.num_layers)
        output_tensor, _ = bilstm(input_tensor, init_bi_hidden)
        output_tensor = output_tensor.view(input_tensor.shape[0], 1, 2, bilstm.hidden_size)
        output_tensor = torch.cat((output_tensor[:, :, 0, :],
                                   output_tensor[:, :, 1, :]),
                                  dim=2).squeeze(dim=1)
        return output_tensor

    def forward(self, context: str, query: str) -> Union:
        context_embedding = self._embed_doc(doc=context)
        query_embedding = self._embed_doc(doc=query)

        # Similarity Matx
        broadcast_context_embedding = context_embedding.repeat_interleave(repeats=query_embedding.shape[0], dim=0)
        broadcast_query_embedding = query_embedding.repeat([context_embedding.shape[0], 1])
        broadcast_hadamard_product = broadcast_query_embedding * broadcast_context_embedding
        broadcast = torch.cat([broadcast_context_embedding, broadcast_query_embedding, broadcast_hadamard_product],
                              dim=1)
        similarity_tensor = self.similarity_alpha(broadcast).squeeze()
        similarity_matx = torch.reshape(similarity_tensor, (context_embedding.shape[0], query_embedding.shape[0]))

        # Context_2_Query ( C2Q )
        c2q_attn = self.softmax_dim1(similarity_matx)
        c2q = torch.matmul(c2q_attn, query_embedding)

        # Query_2_Context ( Q2C )
        q2c_max, _ = similarity_matx.max(dim=1)
        q2c_attn = self.softmax_dim0(q2c_max)
        q2c = torch.matmul(q2c_attn, context_embedding).repeat(context_embedding.shape[0], 1)

        # Query Aware Context Representation
        context_c2q_hadamard_product = context_embedding * c2q
        context_q2c_hadamard_product = context_embedding * q2c
        query_aware_context = torch.cat(
            [context_embedding, c2q, context_c2q_hadamard_product, context_q2c_hadamard_product], dim=1)

        # Model Query Aware Context
        query_aware_context = query_aware_context.unsqueeze(dim=1)
        modelled_query_aware_context = self._run_through_bilstm(input_tensor=query_aware_context,
                                                                bilstm=self.modelling_lstm)

        # Sub-Phrase Positions P1
        query_aware_context = query_aware_context.squeeze(dim=1)
        pos1_context = torch.cat([query_aware_context, modelled_query_aware_context], dim=1)
        pos1_pdist = self.softmax_dim0(self.pos1_linear(pos1_context)).squeeze()

        # Sub-Phrase Positions P2
        modelled_query_aware_context = modelled_query_aware_context.unsqueeze(dim=1)
        modelled2_query_aware_context = self._run_through_bilstm(input_tensor=modelled_query_aware_context,
                                                                 bilstm=self.positioning_lstm)
        modelled2_query_aware_context = modelled2_query_aware_context.squeeze(dim=1)
        pos2_context = torch.cat([query_aware_context, modelled2_query_aware_context], dim=1)
        pos2_pdist = self.softmax_dim0(self.pos2_linear(pos2_context)).squeeze()

        return pos1_pdist, pos2_pdist


class HighwayNetwork_GPU(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 2,
                 ):
        super(HighwayNetwork_GPU, self).__init__()
        # Model Properties
        self.input_dim = input_dim
        self.num_layers = num_layers

        # Model Layers
        self.plain = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(self.num_layers)]).to(device)
        self.gate = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(self.num_layers)]).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, highway_tensor: torch.Tensor) -> torch.Tensor:
        # Highway Networks provide a gated mechanism between a
        # non-linear affine transformation of your input and a
        # original version of your input (https://arxiv.org/pdf/1505.00387.pdf)
        # Output = Gated*NonLinearAffineTransformedInput + (1-Gated)*Input

        for layer_ix in range(self.num_layers):
            curr_gate_layer = self.sigmoid(self.gate[layer_ix](highway_tensor)).to(device)
            curr_plain_layer = self.relu(self.plain[layer_ix](highway_tensor)).to(device)
            highway_tensor = curr_gate_layer * curr_plain_layer + (1 - curr_gate_layer) * highway_tensor
        return highway_tensor


class BiDAF_GPU(nn.Module):
    def __init__(self, elmo_sent: bool = False):
        super(BiDAF_GPU, self).__init__()
        # Model Properties
        self.elmo_sent = elmo_sent
        self.randomize_init_hidden = True

        # Useful Constants
        self.UNK_TOK_IX = 0
        self.CHAR_EMBED_DIM = 100
        self.NUM_OF_CHAR_FILTERS = 100
        self.CHAR_FILTER_WIDTH = 5

        self.ELMO_EMBED_DIM = 256  # This will change if the ELMO options/weights change
        self.OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        self.WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

        self.CONTEXTUAL_LSTM_LAYERS = 1
        self.MODELLING_LSTM_LAYERS = 2
        self.POSITIONAL_LSTM_LAYERS = 1
        self.randomize_init_hidden = True

        # Model Helper
        self.CHAR_SET_CNN = ["UNK"] + list(
            " " + string.punctuation + string.ascii_lowercase + string.ascii_uppercase + string.digits)
        self.CHAR_2_IX = {CHAR: IX for IX, CHAR in enumerate(self.CHAR_SET_CNN)}
        self.token_2_char_ixs = lambda token: torch.LongTensor(
            [self.CHAR_2_IX.get(char, self.UNK_TOK_IX) for char in list(token)])

        # Model Layers
        self.softmax_dim0 = nn.Softmax(dim=0).to(device)
        self.softmax_dim1 = nn.Softmax(dim=1).to(device)

        self.char_embedding = nn.Embedding(num_embeddings=len(self.CHAR_SET_CNN),
                                           embedding_dim=self.CHAR_EMBED_DIM).to(device)
        self.cnn = nn.Conv1d(in_channels=self.CHAR_EMBED_DIM,
                             out_channels=self.NUM_OF_CHAR_FILTERS,
                             kernel_size=self.CHAR_FILTER_WIDTH,
                             padding=self.CHAR_FILTER_WIDTH // 2,
                             stride=1
                             ).to(device)

        self.elmo = Elmo(self.OPTIONS_FILE, self.WEIGHTS_FILE, 1).cuda()

        self.highway = HighwayNetwork_GPU(input_dim=(self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM))

        self.contextual_lstm = nn.LSTM(input_size=self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM,
                                       hidden_size=self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM,
                                       bidirectional=True,
                                       num_layers=self.CONTEXTUAL_LSTM_LAYERS).to(device)

        self.similarity_alpha = nn.Linear(6 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM), 1, bias=False).to(device)

        self.modelling_lstm = nn.LSTM(input_size=8 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM),
                                      hidden_size=self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM,
                                      bidirectional=True,
                                      num_layers=self.MODELLING_LSTM_LAYERS).to(device)

        self.positioning_lstm = nn.LSTM(input_size=2 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM),
                                        hidden_size=self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM,
                                        bidirectional=True,
                                        num_layers=self.POSITIONAL_LSTM_LAYERS).to(device)

        self.pos1_linear = nn.Linear(10 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM), 1, bias=False).to(device)
        self.pos2_linear = nn.Linear(10 * (self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM), 1, bias=False).to(device)

    def _init_bi_hidden(self, batch_size: int = 1, num_layers: int = 1):
        if self.randomize_init_hidden:
            init_hidden = torch.randn(num_layers * 2, batch_size,
                                      self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM).to(device)
        else:
            init_hidden = torch.zeros(num_layers * 2, batch_size,
                                      self.CHAR_EMBED_DIM + self.ELMO_EMBED_DIM).to(device)
        return init_hidden, init_hidden

    def _cnn_embed_doc(self, doc_tokens: List[List[str]]) -> torch.Tensor:
        cnn_doc_feats = []
        doc_tokens = [token for sent_tokens in doc_tokens for token in sent_tokens]
        for token in doc_tokens:
            token_char_ixs = self.token_2_char_ixs(token).to(device)
            token_char_embedded = self.char_embedding(token_char_ixs).to(device)
            token_char_embedded = torch.transpose(token_char_embedded, 0, 1).unsqueeze(dim=0).to(device)
            cnn_features = self.cnn(token_char_embedded)
            cnn_features_maxpool, _ = cnn_features.max(dim=2)
            cnn_doc_feats.append(cnn_features_maxpool)
        cnn_doc_feats = torch.cat(cnn_doc_feats, dim=0)
        return cnn_doc_feats

    def _elmo_embed_doc(self, doc_tokens: List[List[str]]) -> torch.Tensor:
        if not self.elmo_sent:
            doc_tokens = [[token for sent_tokens in doc_tokens for token in sent_tokens]]

        # print(doc_tokens)
        doc_elmo_ids = batch_to_ids(doc_tokens).to(device)
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
        doc_embedded_cnn = self._cnn_embed_doc(doc_tokens).to(device)
        # Embed the Doc with Elmo
        doc_embedded_elmo = self._elmo_embed_doc(doc_tokens).to(device)
        # Concat the Doc Embeddings
        doc_embedded = torch.cat((doc_embedded_cnn, doc_embedded_elmo), dim=1)
        # Pass Embedding through a Highway Network
        doc_embedded_highway = self.highway(doc_embedded)
        # Pass Embedding through the LSTM
        doc_embedded_highway = doc_embedded_highway.unsqueeze(dim=1).to(device)
        doc_embedded_contextual = self._run_through_bilstm(input_tensor=doc_embedded_highway,
                                                           bilstm=self.contextual_lstm)

        return doc_embedded_contextual

    def _run_through_bilstm(self, input_tensor: torch.Tensor, bilstm: torch.nn.modules.rnn):
        init_bi_hidden = self._init_bi_hidden(num_layers=bilstm.num_layers)
        output_tensor, _ = bilstm(input_tensor, init_bi_hidden)
        output_tensor = output_tensor.view(input_tensor.shape[0], 1, 2, bilstm.hidden_size)
        output_tensor = torch.cat((output_tensor[:, :, 0, :],
                                   output_tensor[:, :, 1, :]),
                                  dim=2).squeeze(dim=1)
        return output_tensor

    def forward(self, context: str, query: str) -> Union:
        context_embedding = self._embed_doc(doc=context).to(device)
        query_embedding = self._embed_doc(doc=query).to(device)
        # print(query)
        # print("CONTEXT : {0}".format(context_embedding.shape))
        # print("QUERY : {0}".format(query_embedding.shape))

        # Similarity Matx
        broadcast_context_embedding = context_embedding.repeat_interleave(repeats=query_embedding.shape[0], dim=0).to(device)
        broadcast_query_embedding = query_embedding.repeat([context_embedding.shape[0], 1]).to(device)
        broadcast_hadamard_product = broadcast_query_embedding * broadcast_context_embedding
        broadcast = torch.cat([broadcast_context_embedding, broadcast_query_embedding, broadcast_hadamard_product],
                              dim=1).to(device)
        similarity_tensor = self.similarity_alpha(broadcast).squeeze().to(device)
        similarity_matx = torch.reshape(similarity_tensor, (context_embedding.shape[0], query_embedding.shape[0])).to(device)

        # Context_2_Query ( C2Q )
        c2q_attn = self.softmax_dim1(similarity_matx).to(device)
        c2q = torch.matmul(c2q_attn, query_embedding).to(device)

        # Query_2_Context ( Q2C )
        q2c_max, _ = similarity_matx.max(dim=1)
        q2c_attn = self.softmax_dim0(q2c_max).to(device)
        q2c = torch.matmul(q2c_attn, context_embedding).repeat(context_embedding.shape[0], 1).to(device)

        # Query Aware Context Representation
        context_c2q_hadamard_product = context_embedding * c2q
        context_q2c_hadamard_product = context_embedding * q2c
        query_aware_context = torch.cat(
            [context_embedding, c2q, context_c2q_hadamard_product, context_q2c_hadamard_product], dim=1)

        # Model Query Aware Context
        query_aware_context = query_aware_context.unsqueeze(dim=1).to(device)
        modelled_query_aware_context = self._run_through_bilstm(input_tensor=query_aware_context,
                                                                bilstm=self.modelling_lstm)

        # Sub-Phrase Positions P1
        query_aware_context = query_aware_context.squeeze(dim=1).to(device)
        pos1_context = torch.cat([query_aware_context, modelled_query_aware_context], dim=1)
        pos1_pdist = self.softmax_dim0(self.pos1_linear(pos1_context)).squeeze()

        # Sub-Phrase Positions P2
        modelled_query_aware_context = modelled_query_aware_context.unsqueeze(dim=1).to(device)
        modelled2_query_aware_context = self._run_through_bilstm(input_tensor=modelled_query_aware_context,
                                                                 bilstm=self.positioning_lstm)
        modelled2_query_aware_context = modelled2_query_aware_context.squeeze(dim=1).to(device)
        pos2_context = torch.cat([query_aware_context, modelled2_query_aware_context], dim=1).to(device)
        pos2_pdist = self.softmax_dim0(self.pos2_linear(pos2_context)).squeeze()

        return pos1_pdist, pos2_pdist
