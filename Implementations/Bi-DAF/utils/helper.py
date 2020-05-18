import torch
import torch.nn as nn
import spacy
from typing import List

nlp = spacy.load('en_core_web_sm')


def model_train(model, model_optimizer, dataset):
    model.train()
    epoch_loss = 0
    loss = 0
    for item in dataset:
        model_optimizer.zero_grad()
        item_context, item_query, item_positions = item
        pred_p1_pdist, pred_p2_pdist = model(context=item_context, query=item_query)
        loss -= torch.sum(torch.log(torch.stack([pred_p1_pdist[item_positions[0]], pred_p2_pdist[item_positions[1]]])))
    loss.backward()
    model_optimizer.step()
    epoch_loss += loss.item() / len(dataset)
    return model, model_optimizer, epoch_loss


def model_evaluate(model, dataloader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        loss = 0
        for item in dataloader:
            item_context, item_query, item_positions = item
            pred_p1_pdist, pred_p2_pdist = model(context=item_context, query=item_query)
            loss -= torch.sum(
                torch.log(torch.stack([pred_p1_pdist[item_positions[0]], pred_p2_pdist[item_positions[1]]])))
        epoch_loss += loss.item() / len(dataloader)
    return epoch_loss


def get_best_span(p1_dist_by_sent: torch.Tensor, p2_dist_by_sent: torch.Tensor):
    best_span_sent_ix = 0
    # Add the plus 1 when you fix the data
    best_span_tok_ixs = (0, 0)
    max_value = 0
    for sent_ix, (sent_p1_dist, sent_p2_dist) in enumerate(zip(p1_dist_by_sent, p2_dist_by_sent)):
        best_p1_ix = 0
        for tok_ix in range(len(sent_p1_dist)):
            p1_val = sent_p1_dist[best_p1_ix]
            if p1_val < sent_p1_dist[tok_ix]:
                p1_val = sent_p1_dist[tok_ix]
                best_p1_ix = tok_ix

            p2_val = sent_p2_dist[tok_ix]
            if p1_val * p2_val > max_value:
                max_value = p1_val * p2_val
                best_span_sent_ix = sent_ix
                best_span_tok_ixs = (best_p1_ix, tok_ix)

    # Add the plus 1 when you fix the data
    return (best_span_sent_ix, best_span_tok_ixs[0]), (best_span_sent_ix, best_span_tok_ixs[1])


def mould_into_sents(x: torch.Tensor, sent_lens: List[int]):
    x = x.tolist()
    moulded_x = []
    ref = 0
    for sent_len in sent_lens:
        moulded_x.append(x[ref:sent_len])
        ref+=sent_len
    return moulded_x


def predict(context: str, query: str, model: torch.nn.Module):
    context_doc = nlp(context)
    context_tokens = [[token for token in sent] for sent in context_doc.sents]
    context_sent_lens = [len(sent_tokens) for sent_tokens in context_tokens]

    pred_p1_dist, pred_p2_dist = model(context=context, query=query)
    pred_p1_dist_by_sent = mould_into_sents(pred_p1_dist, context_sent_lens)
    pred_p2_dist_by_sent = mould_into_sents(pred_p2_dist, context_sent_lens)

    ans_span = get_best_span(pred_p1_dist_by_sent, pred_p2_dist_by_sent)
    print(ans_span)
    print(context_tokens)
    p1_token = context_tokens[ans_span[0][0]][ans_span[0][1]]
    p2_token = context_tokens[ans_span[1][0]][ans_span[1][1]]

    highlighted_context = context[:p1_token.idx] + \
                          '<span style="background-color:rgb(135,206,250);">' \
                          + context[p1_token.idx:p2_token.idx] + \
                          '</span>' + \
                          context[p2_token.idx:]
    return highlighted_context
