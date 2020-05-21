import spacy
nlp = spacy.load('en_core_web_sm')

x = "Hi this the dog. He is the best boy. Come say hi to him"
context_doc = nlp(x)
context_tokens = [[token for token in sent] for sent in context_doc.sents]

print(context_tokens)

a = context_tokens[1][3]