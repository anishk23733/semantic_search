from caption import generate_text_embedding, generate_embedding_text
import numpy as np
import nltk.data
from scipy import spatial
# import torch
from itertools import islice

filename = input(">> Enter file to search >: ")

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
s = open(filename, 'r') # specify the file we are searching through

search_text = s.read()
sentences = [s.replace('\n', ' ').replace('\t', ' ').strip() for s in tokenizer.tokenize(search_text)]

print(">> Building search cache...")
# embeddings = {sentence: torch.flatten(generate_text_embedding(sentence)) for sentence in sentences if len(sentence) < 70}
embeddings = {sentence: generate_text_embedding(sentence).cpu().detach().numpy() for sentence in sentences if len(sentence) < 70}
print(">> Completed.\n")

# cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

def search(query):
    # query_embedding = torch.flatten(generate_text_embedding(query))
    query_embedding = generate_text_embedding(query).cpu().detach().numpy()
    # get_dist = lambda sentence: cos(query_embedding, embeddings[sentence])
    get_dist = lambda sentence: spatial.distance.cosine(query_embedding.flatten(), embeddings[sentence].flatten())
    # get_dist = lambda sentence: np.linalg.norm(query_embedding - embeddings[sentence])

    semantic_similarity = list(islice(sorted(embeddings, key=get_dist), 5))
    print(f">> Search query: {query}")
    print('> ' + "\n> ".join(semantic_similarity))
    print("\n")

while True:
    query = input(">: ")
    if query == 'q':
        break
    print()
    search(query)