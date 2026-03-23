import pickle

with open("index/main_index.dict", "rb") as f:
    postings_dict, terms, doc_length = pickle.load(f)

# print(postings_dict)
print(len(terms))
print(terms[:10])
print(list(doc_length.items())[:10])