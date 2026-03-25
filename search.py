from bsbi import BSBIIndex
from compression import VBEPostings
from compression import OptPForDeltaPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')
BSBI_instance2 = BSBIIndex(
    data_dir="collection",
    postings_encoding=OptPForDeltaPostings,
    output_dir="index2"
)
queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]
           
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
        print(f"{doc:30} {score:>.3f}")
    print()
    for (score, doc) in BSBI_instance2.retrieve_tfidf(query, k = 10):
        print(f"{doc:30} {score:>.3f}")
    print()