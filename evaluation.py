import math
import re
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

def dcg(ranking):
  """
  DCG@K untuk ranking biner.
  ranking[i] = 1 jika dokumen pada rank i+1 relevan, 0 jika tidak.
  """
  score = 0.0
  for i in range(1, len(ranking) + 1):
    rel_i = ranking[i - 1]
    score += rel_i / math.log2(i + 1)
  return score

def ndcg(ranking, n_relevant):
  """
  NDCG@K = DCG@K / IDCG@K
  n_relevant = jumlah total dokumen relevan untuk query ini di koleksi
  """
  actual_dcg = dcg(ranking)

  ideal_len = len(ranking)
  ideal_ranking = [1] * min(n_relevant, ideal_len) + [0] * max(0, ideal_len - n_relevant)
  ideal_dcg = dcg(ideal_ranking)

  if ideal_dcg == 0:
    return 0.0
  return actual_dcg / ideal_dcg

def ap(ranking, n_relevant):
  """
  Average Precision (AP) untuk ranking biner.
  n_relevant = jumlah total dokumen relevan untuk query ini di koleksi
  """
  if n_relevant == 0:
    return 0.0

  score = 0.0
  relevant_so_far = 0

  for i in range(1, len(ranking) + 1):
    if ranking[i - 1] == 1:
      relevant_so_far += 1
      precision_at_i = relevant_so_far / i
      score += precision_at_i

  return score / n_relevant

######## >>>>> EVALUASI !
def eval(qrels, query_file = "queries.txt", k = 1000):
  """
  loop ke semua 30 query, hitung score di setiap query,
  lalu hitung mean score untuk semua query.
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection',
                            postings_encoding = VBEPostings,
                            output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ndcg_scores = []
    ap_scores = []

    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      ranking = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k):
        did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
        ranking.append(qrels[qid][did])

      # kalau hasil retrieval kurang dari k, anggap sisanya tidak relevan
      if len(ranking) < k:
        ranking += [0] * (k - len(ranking))

      n_relevant = sum(qrels[qid].values())

      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ndcg_scores.append(ndcg(ranking, n_relevant))
      ap_scores.append(ap(ranking, n_relevant))

  print("Hasil evaluasi TF-IDF terhadap 30 queries")
  print("RBP score  =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score  =", sum(dcg_scores) / len(dcg_scores))
  print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
  print("AP score   =", sum(ap_scores) / len(ap_scores))