import os
import pickle
import contextlib
import heapq
import math
import bisect

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map : IdMap
        Untuk mapping terms ke termIDs
    doc_id_map : IdMap
        Untuk mapping relative paths dari dokumen
        (misal, /collection/0/gamma.txt) ke docIDs
    data_dir : str
        Path ke data
    output_dir : str
        Path ke output index files
    postings_encoding :
        Lihat di compression.py, kandidatnya adalah StandardPostings,
        VBEPostings, dsb.
    index_name : str
        Nama dari file yang berisi inverted index
    """

    def __init__(
        self,
        data_dir,
        output_dir,
        postings_encoding,
        index_name="main_index",
    ):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map dan term_id_map ke output directory via pickle."""
        with open(os.path.join(self.output_dir, "terms.dict"), "wb") as f:
            pickle.dump(self.term_id_map, f)

        with open(os.path.join(self.output_dir, "docs.dict"), "wb") as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map dan term_id_map dari output directory."""
        with open(os.path.join(self.output_dir, "terms.dict"), "rb") as f:
            self.term_id_map = pickle.load(f)

        with open(os.path.join(self.output_dir, "docs.dict"), "rb") as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming Bahasa Inggris.
        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[int, int]]
            Returns all the td_pairs extracted from the block.
            Mengembalikan semua pasangan <termID, docID> dari sebuah block
            (dalam hal ini sebuah sub-direktori di dalam folder collection).

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua
        pemanggilan parse_block(...).
        """
        directory = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []

        for filename in next(os.walk(directory))[2]:
            docname = directory + "/" + filename
            with open(
                docname,
                "r",
                encoding="utf8",
                errors="surrogateescape",
            ) as f:
                for token in f.read().split():
                    td_pairs.append(
                        (self.term_id_map[token], self.doc_id_map[docname])
                    )

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index.

        Disini diterapkan konsep BSBI dimana hanya di-maintain satu dictionary
        besar untuk keseluruhan block. Namun dalam teknik penyimpanannya
        digunakan strategi dari SPIMI yaitu penggunaan struktur data hashtable
        (dalam Python bisa berupa dictionary).

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan juga list
        of sorted Doc IDs. Sekarang di Tugas Pemrograman 2, kita juga perlu
        tambahkan list of TF.

        Parameters
        ----------
        td_pairs : List[Tuple[int, int]]
            List of termID-docID pairs
        index : InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}

        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}

            term_dict[term_id].add(doc_id)

            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0

            term_tf[term_id][doc_id] += 1

        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT.

        Gunakan fungsi sorted_merge_posts_and_tfs(..) di modul util.

        Parameters
        ----------
        indices : List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inverted index yang iterable
            di sebuah block.

        merged_index : InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)

        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(
                    list(zip(postings, tf_list)),
                    list(zip(postings_, tf_list_)),
                )
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_

        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))   jika tf(t, D) > 0
                 = 0                   jika sebaliknya

        w(t, Q) = IDF = log(N / df(t))

        Score = untuk setiap term di query, akumulasikan
                w(t, Q) * w(t, D)
                (tidak perlu dinormalisasi dengan panjang dokumen)

        Catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index,
               len(doc_length)

        Parameters
        ----------
        query : str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Returns
        -------
        List[Tuple[float, str]]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil berdasarkan skor.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]

        with InvertedIndexReader(
            self.index_name,
            self.postings_encoding,
            directory=self.output_dir,
        ) as merged_index:
            scores = {}

            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)

                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]

                        if doc_id not in scores:
                            scores[doc_id] = 0

                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            docs = [
                (score, self.doc_id_map[doc_id])
                for (doc_id, score) in scores.items()
            ]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Ranked retrieval dengan BM25.

        Score(D, Q) = sum_t in Q∩D [
            log(N / df_t) *
            ((k1 + 1) * tf_tD) /
            (k1 * ((1 - b) + b * (dl / avgdl)) + tf_tD)
        ]
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Jangan pakai self.term_id_map[word] langsung untuk OOV,
        # karena itu akan membuat term baru di map.
        terms = []
        for word in query.split():
            if word in self.term_id_map.str_to_id:
                terms.append(self.term_id_map.str_to_id[word])

        with InvertedIndexReader(
            self.index_name,
            self.postings_encoding,
            directory=self.output_dir,
        ) as merged_index:
            scores = {}
            N = len(merged_index.doc_length)
            avgdl = merged_index.avg_doc_length

            if N == 0 or avgdl == 0:
                return []

            for term in terms:
                if term not in merged_index.postings_dict:
                    continue

                df = merged_index.postings_dict[term][1]
                postings, tf_list = merged_index.get_postings_list(term)
                idf = math.log(N / df)

                for doc_id, tf in zip(postings, tf_list):
                    dl = merged_index.doc_length[doc_id]

                    bm25_tf = ((k1 + 1) * tf) / (
                        k1 * ((1 - b) + b * (dl / avgdl)) + tf
                    )

                    if doc_id not in scores:
                        scores[doc_id] = 0.0

                    scores[doc_id] += idf * bm25_tf

            docs = [
                (score, self.doc_id_map[doc_id])
                for doc_id, score in scores.items()
            ]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def _query_to_term_ids(self, query):
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        term_ids = []
        for word in query.split():
            if word in self.term_id_map.str_to_id:
                term_ids.append(self.term_id_map.str_to_id[word])

        return term_ids

    def _bm25_term_score(self, tf, df, dl, N, avgdl, k1=1.2, b=0.75):
        if tf <= 0 or df <= 0 or N == 0 or avgdl == 0:
            return 0.0

        idf = math.log(N / df)
        denom = k1 * ((1 - b) + b * (dl / avgdl)) + tf
        return idf * (((k1 + 1) * tf) / denom)

    def _bm25_term_upper_bound(
        self,
        df,
        max_tf,
        N,
        min_dl,
        avgdl,
        k1=1.2,
        b=0.75,
    ):
        return self._bm25_term_score(max_tf, df, min_dl, N, avgdl, k1, b)

    def retrieve_wand_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        WAND Top-K retrieval dengan scoring BM25.
        Menggunakan upper bound per term agar tidak semua dokumen
        dihitung exact score-nya.
        """
        term_ids = self._query_to_term_ids(query)

        with InvertedIndexReader(
            self.index_name,
            self.postings_encoding,
            directory=self.output_dir,
        ) as merged_index:
            N = len(merged_index.doc_length)
            avgdl = merged_index.avg_doc_length
            min_dl = merged_index.min_doc_length

            if N == 0 or avgdl == 0:
                return []

            cursors = []
            for term in term_ids:
                if term not in merged_index.postings_dict:
                    continue

                postings, tf_list = merged_index.get_postings_list(term)
                _, df, _, _, max_tf = merged_index.postings_dict[term]

                ub = self._bm25_term_upper_bound(
                    df, max_tf, N, min_dl, avgdl, k1, b
                )

                cursors.append(
                    {
                        "term": term,
                        "df": df,
                        "postings": postings,
                        "tf_list": tf_list,
                        "idx": 0,
                        "ub": ub,
                    }
                )

            if len(cursors) == 0:
                return []

            topk_heap = []
            threshold = 0.0

            while True:
                active = [c for c in cursors if c["idx"] < len(c["postings"])]
                if len(active) == 0:
                    break

                active.sort(key=lambda c: c["postings"][c["idx"]])

                ub_sum = 0.0
                pivot_idx = None

                for i, c in enumerate(active):
                    ub_sum += c["ub"]
                    if ub_sum > threshold:
                        pivot_idx = i
                        break

                if pivot_idx is None:
                    break

                pivot_doc = active[pivot_idx]["postings"][active[pivot_idx]["idx"]]
                smallest_doc = active[0]["postings"][active[0]["idx"]]

                if smallest_doc == pivot_doc:
                    score = 0.0
                    dl = merged_index.doc_length[pivot_doc]

                    for c in active:
                        idx = c["idx"]
                        if (
                            idx < len(c["postings"])
                            and c["postings"][idx] == pivot_doc
                        ):
                            tf = c["tf_list"][idx]
                            score += self._bm25_term_score(
                                tf, c["df"], dl, N, avgdl, k1, b
                            )

                    if len(topk_heap) < k:
                        heapq.heappush(topk_heap, (score, pivot_doc))
                    elif score > topk_heap[0][0]:
                        heapq.heapreplace(topk_heap, (score, pivot_doc))

                    threshold = topk_heap[0][0] if len(topk_heap) == k else 0.0

                    for c in active:
                        idx = c["idx"]
                        if (
                            idx < len(c["postings"])
                            and c["postings"][idx] == pivot_doc
                        ):
                            c["idx"] += 1
                else:
                    c = active[0]
                    c["idx"] = bisect.bisect_left(
                        c["postings"],
                        pivot_doc,
                        lo=c["idx"],
                    )

            results = [
                (score, self.doc_id_map[doc_id]) for score, doc_id in topk_heap
            ]
            return sorted(results, key=lambda x: x[0], reverse=True)

    def index(self):
        """
        Base indexing code.
        BAGIAN UTAMA untuk melakukan indexing dengan skema BSBI
        (blocked-sort based indexing).

        Method ini scan terhadap semua data di collection, memanggil
        parse_block untuk parsing dokumen dan memanggil invert_write yang
        melakukan inversion di setiap block dan menyimpannya ke index yang baru.
        """
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = "intermediate_index_" + block_dir_relative
            self.intermediate_indices.append(index_id)

            with InvertedIndexWriter(
                index_id,
                self.postings_encoding,
                directory=self.output_dir,
            ) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(
            self.index_name,
            self.postings_encoding,
            directory=self.output_dir,
        ) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(
                            index_id,
                            self.postings_encoding,
                            directory=self.output_dir,
                        )
                    )
                    for index_id in self.intermediate_indices
                ]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    bsbi_instance = BSBIIndex(
        data_dir="collection",
        postings_encoding=VBEPostings,
        output_dir="index",
    )
    bsbi_instance.index()
    