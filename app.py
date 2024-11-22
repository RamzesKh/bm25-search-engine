from __future__ import annotations
from dataclasses import dataclass
import pickle
import os
from typing import Iterable, Callable, List, Dict, Optional, Type, TypeVar, TypedDict
from nlp4web_codebase.ir.data_loaders.dm import Document
from nlp4web_codebase.ir.data_loaders.sciq import load_sciq
from nlp4web_codebase.ir.models import BaseRetriever
from collections import Counter
import tqdm
import re
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_stopwords
from dataclasses import asdict, dataclass
import math
from abc import abstractmethod
import gradio as gr


def word_splitting(text: str) -> List[str]:
    return word_splitter(text.lower())

def lemmatization(words: List[str]) -> List[str]:
    return words  # We ignore lemmatization here for simplicity

def simple_tokenize(text: str) -> List[str]:
    words = word_splitting(text)
    tokenized = list(filter(lambda w: w not in stopwords, words))
    tokenized = lemmatization(tokenized)
    return tokenized


@dataclass
class PostingList:
    term: str  # The term
    docid_postings: List[int]  # docid_postings[i] means the docid (int) of the i-th associated posting
    tweight_postings: List[float]  # tweight_postings[i] means the term weight (float) of the i-th associated posting


@dataclass
class InvertedIndex:
    posting_lists: List[PostingList]  # docid -> posting_list
    vocab: Dict[str, int]
    cid2docid: Dict[str, int]  # collection_id -> docid
    collection_ids: List[str]  # docid -> collection_id
    doc_texts: Optional[List[str]] = None  # docid -> document text

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "index.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_saved(cls: Type[T], saved_dir: str) -> T:
        index = cls(
            posting_lists=[], vocab={}, cid2docid={}, collection_ids=[], doc_texts=None
        )
        with open(os.path.join(saved_dir, "index.pkl"), "rb") as f:
            index = pickle.load(f)
        return index


# The output of the counting function:
@dataclass
class Counting:
    posting_lists: List[PostingList]
    vocab: Dict[str, int]
    cid2docid: Dict[str, int]
    collection_ids: List[str]
    dfs: List[int]  # tid -> df
    dls: List[int]  # docid -> doc length
    avgdl: float
    nterms: int
    doc_texts: Optional[List[str]] = None

def run_counting(
    documents: Iterable[Document],
    tokenize_fn: Callable[[str], List[str]] = simple_tokenize,
    store_raw: bool = True,  # store the document text in doc_texts
    ndocs: Optional[int] = None,
    show_progress_bar: bool = True,
) -> Counting:
    """Counting TFs, DFs, doc_lengths, etc."""
    posting_lists: List[PostingList] = []
    vocab: Dict[str, int] = {}
    cid2docid: Dict[str, int] = {}
    collection_ids: List[str] = []
    dfs: List[int] = []  # tid -> df
    dls: List[int] = []  # docid -> doc length
    nterms: int = 0
    doc_texts: Optional[List[str]] = []
    for doc in tqdm.tqdm(
        documents,
        desc="Counting",
        total=ndocs,
        disable=not show_progress_bar,
    ):
        if doc.collection_id in cid2docid:
            continue
        collection_ids.append(doc.collection_id)
        docid = cid2docid.setdefault(doc.collection_id, len(cid2docid))
        toks = tokenize_fn(doc.text)
        tok2tf = Counter(toks)
        dls.append(sum(tok2tf.values()))
        for tok, tf in tok2tf.items():
            nterms += tf
            tid = vocab.get(tok, None)
            if tid is None:
                posting_lists.append(
                    PostingList(term=tok, docid_postings=[], tweight_postings=[])
                )
                tid = vocab.setdefault(tok, len(vocab))
            posting_lists[tid].docid_postings.append(docid)
            posting_lists[tid].tweight_postings.append(tf)
            if tid < len(dfs):
                dfs[tid] += 1
            else:
                dfs.append(0)
        if store_raw:
            doc_texts.append(doc.text)
        else:
            doc_texts = None
    return Counting(
        posting_lists=posting_lists,
        vocab=vocab,
        cid2docid=cid2docid,
        collection_ids=collection_ids,
        dfs=dfs,
        dls=dls,
        avgdl=sum(dls) / len(dls),
        nterms=nterms,
        doc_texts=doc_texts,
    )





@dataclass
class BM25Index(InvertedIndex):

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return simple_tokenize(text)

    @staticmethod
    def cache_term_weights(
        posting_lists: List[PostingList],
        total_docs: int,
        avgdl: float,
        dfs: List[int],
        dls: List[int],
        k1: float,
        b: float,
    ) -> None:
        """Compute term weights and caching"""

        N = total_docs
        for tid, posting_list in enumerate(
            tqdm.tqdm(posting_lists, desc="Regularizing TFs")
        ):
            idf = BM25Index.calc_idf(df=dfs[tid], N=N)
            for i in range(len(posting_list.docid_postings)):
                docid = posting_list.docid_postings[i]
                tf = posting_list.tweight_postings[i]
                dl = dls[docid]
                regularized_tf = BM25Index.calc_regularized_tf(
                    tf=tf, dl=dl, avgdl=avgdl, k1=k1, b=b
                )
                posting_list.tweight_postings[i] = regularized_tf * idf

    @staticmethod
    def calc_regularized_tf(
        tf: int, dl: float, avgdl: float, k1: float, b: float
    ) -> float:
        return tf / (tf + k1 * (1 - b + b * dl / avgdl))

    @staticmethod
    def calc_idf(df: int, N: int):
        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    @classmethod
    def build_from_documents(
        cls: Type[BM25Index],
        documents: Iterable[Document],
        store_raw: bool = True,
        output_dir: Optional[str] = None,
        ndocs: Optional[int] = None,
        show_progress_bar: bool = True,
        k1: float = 0.9,
        b: float = 0.4,
    ) -> BM25Index:
        # Counting TFs, DFs, doc_lengths, etc.:
        counting = run_counting(
            documents=documents,
            tokenize_fn=BM25Index.tokenize,
            store_raw=store_raw,
            ndocs=ndocs,
            show_progress_bar=show_progress_bar,
        )

        # Compute term weights and caching:
        posting_lists = counting.posting_lists
        total_docs = len(counting.cid2docid)
        BM25Index.cache_term_weights(
            posting_lists=posting_lists,
            total_docs=total_docs,
            avgdl=counting.avgdl,
            dfs=counting.dfs,
            dls=counting.dls,
            k1=k1,
            b=b,
        )

        # Assembly and save:
        index = BM25Index(
            posting_lists=posting_lists,
            vocab=counting.vocab,
            cid2docid=counting.cid2docid,
            collection_ids=counting.collection_ids,
            doc_texts=counting.doc_texts,
        )
        return index


class BaseInvertedIndexRetriever(BaseRetriever):

    @property
    @abstractmethod
    def index_class(self) -> Type[InvertedIndex]:
        pass

    def __init__(self, index_dir: str) -> None:
        self.index = self.index_class.from_saved(index_dir)

    def get_term_weights(self, query: str, cid: str) -> Dict[str, float]:
        toks = self.index.tokenize(query)
        target_docid = self.index.cid2docid[cid]
        term_weights = {}
        for tok in toks:
            if tok not in self.index.vocab:
                continue
            tid = self.index.vocab[tok]
            posting_list = self.index.posting_lists[tid]
            for docid, tweight in zip(
                posting_list.docid_postings, posting_list.tweight_postings
            ):
                if docid == target_docid:
                    term_weights[tok] = tweight
                    break
        return term_weights

    def score(self, query: str, cid: str) -> float:
        return sum(self.get_term_weights(query=query, cid=cid).values())

    def retrieve(self, query: str, topk: int = 10) -> Dict[str, float]:
        toks = self.index.tokenize(query)
        docid2score: Dict[int, float] = {}
        for tok in toks:
            if tok not in self.index.vocab:
                continue
            tid = self.index.vocab[tok]
            posting_list = self.index.posting_lists[tid]
            for docid, tweight in zip(
                posting_list.docid_postings, posting_list.tweight_postings
            ):
                docid2score.setdefault(docid, 0)
                docid2score[docid] += tweight
        docid2score = dict(
            sorted(docid2score.items(), key=lambda pair: pair[1], reverse=True)[:topk]
        )
        return {
            self.index.collection_ids[docid]: score
            for docid, score in docid2score.items()
        }


class BM25Retriever(BaseInvertedIndexRetriever):

    @property
    def index_class(self) -> Type[BM25Index]:
        return BM25Index


class Hit(TypedDict):
  cid: str
  score: float
  text: str


if __name__ == "__main__":
    T = TypeVar("T", bound="InvertedIndex")
    LANGUAGE = "english"
    word_splitter = re.compile(r"(?u)\b\w\w+\b").findall
    stopwords = set(nltk_stopwords.words(LANGUAGE))
    
    
    sciq = load_sciq()
    counting = run_counting(documents=iter(sciq.corpus), ndocs=len(sciq.corpus))

    bm25_index = BM25Index.build_from_documents(
        documents=iter(sciq.corpus),
        ndocs=len(sciq.corpus),
        k1=0.9,
        b=0.4
        )
    bm25_index.save("output/bm25_index_b")  # Save index to directory
    bm25_retriever = BM25Retriever(index_dir="output/bm25_index_b")
    
    corpus_dict = {doc.collection_id: doc.text for doc in sciq.corpus} 
    
    def handle_search(query: str) -> List[Hit]:
        results = bm25_retriever.retrieve(query)
        hits = [
            {
                "cid": cid,
                "score": score,
                "text": corpus_dict[cid]  # Assuming sciq.corpus maps cids to document texts
            }
            for cid, score in results.items()
        ]
        return hits
       
        
    demo = gr.Interface(
        fn=handle_search,
        inputs=gr.Textbox(label="Enter your search query"), 
        outputs=gr.Textbox(label="Search Results", lines=20, interactive=False), 
        title="BM25 Search Engine Demo on SciQ Dataset", 
        description="Enter your search query to get the results from the SciQ dataset."
    )
    
    
    demo.launch(debug=True)

