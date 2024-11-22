# bm25-search-engine

## **Overview**
This project implements a BM25-based search engine for the SciQ dataset. It provides an intuitive Gradio interface for users to search and retrieve science-related information efficiently. The engine leverages the BM25 ranking algorithm and processes the SciQ dataset, which includes over 13,000 science questions with explanations.

---

## **Features**
- **BM25 Ranking Algorithm**: Efficiently ranks and retrieves relevant documents.
- **SciQ Dataset**: Over 13,000 multiple-choice science questions covering biology, physics, and chemistry.
- **Gradio Interface**: A user-friendly web interface for querying the dataset.

## **Setup Instructions**
- Install the required Python packages using requirements.txt
  - run: pip install -r requirements.txt
- To build the BM25 index, run the main script:
  - run: python main.py
Once the index is built, the Gradio interface launches automatically. Open the provided URL in your browser (default is http://127.0.0.1:7860/) to start using the search engine.
