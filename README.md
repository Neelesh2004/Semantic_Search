# Semantic Search Pipeline with Pinecone and Cross-Encoder

This repository contains a full semantic search pipeline demonstrating the power of bi-encoders combined with cross-encoders to achieve fast and highly accurate document retrieval. 

## 🔍 What it does

The notebook (`semantic_search_pinecone_example.ipynb`) builds an end-to-end semantic search tool using the following stack:

- **Dataset**: SQuAD (Stanford Question Answering Dataset) — uses Wikipedia passages.
- **Embeddings**: Uses OpenAI's `text-embedding-3-small` to encode the query and text passages into vectors. It also includes a **free local fallback** option (`all-MiniLM-L6-v2` using colab CPU) if you do not have an OpenAI API key.
- **Vector Database**: Uses **Pinecone (Serverless)** to index the embedded vectors and query them using Approximate Nearest Neighbor (ANN) search via Cosine Similarity.
- **Reranker**: Employs a Hugging Face Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to accurately rerank the top-K candidates retrieved from Pinecone.

### The Pipeline Structure
```text
Query ──► Embed ──► Pinecone ANN Search ──► Top-K candidates
                                                   │
                                  ┌────────────────┘
                                  ▼
                       Cross-Encoder Reranking ──► Final Top-N Best Matches
```

It showcases how using a **Two-Stage Pipeline** gives the best of both worlds:
1. **Stage 1 (Retrieval):** Pinecone + Bi-Encoder handles vast amounts of search across millions of items efficiently.
2. **Stage 2 (Reranking):** Cross-Encoder accurately re-scores the retrieved subset so the final ranking understands fine-grained token-level relationships between the query and documents.

It also includes:
- Interactive search queries.
- Latency benchmarks comparing bi-encoder-only vs bi-encoder + cross-encoder searches.
- Visualizing data score distributions using Matplotlib charts.

## 🚀 How to Execute

To easily run the pipeline without having to set up the environment locally, we recommend running it via Google Colab.

1. **Download the Notebook:** Download the `semantic_search_pinecone_example.ipynb` file from this repository.
2. **Open in Google Colab:** Navigate to [Google Colab](https://colab.research.google.com/) and click on **File > Upload notebook**, then select the downloaded `.ipynb` file.
3. **Run the Notebook:** Run the cells sequentially. 
    - The first cell will install the required pip dependencies automatically (`openai`, `pinecone`, `sentence-transformers`, `datasets`, `tqdm`).
    - You will be asked to enter your OpenAI API key and Pinecone API key. Once loaded, you can experiment with search queries and view the reranking capabilities in real time.
    *(Note: If you lack an OpenAI API key, follow the instructions in the notebook to trigger the local fallback embedding feature).*
