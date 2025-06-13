# Awesome-Chunker

🚀🚀🚀This project aims to gather and synthesize diverse text chunking methods for the Retrieval-Augmented Generation (RAG) task, including but not limited to academic papers, algorithm interpretations, and code implementations.

💡💡💡The goal of the project is to provide researchers with a comprehensive repository of resources to support their ability to more access easily and understand various text chunking techniques when developing and optimizing RAG-based AI systems.

## Installation
To use this project, you need to install the required dependencies. These dependencies are for reference only. You can install them using `pip`:
```bash
pip install -r requirements.txt
```

## 1. classic_chunker
Classic chunking methods that divide text based on characters, document types, and other fundamental approaches.
- **Character Splitting**
    - **manual splitting**: Manually split text by a fixed character length. The code is located in `classic_chunker/character_splitting.py`.
  
    - **LangChain's CharacterTextSplitter**: Use `CharacterTextSplitter` from the LangChain library for character - based splitting. The code is in `classic_chunker/character_splitting.py`.
     
    - **Llama Index's SentenceSplitter**: Split text with `SentenceSplitter` from Llama Index. The code can be found in `classic_chunker/character_splitting.py`.
      
- **Document - Specific Splitting**: Split different types of documents such as Markdown, Python, and JavaScript. The code is in `classic_chunker/document_specific_splitting.py`.

- **Recursive Character Text Splitting**: Employ `RecursiveCharacterTextSplitter` from LangChain to perform recursive splitting based on different delimiter hierarchies. The code is located in `classic_chunker/recursive_character_text_splitting.py`.

## 2. semantic_chunker
Semantic chunking methods that cluster and merge text chunks based on semantic similarity.

- The core code is in `semantic_chunker/core.py`, which contains the core logic for semantic chunking.

- The code can be referred to as `semantic_chunker/sentence_order.py` and `semantic_chunker/sentence_disorder.py`.

- `semantic_chunker/sentence_order.py` does not disrupt the sentence order, while `semantic_chunker/sentence_disorder.py` clusters based on semantic clustering.

- For the relevant models, you can download the models to the `models` folder by using the `download_model.py`.

## 3. dense_x_retrieval
This method proposes to use **propositions** as the new search unit.

For detailed instructions, see `dense_x_retrieval/doc/Dense X Retrieval- What Retrieval Granularity Shou`.

The chunking method based on the paper [Dense X Retrieval](https://arxiv.org/pdf/2312.06648). 

- `dense_x_retrieval/dense_x_retrieval.py` is a simple running example.The code can be referred to https://github.com/chentong0/factoid-wiki.

- For the relevant models, you can download the models to the `models` folder by using the `download_model.py`.

## 4. LumberChunker
A method that leverages a Large Language Model (LLM) to dynamically segment documents into semantically independent chunks. It identifies content transition points through iterative prompts to the LLM.

The chunking method based on the paper [LumberChunker](https://arxiv.org/pdf/2406.17526). 

The complete code can be referred to https://github.com/joaodsmarques/LumberChunker.

- For detailed instructions, see `LumberChunker/doc/LumberChunker - Long-Form Narrative Document Segmentation/LumberChunker.md`.

- The example code is in `LumberChunker/Code/LumberChunker.py` or `LumberChunker/Code/LumberChunker-Segmentation.py`.

## 5. Meta-Chunking
This method introduces the concept of Meta-Chunking and defines a new text chunking granularity - Meta-Chunking, which lies between sentences and paragraphs. It can capture the deep logical relationships between sentences and enhance the logical coherence of text chunking.

The chunking method based on the paper [Meta-Chunking](https://arxiv.org/abs/2410.12788).

The complete code can be referred to https://github.com/IAAR-Shanghai/Meta-Chunking.

- For detailed instructions, see `Meta-Chunking/doc/Meta-Chunking- Learning Efficient Text Segmentatio.md`.

- The example code is in `Meta-Chunking/example/perplexity_chunking.py`.
