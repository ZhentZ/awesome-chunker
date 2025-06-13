import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from semantic_chunker import SemanticChunker


class SemanticTextChunker:
    def __init__(self, model_path="../../models/sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens=100, cluster_threshold=0.4, similarity_threshold=0.4,
                 buffer_size=1, breakpoint_percentile_threshold=95, max_chunk_length=512):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.cluster_threshold = cluster_threshold
        self.similarity_threshold = similarity_threshold
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.max_chunk_length = max_chunk_length

    def load_and_preprocess_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            essay = file.read()
        single_sentences_list = re.split(r'(?<=[.?!])\s+', essay)
        sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]
        return sentences

    def combine_sentences(self, sentences):
        for i in range(len(sentences)):
            combined_sentence = ''
            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '
            combined_sentence += sentences[i]['sentence']
            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']
            sentences[i]['combined_sentence'] = combined_sentence
        return sentences

    def get_sentence_embeddings(self, sentences):
        semantic_chunker = SemanticChunker(
            model_name=self.model_path,
            max_tokens=self.max_tokens,
            cluster_threshold=self.cluster_threshold,
            similarity_threshold=self.similarity_threshold)

        embeddings = semantic_chunker.get_embeddings([{'text': x['combined_sentence']} for x in sentences])

        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        return sentences

    def calculate_cosine_distances(self, sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            distance = 1 - similarity
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance
        return distances, sentences

    def visualize_and_find_breakpoints(self, distances, sentences):
        breakpoint_distance_threshold = np.percentile(distances, self.breakpoint_percentile_threshold)
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
        return indices_above_thresh

    def combine_sentences_into_chunks(self, sentences, indices_above_thresh):
        start_index = 0
        chunks = []
        current_chunk = []

        for index in indices_above_thresh:
            end_index = index
            for i in range(start_index, end_index + 1):
                sentence = sentences[i]['sentence']
                if len(' '.join(current_chunk + [sentence])) <= self.max_chunk_length:
                    current_chunk.append(sentence)
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
            start_index = index + 1

        for i in range(start_index, len(sentences)):
            sentence = sentences[i]['sentence']
            if len(' '.join(current_chunk + [sentence])) <= self.max_chunk_length:
                current_chunk.append(sentence)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def semantic_chunking(self, file_path):
        sentences = self.load_and_preprocess_text(file_path)
        sentences = self.combine_sentences(sentences)
        sentences = self.get_sentence_embeddings(sentences)
        distances, sentences = self.calculate_cosine_distances(sentences)
        indices_above_thresh = self.visualize_and_find_breakpoints(distances, sentences)
        chunks = self.combine_sentences_into_chunks(sentences, indices_above_thresh)
        return chunks


if __name__ == "__main__":
    file_path = '../../data/PGEssays/mit.txt'
    chunker = SemanticTextChunker()
    chunks = chunker.semantic_chunking(file_path)
    for i, chunk in enumerate(chunks[:2]):
        buffer = 200
        print(f"Chunk #{i}")
        print(chunk[:buffer].strip())
        print("...")
        print(chunk[-buffer:].strip())
        print("\n")