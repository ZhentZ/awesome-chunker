import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from collections import defaultdict


class SemanticChunker:
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',  # 用于生成文本嵌入的模型名称，默认为 all-MiniLM-L6-v2
        max_tokens=512,  # 合并后每个文本块允许的最大令牌数，默认为 512
        cluster_threshold=0.5,  # 聚类的阈值，控制聚类的粒度，默认为 0.5
        similarity_threshold=0.4  # 确定语义对的最小相似度阈值，默认为 0.4
    ):
        self.device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"  # 选择可用的设备进行计算，优先选择 CUDA，其次是 MPS，最后是 CPU
        )
        print(f"[Info] Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.max_tokens = max_tokens
        self.cluster_threshold = cluster_threshold
        self.similarity_threshold = similarity_threshold
        self.tokenizer = self.model.tokenizer if hasattr(self.model, "tokenizer") else None

    def get_embeddings(self, chunks: List[Dict[str, Any]]):
        """
        获取文本块的嵌入表示。
        :param chunks:
        :return:
        """
        if not chunks:
            return np.array([])
        texts = [chunk["text"] for chunk in chunks]
        return np.array(self.model.encode(texts, show_progress_bar=False))

    def compute_similarity(self, embeddings):
        """
        计算嵌入表示之间的余弦相似度矩阵
        :param embeddings:
        :return:
        """
        if embeddings.size == 0:
            return np.zeros((0, 0))
        return cosine_similarity(embeddings)

    def cluster_chunks(self, similarity_matrix, threshold=0.5):
        """
        使用并查集算法对相似度矩阵进行聚类。
        :param similarity_matrix:
        :param threshold:
        :return:
        """
        n = similarity_matrix.shape[0]  # 获取嵌入矩阵的行数，即文本块的数量
        parent = list(range(n))  # 初始化父节点列表，每个节点的父节点为自身

        def find(x):
            """
            查找并查集的根节点，并进行路径压缩以优化查询效率。
            """
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            """
            合并两个节点所在的集合（并查集的合并操作）
            """
            parent[find(x)] = find(y)

        # 遍历相似度矩阵，将相似度大于等于阈值的文本块合并到同一个集合中
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    union(i, j)

        clusters = [find(i) for i in range(n)]  # 为每个文本块找到其所在集合的根节点
        cluster_map = {cid: idx for idx, cid in enumerate(sorted(set(clusters)))}  # 创建一个映射，将每个唯一的集合ID映射到一个连续的索引
        return [cluster_map[c] for c in clusters]

    def merge_chunks(self, chunks: List[Dict[str, Any]], clusters: List[int]) -> List[Dict[str, Any]]:
        """
        合并相似的文本块。
        :param chunks:
        :param clusters:
        :return:
        """
        # 如果输入的文本块列表或聚类列表为空，则返回空列表
        if not chunks or not clusters:
            return []

        cluster_map = defaultdict(list)  # 使用 defaultdict 存储每个聚类对应的文本块列表
        # 将每个文本块添加到其所属的聚类列表中
        for idx, cluster_id in enumerate(clusters):
            cluster_map[cluster_id].append(chunks[idx])

        merged_chunks = []  # 创建一个空列表，用于存储合并后的文本块
        # 遍历每个聚类，将文本块合并到一起
        for chunk_list in cluster_map.values():
            current_text = ""
            current_meta = []
            # 遍历每个文本块
            for chunk in chunk_list:
                # 尝试将当前文本和下一个文本块合并
                next_text = (current_text + " " + chunk["text"]).strip()
                # 如果有分词器，则计算合并后文本的令牌数
                if self.tokenizer:
                    num_tokens = len(self.tokenizer.encode(next_text))
                else:
                    num_tokens = len(next_text.split())
                # 如果合并后文本的令牌数超过最大令牌数，且当前文本不为空
                if num_tokens > self.max_tokens and current_text:
                    # 将当前合并文本添加到合并后的文本块列表中
                    merged_chunks.append({
                        "text": current_text,
                        "metadata": current_meta
                    })
                    current_text = chunk["text"]  # 重置当前合并文本为下一个文本块
                    current_meta = [chunk]  # 重置当前合并文本的元数据为下一个文本块
                else:
                    current_text = next_text
                    current_meta.append(chunk)

            if current_text:
                merged_chunks.append({
                    "text": current_text,
                    "metadata": current_meta
                })

        return merged_chunks

    def find_top_semantic_pairs(self, similarity_matrix, min_similarity=0.4, top_k=50):
        """
        找到相似度矩阵中最相似的文本对。
        """
        pairs = []
        n = similarity_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[i, j]
                if sim >= min_similarity:
                    pairs.append((i, j, sim))
        pairs.sort(key=lambda x: x[2], reverse=True)  # 按相似度降序排序
        return pairs[:top_k]

    def chunk(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对文本块进行聚类和合并。
        """
        embeddings = self.get_embeddings(chunks)
        similarity_matrix = self.compute_similarity(embeddings)
        clusters = self.cluster_chunks(similarity_matrix, threshold=self.cluster_threshold)
        return self.merge_chunks(chunks, clusters)

    def get_debug_info(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optional: for visualization or export/debug purposes."""
        embeddings = self.get_embeddings(chunks)
        similarity_matrix = self.compute_similarity(embeddings)
        clusters = self.cluster_chunks(similarity_matrix, threshold=self.cluster_threshold)
        merged_chunks = self.merge_chunks(chunks, clusters)
        semantic_pairs = self.find_top_semantic_pairs(similarity_matrix, min_similarity=self.similarity_threshold)

        return {
            "original_chunks": chunks,
            "embeddings": embeddings,
            "similarity_matrix": similarity_matrix,
            "clusters": clusters,
            "semantic_pairs": semantic_pairs,
            "merged_chunks": merged_chunks
        }
