from langchain.text_splitter import CharacterTextSplitter
from typing import List, Union
import logging

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, Document


def chunk_text_manual(text, chunk_size):
    """
    手动实现文本分块
    :param text:
    :param chunk_size:
    :return:
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def chunk_text_langChain(text, chunk_size=35, chunk_overlap=0, separator=None, strip_whitespace=True):
    """
    使用LangChain的CharacterTextSplitter进行文本分块
    :param text: 要切分的文本
    :param chunk_size: 切分的文本块大小
    :param chunk_overlap: 切分的文本块重叠大小
    :param separator: 按指定分隔符切分文本
    :param strip_whitespace: 是否去除块首尾的空白字符,True（默认）自动移除块首尾的空白字符,False则保留空白字符
    :return:
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator, strip_whitespace=strip_whitespace)
    documents = text_splitter.create_documents([text])
    return documents


def chunk_text_llama_index(
        input_text: Union[str, List[str], str],  # 支持单个文本、文本列表或文件路径
        chunk_size: int = 200,  # 每个块的最大 token 数（默认 200）
        chunk_overlap: int = 15,  # 相邻块重叠 token 数（默认 15）
        use_file_loader: bool = False,  # 是否为文件路径输入
        file_extension: str = ".txt",  # 文件扩展名（默认 txt）
        **kwargs  # 其他 Llama Index 配置参数
) -> List[Document]:
    """
    使用 Llama Index 的 SentenceSplitter 进行智能句子分割

    参数:
        input_text: 输入文本（字符串）、文本列表或文件路径
        chunk_size: 每个文本块的最大 token 数（默认 200）
        chunk_overlap: 相邻块的重叠 token 数（默认 15）
        use_file_loader: 是否启用文件加载模式（默认 False）
        file_extension: 输入文件扩展名（默认 .txt）
        **kwargs: 传递给 SentenceSplitter 的其他参数（如 `separator`）

    返回:
        List[Document]: Llama Index 的文档节点列表
    """
    try:
        # 初始化分割器
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )

        # 处理输入数据
        if use_file_loader:
            # 验证文件路径有效性
            if not isinstance(input_text, str):
                raise ValueError("文件路径输入必须为字符串")
            if not input_text.endswith(file_extension):
                raise ValueError(f"文件扩展名必须为 {file_extension}")

            # 加载文件数据
            loader = SimpleDirectoryReader(input_dir="", input_files=[input_text])
            documents = loader.load_data()
        else:
            # 处理文本输入（支持单个文本或文本列表）
            if isinstance(input_text, str):
                documents = [Document(text=input_text)]
            elif isinstance(input_text, list):
                documents = [Document(text=t) for t in input_text if t.strip()]
            else:
                raise TypeError("输入必须为字符串、字符串列表或文件路径")

        # 执行分割
        nodes = splitter.get_nodes_from_documents(documents)

        # 转换为文档列表（可选：如需返回纯文本块，可改为 [node.text for node in nodes]）
        return [Document(text=node.text, metadata=node.metadata) for node in nodes]

    except Exception as e:
        logging.error(f"文本分割失败: {str(e)}")
        raise


if __name__ == '__main__':
    text = "  This is the text I would like to chunk up. It is the example text for this exercise  "
    # print(chunk_text_manual(text, 35))
    # documents = chunk_text_langChain(text, 35, 0, separator="", strip_whitespace=False)
    # for doc in documents:
    #     print(doc.page_content)
    # chunks = chunk_text_llama_index(
    #     input_text=text,
    #     chunk_size=15,
    #     chunk_overlap=5
    # )
    # print("示例1 分割结果:", [c.text for c in chunks])
    # 示例 3: 从文件加载并分割（需确保文件存在）
    chunks = chunk_text_llama_index(
        input_text="../data/PGEssays/mit.txt",
        use_file_loader=True,
        file_extension=".txt",
        chunk_size=300
    )
    print("示例3 分割结果:", f"共生成 {len(chunks)} 个文本块")
    for i, chunk in enumerate(chunks):
        print(f"文本块 {i + 1}: {chunk.text}")
