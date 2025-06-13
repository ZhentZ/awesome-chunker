from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.docstore.document import Document


def split_markdown_text(markdown_text: str, chunk_size: int = 40, chunk_overlap: int = 0):
    """
    使用 MarkdownTextSplitter 分割 Markdown 文档。

    :param markdown_text: 要分割的 Markdown 文本
    :param chunk_size: 每个块的最大字符数
    :param chunk_overlap: 块之间的重叠字符数
    :return: 分割后的块列表
    """
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([markdown_text])
    return chunks


def split_python_code(python_text: str, chunk_size: int = 100, chunk_overlap: int = 0):
    """
    使用 PythonCodeTextSplitter 分割 Python 代码文本。

    :param python_text: 要分割的 Python 代码文本
    :param chunk_size: 每个块的最大字符数
    :param chunk_overlap: 块之间的重叠字符数
    :return: 分割后的块列表
    """
    python_splitter = PythonCodeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = python_splitter.create_documents([python_text])
    return chunks


def split_javascript_code(javascript_text: str, chunk_size: int = 65, chunk_overlap: int = 0):
    """
    使用 RecursiveCharacterTextSplitter 分割 JavaScript 代码文本。

    :param javascript_text: 要分割的 JavaScript 代码文本
    :param chunk_size: 每个块的最大字符数
    :param chunk_overlap: 块之间的重叠字符数
    :return: 分割后的块列表
    """
    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = js_splitter.create_documents([javascript_text])
    return chunks


if __name__ == "__main__":
    # 示例调用
    javascript_text = """
    // Function is called, the return value will end up in x
    let x = myFunction(4, 3);

    function myFunction(a, b) {
    // Function returns the product of a and b
      return a * b;
    }
    """

    chunks = split_javascript_code(javascript_text, chunk_size=65, chunk_overlap=0)
    for chunk in chunks:
        print(chunk)