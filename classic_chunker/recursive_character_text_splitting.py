from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def recursive_character_text_splitting(text, chunk_size=65, chunk_overlap=10):
    """
    使用递归字符文本分割器对文本进行分割。

    :param text: 待分割的文本
    :param chunk_size: 每个块的最大字符数，默认为65
    :param chunk_overlap: 块之间的重叠字符数，默认为0
    :return: 分割后的文档列表
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])


if __name__ == "__main__":
    # 示例使用
    text = """
    One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.
    
    Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.
    
    It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer. [1]
    """

    chunks = recursive_character_text_splitting(text, chunk_size=30)
    for chunk in chunks:
        print("分块结果:", chunk.page_content)