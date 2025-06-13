import json
import nltk
nltk.data.path.append('../models/nltk_data')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def dense_x_retrieval(tokenizer, model, text, title='', section='', target_size=256):
    """
    定义 dense_x_retrieval 函数，
    接收 tokenizer（分词器）、model（模型）、text（输入文本）、title（标题，默认为空字符串）、section（章节，默认为空字符串）和 target_size（目标大小，默认为 256）作为参数
    """
    title = title  # 将传入的 title 赋值给变量 title
    section = section  # 将传入的 section 赋值给变量 section
    target_size = target_size  # 将传入的 target_size 赋值给变量 target_size，并添加注释说明模型最大接受长度为 512

    full_segments = sent_tokenize(text)  # 使用 sent_tokenize 函数将输入文本按句子分割成一个列表
    merged_paragraphs = []  # 初始化一个空列表，用于存储合并后的段落
    current_paragraph = ""  # 初始化一个空字符串，用于存储当前正在合并的段落
    for paragraph in full_segments:
        # 遍历分割后的每个句子
        tmp_input_text = f"Title: {title}. Section: {section}. Content: {current_paragraph + ' ' + paragraph}"  # 构建一个临时输入文本，包含标题、章节和当前段落与新句子的组合

        if len((tokenizer(tmp_input_text, return_tensors="pt").input_ids)[0].tolist()) <= target_size:
            # 使用 tokenizer 对临时输入文本进行分词，并获取输入 ID 的长度，检查是否超过目标大小, 如果不超过目标大小，将新句子添加到当前段落中
            current_paragraph += ' ' + paragraph
        else:
            merged_paragraphs.append(current_paragraph)  # 如果超过目标大小，将当前段落添加到合并后的段落列表中
            current_paragraph = paragraph  # 重置当前段落为新句子

    if current_paragraph:
        # 如果当前段落不为空
        merged_paragraphs.append(current_paragraph)  # 将当前段落添加到合并后的段落列表中

    final_prop_list = []  # 初始化一个空列表，用于存储最终的命题列表

    for chunk in merged_paragraphs:
        # 遍历合并后的每个段落
        content = chunk  # 将当前段落赋值给变量 content

        input_text = f"Title: {title}. Section: {section}. Content: {content}"  # 构建输入文本，包含标题、章节和当前段落内容
        print(input_text)  # 打印输入文本
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # 使用 tokenizer 对输入文本进行分词，并获取输入 ID
        outputs = model.generate(input_ids.to(model.device), max_new_tokens=512).cpu()  # 使用模型对输入 ID 进行生成，设置最大新生成的 token 数为 512，并将结果移到 CPU 上

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  # 使用 tokenizer 对生成的输出 ID 进行解码，跳过特殊 token

        try:
            prop_list = json.loads(output_text)  # 尝试将解码后的输出文本解析为 JSON 列表
            final_prop_list = final_prop_list + prop_list  # 将解析后的列表添加到最终的命题列表中
        except:
            print("[ERROR] Failed to parse output text as JSON.")  # 如果解析失败，打印错误信息
        # print(json.dumps(prop_list, indent=2))

    return final_prop_list


if __name__ == '__main__':

    model_name = "../models/chentong00/propositionizer-wiki-flan-t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    title = "Leaning Tower of Pisa"
    section = ""
    text = "Prior to restoration work performed between 1990 and 2001, Leaning Tower of Pisa leaned at an angle of 5.5 degrees, but the tower now leans at about 3.99 degrees. This means the top of the tower is displaced horizontally 3.9 meters (12 ft 10 in) from the center."

    # 调用 dense_x_retrieval 函数
    result = dense_x_retrieval(tokenizer, model, text, title, section)
    print(result)