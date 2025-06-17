from utils.ppl_calculate import Chunking
from typing import List, Dict
import re
import math
from nltk.tokenize import sent_tokenize
import jieba
import torch
import time


# 定义一个函数，用于根据标点符号分割文本
def split_text_by_punctuation(text, language):

    if language == 'zh':
        sentences = jieba.cut(text, cut_all=False)  # 使用 jieba 进行分词
        sentences_list = list(sentences)  # 将分词结果转换为列表
        sentences = []  # 初始化一个空列表，用于存储分割后的句子
        temp_sentence = ""  # 初始化一个空字符串，用于临时存储句子

        for word in sentences_list:  # 遍历分词结果
            if word in ["。", "！", "？", "；"]:  # 如果当前词是中文标点符号
                sentences.append(temp_sentence.strip() + word)  # 将临时句子添加到句子列表中
                temp_sentence = ""  # 清空临时句子
            else:
                temp_sentence += word  # 否则，将当前词添加到临时句子中

        # 如果临时句子不为空
        if temp_sentence:
            sentences.append(temp_sentence.strip())  # 将临时句子添加到句子列表中

        return sentences  # 返回分割后的句子列表
    else:
        full_segments = sent_tokenize(text)  # 使用 sent_tokenize 函数进行英文句子分割
        ret = []  # 初始化一个空列表，用于存储最终的句子列表
        # 遍历分割后的句子
        for item in full_segments:
            item_l = item.strip().split(' ')  # 去除句子前后的空格，并按空格分割成单词列表

            # 如果单词列表长度大于 512
            if len(item_l) > 512:
                # 如果单词列表长度大于 1024
                if len(item_l) > 1024:
                    item = ' '.join(item_l[:256]) + "..."  # 取前 256 个单词，并添加省略号
                else:
                    item = ' '.join(item_l[:512]) + "..."  # 取前 512 个单词，并添加省略号

            ret.append(item)  # 将处理后的句子添加到最终列表中

        return ret


# 定义一个函数，用于查找列表中的局部最小值索引
def find_minima(values, threshold):

    minima_indices = []  # 初始化一个空列表，用于存储局部最小值的索引
    # 遍历列表，从第二个元素到倒数第二个元素
    for i in range(1, len(values) - 1):
        # 如果当前元素小于前一个元素且小于后一个元素
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            # 如果当前元素与前一个元素的差值大于等于阈值，或者与后一个元素的差值大于等于阈值
            if (values[i - 1] - values[i] >= threshold) or (values[i + 1] - values[i] >= threshold):
                minima_indices.append(i)  # 将当前元素的索引添加到局部最小值索引列表中
        # 如果当前元素小于前一个元素且等于后一个元素
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            # 如果当前元素与前一个元素的差值大于等于阈值
            if values[i - 1] - values[i] >= threshold:
                minima_indices.append(i)  # 将当前元素的索引添加到局部最小值索引列表中

    return minima_indices  # 返回局部最小值的索引列表


# 定义一个函数，用于动态查找列表中的局部最小值索引
def find_minima_dynamic(values, threshold, threshold_zlist):

    minima_indices = []  # 初始化一个空列表，用于存储局部最小值的索引
    # 遍历列表，从第二个元素到倒数第二个元素
    for i in range(1, len(values) - 1):
        # 如果当前元素小于前一个元素且小于后一个元素
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            # 如果当前元素与前一个元素的差值大于等于阈值，或者与后一个元素的差值大于等于阈值
            if (values[i - 1] - values[i] >= threshold) or (values[i + 1] - values[i] >= threshold):
                minima_indices.append(i)  # 将当前元素的索引添加到局部最小值索引列表中
                threshold_zlist.append(min(values[i - 1] - values[i], values[i + 1] - values[i]))  # 将当前元素与前一个元素和后一个元素差值的最小值添加到阈值列表中

        # 如果当前元素小于前一个元素且等于后一个元素
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            # 如果当前元素与前一个元素的差值大于等于阈值
            if values[i - 1] - values[i] >= threshold:
                minima_indices.append(i)  # 将当前元素的索引添加到局部最小值索引列表中
                threshold_zlist.append(values[i - 1] - values[i])  # 将当前元素与前一个元素的差值添加到阈值列表中

        # 如果阈值列表的长度大于等于 100
        if len(threshold_zlist) >= 100:
            last_ten = threshold_zlist  # [-100:]  # 取阈值列表的所有元素
            avg = min(last_ten)  # 计算阈值列表的最小值
            threshold = avg  # 更新阈值为最小值

    return minima_indices, threshold, threshold_zlist  # 返回局部最小值的索引列表、更新后的阈值和阈值列表


# 定义一个函数，用于使用困惑度进行文本分块（重叠方式）
def extract_by_html2text_db_chongdie(sub_text, model, tokenizer, threshold, language='zh') -> List[str]:

    temp_para = sub_text  # 将输入文本赋值给临时变量
    # 如果语言为中文
    if language == 'zh':
        # 去除文本中的空白字符
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)
        # 去除文本中的连续空格
        # cleaned_text = re.sub(r'  ', '', text)
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 如果语言不是中文
    else:
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 使用 split_text_by_punctuation 函数分割文本
    segments = split_text_by_punctuation(cleaned_text, language)
    # 过滤掉分割后的空句子
    segments = [item for item in segments if item.strip()]
    # 初始化 Chunking 类的实例
    ch = Chunking(model, tokenizer)
    # 初始化一个空列表，用于存储每个句子的长度
    len_sentences = []
    # 初始化一个空的张量，用于存储输入的 token ID
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 初始化一个空的张量，用于存储注意力掩码
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 遍历分割后的句子
    for context in segments:
        # 使用分词器对句子进行分词
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        # 将分词后的 token ID 移动到模型所在的设备上
        input_id = tokenized_text["input_ids"].to(model.device)
        # 将当前句子的 token ID 拼接到底层的输入 token ID 中
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        # 记录当前句子的长度
        len_sentences.append(input_id.shape[1])
        # 将分词后的注意力掩码移动到模型所在的设备上
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        # 将当前句子的注意力掩码拼接到底层的注意力掩码中
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
    # 调用 Chunking 类的 get_ppl_batch 方法，计算困惑度损失和历史键值对
    loss, past_key_values = ch.get_ppl_batch(
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    # 初始化一个空列表，用于存储每个句子的平均困惑度
    first_cluster_ppl = []
    # 初始化索引为 0
    index = 0
    # 遍历每个句子的长度
    for i in range(len(len_sentences)):
        # 如果是第一个句子
        if i == 0:
            # 计算第一个句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[0:len_sentences[i] - 1].mean().item())
            # 更新索引
            index += len_sentences[i] - 1
        # 如果不是第一个句子
        else:
            # 计算当前句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[index:index + len_sentences[i]].mean().item())
            # 更新索引
            index += len_sentences[i]
    # 调用 find_minima 函数，查找平均困惑度列表中的局部最小值索引
    minima_indices = find_minima(first_cluster_ppl, threshold)
    # 初始化一个空列表，用于存储分块的索引
    first_chunk_indices = []
    # 初始化一个空列表，用于存储分块的句子
    first_chunk_sentences = []
    # 定义分块的分割点，包括起始点、局部最小值索引和结束点
    split_points = [0] + minima_indices + [len(first_cluster_ppl) - 1]
    # 遍历分割点
    for i in range(len(split_points) - 1):
        # 初始化一个空列表，用于存储当前分块的索引
        tmp_index = []
        # 初始化一个空列表，用于存储当前分块的句子
        tmp_sentence = []
        # 遍历当前分割点之间的索引
        for sp_index in range(split_points[i], split_points[i + 1] + 1):
            # 将当前索引添加到当前分块的索引列表中
            tmp_index.append(sp_index)
            # 将当前索引对应的句子添加到当前分块的句子列表中
            tmp_sentence.append(segments[sp_index])
        # 将当前分块的索引列表添加到分块索引列表中
        first_chunk_indices.append(tmp_index)
        # 将当前分块的句子列表添加到分块句子列表中
        first_chunk_sentences.append(tmp_sentence)
    # 初始化一个空列表，用于存储最终的分块结果
    final_chunks = []
    # 遍历分块句子列表
    for sent_list in first_chunk_sentences:
        # 将每个分块的句子拼接成一个字符串，并添加到最终分块结果列表中
        final_chunks.append(''.join(sent_list))
    # 打印分块的索引列表
    print('111', first_chunk_indices)
    # 返回最终的分块结果列表
    return final_chunks


# 定义一个函数，用于使用困惑度进行文本分块（非列表方式）
def extract_by_html2text_db_nolist(sub_text, model, tokenizer, threshold, language='zh') -> List[str]:
    # 将输入文本赋值给临时变量
    temp_para = sub_text
    # 如果语言为中文
    if language == 'zh':
        # 去除文本中的空白字符
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)
        # 去除文本中的连续空格
        # cleaned_text = re.sub(r'  ', '', text)
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 如果语言不是中文
    else:
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 使用 split_text_by_punctuation 函数分割文本
    segments = split_text_by_punctuation(cleaned_text, language)
    # 过滤掉分割后的空句子
    segments = [item for item in segments if item.strip()]
    # 初始化 Chunking 类的实例
    ch = Chunking(model, tokenizer)
    # 初始化一个空列表，用于存储每个句子的长度
    len_sentences = []
    # 初始化一个空的张量，用于存储输入的 token ID
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 初始化一个空的张量，用于存储注意力掩码
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 遍历分割后的句子
    for context in segments:
        # 使用分词器对句子进行分词
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        # 将分词后的 token ID 移动到模型所在的设备上
        input_id = tokenized_text["input_ids"].to(model.device)
        # 将当前句子的 token ID 拼接到底层的输入 token ID 中
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        # 记录当前句子的长度
        len_sentences.append(input_id.shape[1])
        # 将分词后的注意力掩码移动到模型所在的设备上
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        # 将当前句子的注意力掩码拼接到底层的注意力掩码中
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
    # 调用 Chunking 类的 get_ppl_batch 方法，计算困惑度损失和历史键值对
    loss, past_key_values = ch.get_ppl_batch(
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    # 初始化一个空列表，用于存储每个句子的平均困惑度
    first_cluster_ppl = []
    # 初始化索引为 0
    index = 0
    # 遍历每个句子的长度
    for i in range(len(len_sentences)):
        # 如果是第一个句子
        if i == 0:
            # 计算第一个句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[0:len_sentences[i] - 1].mean().item())
            # 更新索引
            index += len_sentences[i] - 1
        # 如果不是第一个句子
        else:
            # 计算当前句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[index:index + len_sentences[i]].mean().item())
            # 更新索引
            index += len_sentences[i]
    # 调用 find_minima 函数，查找平均困惑度列表中的局部最小值索引
    minima_indices = find_minima(first_cluster_ppl, threshold)
    # 初始化一个空列表，用于存储分块的索引
    first_chunk_indices = []
    # 初始化一个空列表，用于存储分块的句子
    first_chunk_sentences = []
    # 定义分块的分割点，包括起始点、局部最小值索引和结束点
    split_points = [0] + minima_indices + [len(first_cluster_ppl) - 1]
    # 遍历分割点
    for i in range(len(split_points) - 1):
        # 初始化一个空列表，用于存储当前分块的索引
        tmp_index = []
        # 初始化一个空列表，用于存储当前分块的句子
        tmp_sentence = []
        # 如果是第一个分块
        if i == 0:
            # 将第一个句子的索引添加到当前分块的索引列表中
            tmp_index.append(0)
            # 将第一个句子添加到当前分块的句子列表中
            tmp_sentence.append(segments[0])
        # 遍历当前分割点之间的索引
        for sp_index in range(split_points[i] + 1, split_points[i + 1] + 1):
            # 将当前索引添加到当前分块的索引列表中
            tmp_index.append(sp_index)
            # 将当前索引对应的句子添加到当前分块的句子列表中
            tmp_sentence.append(segments[sp_index])
        # 将当前分块的索引列表添加到分块索引列表中
        first_chunk_indices.append(tmp_index)
        # 将当前分块的句子列表添加到分块句子列表中
        first_chunk_sentences.append(tmp_sentence)
    # 初始化一个空列表，用于存储最终的分块结果
    final_chunks = []
    # 遍历分块句子列表
    for sent_list in first_chunk_sentences:
        # 将每个分块的句子拼接成一个字符串，并添加到最终分块结果列表中
        final_chunks.append(''.join(sent_list))
    # 打印分块的索引列表
    print('111', first_chunk_indices)
    # 返回最终的分块结果列表
    return final_chunks


# 定义一个函数，用于使用困惑度进行文本分块（动态阈值方式）
def extract_by_html2text_db_dynamic(sub_text, model, tokenizer, threshold, threshold_zlist, language='zh') -> List[str]:
    # 将输入文本赋值给临时变量
    temp_para = sub_text
    # 如果语言为中文
    if language == 'zh':
        # 去除文本中的空白字符
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)
        # 去除文本中的连续空格
        # cleaned_text = re.sub(r'  ', '', text)
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 如果语言不是中文
    else:
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 使用 split_text_by_punctuation 函数分割文本
    segments = split_text_by_punctuation(cleaned_text, language)
    # 过滤掉分割后的空句子
    segments = [item for item in segments if item.strip()]
    # 初始化 Chunking 类的实例
    ch = Chunking(model, tokenizer)
    # 初始化一个空列表，用于存储每个句子的长度
    len_sentences = []
    # 初始化一个空的张量，用于存储输入的 token ID
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 初始化一个空的张量，用于存储注意力掩码
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 遍历分割后的句子
    for context in segments:
        # 使用分词器对句子进行分词
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        # 将分词后的 token ID 移动到模型所在的设备上
        input_id = tokenized_text["input_ids"].to(model.device)
        # 将当前句子的 token ID 拼接到底层的输入 token ID 中
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        # 记录当前句子的长度
        len_sentences.append(input_id.shape[1])
        # 将分词后的注意力掩码移动到模型所在的设备上
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        # 将当前句子的注意力掩码拼接到底层的注意力掩码中
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
    # 调用 Chunking 类的 get_ppl_batch 方法，计算困惑度损失和历史键值对
    loss, past_key_values = ch.get_ppl_batch(
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    # 初始化一个空列表，用于存储每个句子的平均困惑度
    first_cluster_ppl = []
    # 初始化索引为 0
    index = 0
    # 遍历每个句子的长度
    for i in range(len(len_sentences)):
        # 如果是第一个句子
        if i == 0:
            # 计算第一个句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[0:len_sentences[i] - 1].mean().item())
            # 更新索引
            index += len_sentences[i] - 1
        # 如果不是第一个句子
        else:
            # 计算当前句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[index:index + len_sentences[i]].mean().item())
            # 更新索引
            index += len_sentences[i]
    # 调用 find_minima_dynamic 函数，动态查找平均困惑度列表中的局部最小值索引
    minima_indices, threshold, threshold_zlist = find_minima_dynamic(first_cluster_ppl, threshold, threshold_zlist)
    # 初始化一个空列表，用于存储分块的索引
    first_chunk_indices = []
    # 初始化一个空列表，用于存储分块的句子
    first_chunk_sentences = []
    # 定义分块的分割点，包括起始点、局部最小值索引和结束点
    split_points = [0] + minima_indices + [len(first_cluster_ppl) - 1]
    # 遍历分割点
    for i in range(len(split_points) - 1):
        # 初始化一个空列表，用于存储当前分块的索引
        tmp_index = []
        # 初始化一个空列表，用于存储当前分块的句子
        tmp_sentence = []
        # 如果是第一个分块
        if i == 0:
            # 将第一个句子的索引添加到当前分块的索引列表中
            tmp_index.append(0)
            # 将第一个句子添加到当前分块的句子列表中
            tmp_sentence.append(segments[0])
        # 遍历当前分割点之间的索引
        for sp_index in range(split_points[i] + 1, split_points[i + 1] + 1):
            # 将当前索引添加到当前分块的索引列表中
            tmp_index.append(sp_index)
            # 将当前索引对应的句子添加到当前分块的句子列表中
            tmp_sentence.append(segments[sp_index])
        # 将当前分块的索引列表添加到分块索引列表中
        first_chunk_indices.append(tmp_index)
        # 将当前分块的句子列表添加到分块句子列表中
        first_chunk_sentences.append(tmp_sentence)
    # 初始化一个空列表，用于存储最终的分块结果
    final_chunks = []
    # 遍历分块句子列表
    for sent_list in first_chunk_sentences:
        # 将每个分块的句子拼接成一个字符串，并添加到最终分块结果列表中
        final_chunks.append(''.join(sent_list))
    # 打印分块的索引列表
    print('111', first_chunk_indices)
    # 返回最终的分块结果列表、更新后的阈值和阈值列表
    return final_chunks, threshold, threshold_zlist


# 定义一个函数，用于使用困惑度进行文本分块（动态批量方式）
def extract_by_html2text_db_dynamic_batch(sub_text, model, tokenizer, threshold, threshold_zlist, language='zh',
                                          past_key_values=None) -> List[str]:  # 不重叠
    # 将输入文本赋值给临时变量
    temp_para = sub_text
    # 如果语言为中文
    if language == 'zh':
        # 去除文本中的空白字符
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)
        # 去除文本中的连续空格
        # cleaned_text = re.sub(r'  ', '', text)
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 如果语言不是中文
    else:
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 使用 split_text_by_punctuation 函数分割文本
    segments = split_text_by_punctuation(cleaned_text, language)
    # 过滤掉分割后的空句子
    segments = [item for item in segments if item.strip()]
    # 初始化 Chunking 类的实例
    ch = Chunking(model, tokenizer)
    # 初始化一个空列表，用于存储每个句子的长度
    len_sentences = []
    # 初始化一个空的张量，用于存储输入的 token ID
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 初始化一个空的张量，用于存储注意力掩码
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 遍历分割后的句子
    for context in segments:
        # 使用分词器对句子进行分词
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        # 将分词后的 token ID 移动到模型所在的设备上
        input_id = tokenized_text["input_ids"].to(model.device)
        # 将当前句子的 token ID 拼接到底层的输入 token ID 中
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        # 记录当前句子的长度
        len_sentences.append(input_id.shape[1])
        # 将分词后的注意力掩码移动到模型所在的设备上
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        # 将当前句子的注意力掩码拼接到底层的注意力掩码中
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
    # 定义批量大小
    batch_size = 4096  # 6000
    # 计算总批次数
    total_batches = math.ceil(input_ids.shape[1] / batch_size)
    # 初始化一个空的张量，用于存储损失
    loss = torch.tensor([], device=model.device, dtype=torch.long)
    # 遍历每个批次
    for i in range(total_batches):
        # 计算当前批次的起始位置
        start = i * batch_size
        # 计算当前批次的结束位置
        end = start + batch_size
        # 截取当前批次的输入 token ID
        input_ids_tmp = input_ids[:, start:end]
        # 截取当前批次的注意力掩码
        attention_mask_tmp = attention_mask[:, :end]
        # 在当前批次的输入 token ID 前添加一个空格的 token ID
        input_ids_tmp = torch.cat(
            [tokenizer(' ', return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device),
             input_ids_tmp], dim=-1)
        # 在当前批次的注意力掩码后添加一个全为 1 的张量
        attention_mask_tmp = torch.cat(
            [attention_mask_tmp, torch.ones((1, i + 1), device=model.device, dtype=torch.long)], dim=-1)
        # 获取当前批次的输入 token ID 的长度
        size = input_ids_tmp.shape[1]
        # 如果注意力掩码的长度大于 24576
        if attention_mask_tmp.shape[1] > 24576:  # 72000
            # 更新历史键值对
            past_key_values = [
                [k[:, :, size + 1:], v[:, :, size + 1:]]
                for k, v in past_key_values
            ]
            # 截取注意力掩码的一部分
            attention_mask_tmp = attention_mask_tmp[:,
                                 attention_mask_tmp.shape[1] - size - past_key_values[0][0].shape[2]:]
        # 调用 Chunking 类的 get_ppl_batch 方法，计算当前批次的损失和历史键值对
        loss_tmp, past_key_values = ch.get_ppl_batch(
            input_ids_tmp,
            attention_mask_tmp,
            past_key_values=past_key_values,
            return_kv=True
        )
        # 将当前批次的损失拼接到底层的损失中
        loss = torch.cat([loss, loss_tmp], dim=-1)
    # 初始化一个空列表，用于存储每个句子的平均困惑度
    first_cluster_ppl = []
    # 初始化索引为 0
    index = 0
    # 遍历每个句子的长度
    for i in range(len(len_sentences)):
        # 如果是第一个句子
        if i == 0:
            # 计算第一个句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[1:len_sentences[i]].mean().item())
        # 如果不是第一个句子
        else:
            # 计算当前句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[index:index + len_sentences[i]].mean().item())
        # 更新索引
        index += len_sentences[i]
    # 调用 find_minima_dynamic 函数，动态查找平均困惑度列表中的局部最小值索引
    minima_indices, threshold, threshold_zlist = find_minima_dynamic(first_cluster_ppl, threshold, threshold_zlist)
    # 初始化一个空列表，用于存储分块的索引
    first_chunk_indices = []
    # 初始化一个空列表，用于存储分块的句子
    first_chunk_sentences = []
    # 定义分块的分割点，包括起始点、局部最小值索引和结束点
    split_points = [0] + minima_indices + [len(first_cluster_ppl) - 1]
    # 遍历分割点
    for i in range(len(split_points) - 1):
        # 初始化一个空列表，用于存储当前分块的索引
        tmp_index = []
        # 初始化一个空列表，用于存储当前分块的句子
        tmp_sentence = []
        # 如果是第一个分块
        if i == 0:
            # 将第一个句子的索引添加到当前分块的索引列表中
            tmp_index.append(0)
            # 将第一个句子添加到当前分块的句子列表中
            tmp_sentence.append(segments[0])
        # 遍历当前分割点之间的索引
        for sp_index in range(split_points[i] + 1, split_points[i + 1] + 1):
            # 将当前索引添加到当前分块的索引列表中
            tmp_index.append(sp_index)
            # 将当前索引对应的句子添加到当前分块的句子列表中
            tmp_sentence.append(segments[sp_index])
        # 将当前分块的索引列表添加到分块索引列表中
        first_chunk_indices.append(tmp_index)
        # 将当前分块的句子列表添加到分块句子列表中
        first_chunk_sentences.append(tmp_sentence)
    # 初始化一个空列表，用于存储最终的分块结果
    final_chunks = []
    # 遍历分块句子列表
    for sent_list in first_chunk_sentences:
        # 将每个分块的句子拼接成一个字符串，并添加到最终分块结果列表中
        final_chunks.append(''.join(sent_list))
    # 打印分块的索引列表
    print('111', first_chunk_indices)
    # 返回最终的分块结果列表、更新后的阈值和阈值列表
    return final_chunks, threshold, threshold_zlist


# 定义一个函数，用于使用困惑度进行文本分块（基准方式）
def extract_by_html2text_db_bench(sub_text, model, tokenizer, threshold, language='zh', batch_size=4096,
                                  max_txt_size=9000, past_key_values=None) -> List[str]:
    # 将输入文本赋值给临时变量
    temp_para = sub_text
    # 如果语言为中文
    if language == 'zh':
        # 去除文本中的空白字符
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)
        # 去除文本中的连续空格
        # cleaned_text = re.sub(r'  ', '', text)
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 如果语言不是中文
    else:
        # 直接使用原始文本作为清理后的文本
        cleaned_text = temp_para
    # 使用 split_text_by_punctuation 函数分割文本
    segments = split_text_by_punctuation(cleaned_text, language)
    # 过滤掉分割后的空句子
    segments = [item for item in segments if item.strip()]
    # 初始化 Chunking 类的实例
    ch = Chunking(model, tokenizer)
    # 初始化一个空列表，用于存储每个句子的长度
    len_sentences = []
    # 初始化一个空的张量，用于存储输入的 token ID
    input_ids = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 初始化一个空的张量，用于存储注意力掩码
    attention_mask = torch.tensor([[]], device=model.device, dtype=torch.long)
    # 遍历分割后的句子
    for context in segments:
        # 使用分词器对句子进行分词
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        # 将分词后的 token ID 移动到模型所在的设备上
        input_id = tokenized_text["input_ids"].to(model.device)
        # 将当前句子的 token ID 拼接到底层的输入 token ID 中
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        # 记录当前句子的长度
        len_sentences.append(input_id.shape[1])
        # 将分词后的注意力掩码移动到模型所在的设备上
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        # 将当前句子的注意力掩码拼接到底层的注意力掩码中
        attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
    # 定义批量大小
    batch_size = batch_size
    # 计算总批次数
    total_batches = math.ceil(input_ids.shape[1] / batch_size)
    # 打印输入 token ID 的长度
    print('111', input_ids.shape[1])
    # 初始化一个空的张量，用于存储损失
    loss = torch.tensor([], device=model.device, dtype=torch.long)
    # 遍历每个批次
    for i in range(total_batches):
        # 计算当前批次的起始位置
        start = i * batch_size
        # 计算当前批次的结束位置
        end = start + batch_size
        # 截取当前批次的输入 token ID
        input_ids_tmp = input_ids[:, start:end]
        # 截取当前批次的注意力掩码
        attention_mask_tmp = attention_mask[:, :end]
        # 在当前批次的输入 token ID 前添加一个空格的 token ID
        input_ids_tmp = torch.cat(
            [tokenizer(' ', return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device),
             input_ids_tmp], dim=-1)
        # 在当前批次的注意力掩码后添加一个全为 1 的张量
        attention_mask_tmp = torch.cat(
            [attention_mask_tmp, torch.ones((1, i + 1), device=model.device, dtype=torch.long)], dim=-1)
        # 获取当前批次的输入 token ID 的长度
        size = input_ids_tmp.shape[1]
        # 如果注意力掩码的长度大于最大文本大小
        if attention_mask_tmp.shape[1] > max_txt_size:
            # 更新历史键值对
            past_key_values = [
                [k[:, :, size + 1:], v[:, :, size + 1:]]
                for k, v in past_key_values
            ]
            # 截取注意力掩码的一部分
            attention_mask_tmp = attention_mask_tmp[:,
                                 attention_mask_tmp.shape[1] - size - past_key_values[0][0].shape[2]:]
        # 调用 Chunking 类的 get_ppl_batch 方法，计算当前批次的损失和历史键值对
        loss_tmp, past_key_values = ch.get_ppl_batch(
            input_ids_tmp,
            attention_mask_tmp,
            past_key_values=past_key_values,
            return_kv=True
        )
        # 将当前批次的损失拼接到底层的损失中
        loss = torch.cat([loss, loss_tmp], dim=-1)
    # 初始化一个空列表，用于存储每个句子的平均困惑度
    first_cluster_ppl = []
    # 初始化索引为 0
    index = 0
    # 遍历每个句子的长度
    for i in range(len(len_sentences)):
        # 如果是第一个句子
        if i == 0:
            # 计算第一个句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[1:len_sentences[i]].mean().item())
        # 如果不是第一个句子
        else:
            # 计算当前句子的平均困惑度，并添加到列表中
            first_cluster_ppl.append(loss[index:index + len_sentences[i]].mean().item())
        # 更新索引
        index += len_sentences[i]
    # 调用 find_minima 函数，查找平均困惑度列表中的局部最小值索引
    minima_indices = find_minima(first_cluster_ppl, threshold)
    # 初始化一个空列表，用于存储分块的索引
    first_chunk_indices = []
    # 初始化一个空列表，用于存储分块的句子
    first_chunk_sentences = []
    # 定义分块的分割点，包括起始点、局部最小值索引和结束点
    split_points = [0] + minima_indices + [len(first_cluster_ppl) - 1]
    # 遍历分割点
    for i in range(len(split_points) - 1):
        # 初始化一个空列表，用于存储当前分块的索引
        tmp_index = []
        # 初始化一个空列表，用于存储当前分块的句子
        tmp_sentence = []
        # 如果是第一个分块
        if i == 0:
            # 将第一个句子的索引添加到当前分块的索引列表中
            tmp_index.append(0)
            # 将第一个句子添加到当前分块的句子列表中
            tmp_sentence.append(segments[0])
        # 遍历当前分割点之间的索引
        for sp_index in range(split_points[i] + 1, split_points[i + 1] + 1):
            # 将当前索引添加到当前分块的索引列表中
            tmp_index.append(sp_index)
            # 将当前索引对应的句子添加到当前分块的句子列表中
            tmp_sentence.append(segments[sp_index])
        # 将当前分块的索引列表添加到分块索引列表中
        first_chunk_indices.append(tmp_index)
        # 将当前分块的句子列表添加到分块句子列表中
        first_chunk_sentences.append(tmp_sentence)
    # 初始化一个空列表，用于存储最终的分块结果
    final_chunks = []
    # 遍历分块句子列表
    for sent_list in first_chunk_sentences:
        # 将每个分块的句子拼接成一个字符串，并添加到最终分块结果列表中
        final_chunks.append(''.join(sent_list))
    # 打印分块的索引列表
    print('111', first_chunk_indices)
    # 返回最终的分块结果列表
    return final_chunks


# 定义一个函数，用于使用困惑度进行文本分块（综合方式）
def llm_chunker_ppl(sub_text, model, tokenizer, threshold, language='zh', batch_size=4096, max_txt_size=9000,
                    dynamic_merge='no', target_size=200) -> List[str]:
    # 记录开始时间
    start_time = time.time()
    # 如果语言为英文
    if language == 'en':
        # 计算文本的单词数量
        txt_length = len(sub_text.split())
    # 如果语言不是英文
    else:
        # 计算文本的字符数量
        txt_length = len(sub_text)
    # 如果文本长度小于等于 4096
    if txt_length <= 4096:
        # 调用 extract_by_html2text_db_nolist 函数进行文本分块
        new_final_chunks = extract_by_html2text_db_nolist(sub_text, model, tokenizer, threshold, language)
    # 如果文本长度大于 4096
    else:
        # 调用 extract_by_html2text_db_bench 函数进行文本分块
        new_final_chunks = extract_by_html2text_db_bench(sub_text, model, tokenizer, threshold, language, batch_size,
                                                         max_txt_size)
    # 如果需要动态合并分块
    if dynamic_merge != 'no':
        # 初始化一个空列表，用于存储合并后的分块
        merged_paragraphs = []
        # 初始化一个空字符串，用于临时存储当前合并的分块
        current_paragraph = ""
        # 如果语言为英文
        if language == 'en':
            # 遍历分块结果
            for paragraph in new_final_chunks:
                # 检查添加新分块到当前分块是否超过目标大小
                if len(current_paragraph.split()) + len(paragraph.split()) <= target_size:
                    # 如果不超过，将新分块添加到当前分块中
                    current_paragraph += ' ' + paragraph
                else:
                    # 如果超过，将当前分块添加到合并后的分块列表中
                    merged_paragraphs.append(current_paragraph)
                    # 将当前分块重置为新分块
                    current_paragraph = paragraph
            # 如果当前分块不为空
            if current_paragraph:
                # 将当前分块添加到合并后的分块列表中
                merged_paragraphs.append(current_paragraph)
        # 如果语言不是英文
        else:
            # 遍历分块结果
            for paragraph in new_final_chunks:
                # 检查添加新分块到当前分块是否超过目标大小
                if len(current_paragraph) + len(paragraph) <= target_size:
                    # 如果不超过，将新分块添加到当前分块中
                    current_paragraph += paragraph
                else:
                    # 如果超过，将当前分块添加到合并后的分块列表中
                    merged_paragraphs.append(current_paragraph)
                    # 将当前分块重置为新分块
                    current_paragraph = paragraph
            # 如果当前分块不为空
            if current_paragraph:
                # 将当前分块添加到合并后的分块列表中
                merged_paragraphs.append(current_paragraph)
    # 如果不需要动态合并分块
    else:
        # 直接将分块结果赋值给合并后的分块列表
        merged_paragraphs = new_final_chunks
    # 记录结束时间
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - start_time
    # 这里原代码没有返回值，可根据需求添加返回值，例如返回合并后的分块列表
    return merged_paragraphs
