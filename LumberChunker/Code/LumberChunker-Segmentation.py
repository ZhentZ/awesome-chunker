import time  # 导入time模块，用于处理时间相关操作，如延时
import re  # 导入re模块，用于正则表达式匹配
import pandas as pd  # 导入pandas库，用于数据处理和分析
from tqdm import tqdm  # 导入tqdm库，用于在循环中显示进度条
import argparse  # 导入argparse模块，用于解析命令行参数
import sys  # 导入sys模块，用于与Python解释器进行交互，如退出程序
import os  # 导入os模块，用于与操作系统进行交互，如文件和目录操作

from langchain_community.chat_models import ChatOpenAI  # 从openai库导入OpenAI类，用于与OpenAI API交互


def create_directory(path):  # 定义创建目录的函数
    if not os.path.exists(path):  # 检查目录是否存在
        os.makedirs(path)  # 如果不存在，则创建目录
        print(f"Created directory: {path}")  # 打印创建目录的信息


# If using ChatGPT
client = ChatOpenAI(model_name="deepseek-v3-0324",
                    temperature=0.95,
                    openai_api_key="",
                    openai_api_base=""
                    )  # 创建OpenAI客户端实例，需插入OpenAI API密钥

# Debug 模式下的参数设置
DEBUG = True
if DEBUG:
    # 手动设置参数值
    sys.argv = [sys.argv[0],
                '--out_path', '../Example',
                '--model_type', 'deepseek-v3-0324',
                '--book_name', 'A_Christmas_Carol_-_Charles_Dickens']


# Argument parsing
parser = argparse.ArgumentParser(description="Process some text.")  # 创建命令行参数解析器，添加描述信息
parser.add_argument('--out_path', type=str, default='../Example',
                    help='Output directory path')  # 添加输出目录路径参数，默认值为当前目录
parser.add_argument('--model_type', type=str, required=True, default='deepseek-v3-0324',
                    help='deepseek (defaults to deepseek-v3-0324)')  # 添加模型类型参数，必填
parser.add_argument('--book_name', type=str, required=True, default='A_Christmas_Carol_-_Charles_Dickens',
                    help='Example: A_Christmas_Carol_-_Charles_Dickens')  # 添加书籍名称参数，必填

args = parser.parse_args()  # 解析命令行参数
out_path = args.out_path  # 获取输出目录路径
book_name = args.book_name  # 获取书籍名称
model_type = args.model_type  # 获取模型类型

# Ensure the output directory exists
create_directory(args.out_path)  # 确保输出目录存在

if model_type not in ["deepseek-v3-0324"]:  # 检查模型类型是否有效
    print("Choose Valid Model Type")  # 若无效，打印错误信息
    sys.exit(1)  # 退出程序，返回错误码1


# Count_Words idea is to approximate the number of tokens in the sentence. We are assuming 1 word ~ 1.2 Tokens
def count_words(input_string):  # 定义计算字符串中单词数量（近似为令牌数量）的函数
    words = input_string.split()  # 将字符串按空格分割成单词列表
    return round(1.2 * len(words))  # 返回单词数量乘以1.2并四舍五入后的结果


# Function to add IDs to each Dataframe Row
def add_ids(row):  # 定义为DataFrame的每一行添加ID的函数
    global current_id  # 声明使用全局变量current_id
    # Add ID to the chunk
    row['Chunk'] = f'ID {current_id}: {row["Chunk"]}'  # 在Chunk列的值前添加ID前缀
    current_id += 1  # 增加current_id的值
    return row  # 返回修改后的行


system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""  # 定义系统提示，用于指导LLM识别内容变化的段落


def LLM_prompt(model_type, user_prompt):  # 定义向LLM发送提示并获取响应的函数
    if model_type == "deepseek-v3-0324":  # 如果使用ChatGPT模型
        while True:  # 无限循环，直到成功获取响应
            try:
                # 调用 ChatOpenAI 的方法生成响应
                response = client.invoke(
                    input=[
                        {"role": "system", "content": system_prompt},  # 传入系统提示
                        {"role": "user", "content": user_prompt},  # 传入用户提示
                    ]
                )
                return response.content  # 返回生成内容的文本
            except Exception as e:  # 捕获异常
                if str(e) == "list index out of range":  # 如果异常信息为列表索引超出范围
                    print("GPT thinks prompt is unsafe")  # 打印提示信息
                    return "content_flag_increment"  # 返回标记，表示内容被标记为不安全
                else:
                    print(f"An error occurred: {e}. Retrying in 1 minute...")  # 打印错误信息
                    time.sleep(60)  # 等待1分钟后重试


dataset = pd.read_parquet(
    "../../data/GutenQA_paragraphs.parquet", engine="pyarrow")  # 从指定路径读取Parquet文件，使用pyarrow引擎
fileOut = f'{out_path}/Gemini_Chunks_-_{book_name}.xlsx'  # 定义输出文件的路径和名称

# Filter the DataFrame to show only rows with the specified book name
paragraph_chunks = dataset[dataset['Book Name'] == book_name].reset_index(drop=True)  # 过滤DataFrame，只保留指定书籍名称的行，并重置索引

# Check if the filtered DataFrame is empty
if paragraph_chunks.empty:  # 检查过滤后的DataFrame是否为空
    sys.exit("Choose a valid book name!")  # 若为空，打印错误信息并退出程序

id_chunks = paragraph_chunks['Chunk'].to_frame()  # 提取Chunk列，转换为DataFrame

# Initialize a global variable for current_id and Apply the function along the rows of the DataFrame
current_id = 0  # 初始化全局变量current_id为0
id_chunks = id_chunks.apply(add_ids, axis=1)  # 对id_chunks的每一行应用add_ids函数，添加ID前缀

chunk_number = 0  # 初始化块编号为0
i = 0  # 初始化计数器为0

new_id_list = []  # 初始化新ID列表

word_count_aux = []  # 初始化单词计数辅助列表
current_iteration = 0  # 初始化当前迭代次数为0

while chunk_number < len(id_chunks) - 5:  # 当块编号小于id_chunks长度减5时，继续循环
    word_count = 0  # 初始化单词计数为0
    i = 0  # 重置计数器为0
    while word_count < 550 and i + chunk_number < len(id_chunks) - 1:  # 当单词计数小于550且索引未超出范围时，继续循环
        i += 1  # 增加计数器的值
        final_document = "\n".join(
            f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))  # 拼接文档内容
        word_count = count_words(final_document)  # 计算文档的单词数量

    if (i == 1):  # 如果计数器为1
        final_document = "\n".join(
            f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))  # 拼接文档内容
    else:
        final_document = "\n".join(
            f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i - 1 + chunk_number))  # 拼接文档内容

    question = f"\nDocument:\n{final_document}"  # 构造问题，包含文档内容

    word_count = count_words(final_document)  # 计算文档的单词数量
    word_count_aux.append(word_count)  # 将单词数量添加到辅助列表中
    chunk_number = chunk_number + i - 1  # 更新块编号

    prompt = system_prompt + question  # 构造完整的提示，包含系统提示和问题
    gpt_output = LLM_prompt(model_type=model_type, user_prompt=prompt)  # 调用LLM_prompt函数，获取LLM的输出

    # For books where there is dubious content, Gemini refuses to run the prompt and returns mistake. This is to avoid being stalled here forever.
    if gpt_output == "content_flag_increment":  # 如果输出为内容标记增量
        chunk_number = chunk_number + 1  # 增加块编号
    else:
        pattern = r"Answer: ID \w+"  # 定义正则表达式模式，用于匹配答案格式
        match = re.search(pattern, gpt_output)  # 在输出中搜索匹配的模式

        if match == None:  # 如果未找到匹配项
            print("repeat this one")  # 打印提示信息
        else:
            gpt_output1 = match.group(0)  # 获取匹配的内容
            print(gpt_output1)  # 打印匹配的内容
            pattern = r'\d+'  # 定义正则表达式模式，用于匹配数字
            match = re.search(pattern, gpt_output1)  # 在匹配的内容中搜索数字
            chunk_number = int(match.group())  # 将数字转换为整数，更新块编号
            new_id_list.append(chunk_number)  # 将块编号添加到新ID列表中
            if (new_id_list[-1] == chunk_number):  # 如果新ID列表的最后一个元素等于当前块编号
                chunk_number = chunk_number + 1  # 增加块编号

# Add the last chunk to the list
new_id_list.append(len(id_chunks))  # 将id_chunks的长度添加到新ID列表中

# Remove IDs as they no longer make sense here.
id_chunks['Chunk'] = id_chunks['Chunk'].str.replace(r'^ID \d+:\s*', '', regex=True)  # 从Chunk列中移除ID前缀

# Create final dataframe from chunks
new_final_chunks = []  # 初始化最终块列表
chapter_chunk = []  # 初始化章节块列表
for i in range(len(new_id_list)):  # 遍历新ID列表

    # Calculate the start and end indices of each chunk
    start_idx = new_id_list[i - 1] if i > 0 else 0  # 计算当前块的起始索引
    end_idx = new_id_list[i]  # 计算当前块的结束索引
    new_final_chunks.append('\n'.join(id_chunks.iloc[start_idx: end_idx, 0]))  # 将当前块的内容拼接并添加到最终块列表中

    # When building final Dataframe, sometimes text from different chapters is concatenated. When this happens we update the Chapter column accordingly.
    if (paragraph_chunks["Chapter"][start_idx] != paragraph_chunks["Chapter"][end_idx - 1]):  # 如果起始章节和结束章节不同
        chapter_chunk.append(
            f"{paragraph_chunks['Chapter'][start_idx]} and {paragraph_chunks['Chapter'][end_idx - 1]}")  # 将两个章节名称拼接并添加到章节块列表中
    else:
        chapter_chunk.append(paragraph_chunks['Chapter'][start_idx])  # 将起始章节名称添加到章节块列表中

# Write new Chunks Dataframe
df_new_final_chunks = pd.DataFrame({'Chapter': chapter_chunk, 'Chunk': new_final_chunks})  # 创建最终的DataFrame，包含章节和块内容
df_new_final_chunks.to_excel(fileOut, index=False)  # 将DataFrame保存为Excel文件，不包含索引
print(f"{book_name} Completed!")  # 打印完成信息
