import time
import re
import pandas as pd
import sys
import os
from langchain_community.chat_models import ChatOpenAI


class LumberChunker:
    def __init__(self, model_type, book_name):
        self.model_type = model_type
        self.book_name = book_name
        self.current_id = 0
        self.client = ChatOpenAI(model_name=model_type,
                                 temperature=0.95,
                                 openai_api_key="",
                                 openai_api_base="")
        self.system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

        Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

        Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

        Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

    def count_words(self, input_string):
        words = input_string.split()
        return round(1.2 * len(words))

    def add_ids(self, row):
        row['Chunk'] = f'ID {self.current_id}: {row["Chunk"]}'
        self.current_id += 1
        return row

    def LLM_prompt(self, user_prompt):
        while True:
            try:
                response = self.client.invoke(
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                return response.content
            except Exception as e:
                if str(e) == "list index out of range":
                    print("GPT thinks prompt is unsafe")
                    return "content_flag_increment"
                else:
                    print(f"An error occurred: {e}. Retrying in 1 minute...")
                    time.sleep(60)

    def chunk_book(self, input_path, output_path):
        self.create_directory(output_path)
        dataset = pd.read_parquet(input_path, engine="pyarrow")
        fileOut = f'{output_path}/Gemini_Chunks_-_{self.book_name}.xlsx'

        paragraph_chunks = dataset[dataset['Book Name'] == self.book_name].reset_index(drop=True)
        if paragraph_chunks.empty:
            sys.exit("Choose a valid book name!")

        id_chunks = paragraph_chunks['Chunk'].to_frame()
        id_chunks = id_chunks.apply(self.add_ids, axis=1)

        chunk_number = 0
        new_id_list = []
        word_count_aux = []

        while chunk_number < len(id_chunks) - 5:
            word_count = 0
            i = 0
            while word_count < 550 and i + chunk_number < len(id_chunks) - 1:
                i += 1
                final_document = "\n".join(
                    f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
                word_count = self.count_words(final_document)

            if i == 1:
                final_document = "\n".join(
                    f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
            else:
                final_document = "\n".join(
                    f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i - 1 + chunk_number))

            question = f"\nDocument:\n{final_document}"
            word_count = self.count_words(final_document)
            word_count_aux.append(word_count)
            chunk_number = chunk_number + i - 1

            prompt = self.system_prompt + question
            gpt_output = self.LLM_prompt(user_prompt=prompt)

            if gpt_output == "content_flag_increment":
                chunk_number = chunk_number + 1
            else:
                pattern = r"Answer: ID \w+"
                match = re.search(pattern, gpt_output)

                if match is None:
                    print("repeat this one")
                else:
                    gpt_output1 = match.group(0)
                    print(gpt_output1)
                    pattern = r'\d+'
                    match = re.search(pattern, gpt_output1)
                    chunk_number = int(match.group())
                    new_id_list.append(chunk_number)
                    if new_id_list[-1] == chunk_number:
                        chunk_number = chunk_number + 1

        new_id_list.append(len(id_chunks))
        id_chunks['Chunk'] = id_chunks['Chunk'].str.replace(r'^ID \d+:\s*', '', regex=True)

        new_final_chunks = []
        chapter_chunk = []
        for i in range(len(new_id_list)):
            start_idx = new_id_list[i - 1] if i > 0 else 0
            end_idx = new_id_list[i]
            new_final_chunks.append('\n'.join(id_chunks.iloc[start_idx: end_idx, 0]))

            if paragraph_chunks["Chapter"][start_idx] != paragraph_chunks["Chapter"][end_idx - 1]:
                chapter_chunk.append(
                    f"{paragraph_chunks['Chapter'][start_idx]} and {paragraph_chunks['Chapter'][end_idx - 1]}")
            else:
                chapter_chunk.append(paragraph_chunks['Chapter'][start_idx])

        df_new_final_chunks = pd.DataFrame({'Chapter': chapter_chunk, 'Chunk': new_final_chunks})
        df_new_final_chunks.to_excel(fileOut, index=False)
        print(f"{self.book_name} Completed!")


if __name__ == "__main__":
    # Example usage:
    input_path = "../../data/GutenQA_paragraphs.parquet"
    output_path = "../Example"
    chunker = LumberChunker(model_type='deepseek-v3-0324', book_name='A_Christmas_Carol_-_Charles_Dickens')
    chunker.chunk_book(input_path=input_path, output_path=output_path)