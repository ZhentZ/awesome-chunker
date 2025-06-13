# import os
# # 方法 A：通过环境变量指定镜像源（优先推荐）
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# from huggingface_hub import hf_hub_download, snapshot_download
#
#
# repo_id = "chentong00/propositionizer-wiki-flan-t5-large"
#
# # 下载整个仓库（含所有文件）
# snapshot_download(
#   repo_id=repo_id,
#   local_dir="../models/chentong00/propositionizer-wiki-flan-t5-large"
# )
import nltk
nltk.download('punkt')