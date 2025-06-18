from dynaconf import Dynaconf
from pathlib import Path
import os
curr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"items")

def get_setting_file(directory=curr_path, suffixed=(".yml",".toml",".json")):
    """
    获取指定目录下所有指定后缀的配置文件
    :param directory: 目录
    :param suffixed: 后缀
    :return: 配置文件列表
    """
    directory = Path(directory)
    return [str(p) for p in directory.rglob('*') if p.suffix in suffixed]
    
preference = Dynaconf(
    CACHE_ENABLED = True,
    settings_files=get_setting_file(),
    CACHE_TIMEOUT = 3600
)