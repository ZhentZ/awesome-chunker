from pydantic import BaseModel
import re
from rich import print as rprint

class DataForm(BaseModel):
    chunk: str

class OriginalData(BaseModel):
    content: str
    created_at: str #时间戳
    original_file_name: str
    file_name: str
    
    def static(self):
        return len(self.content)

    def extract_images(self, verbose=False):
        pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        images = re.findall(pattern, self.content)
        if verbose: 
            rprint(images)
        return images