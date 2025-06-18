from pathlib import Path
import chardet
from datetime import datetime
from .dataform import OriginalData

class FileHandler:

    def __init__(self, file_path):
        """
        Initialize a FileHandler instance with the given file path. 

        Args:
            file_path(str or Path): Path to the file to be handled.
        """

        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"File {file_path} not found")
    

    def detect_encoding(self):
        """
        # Read the first 4096 bytes of the file to detect encoding
        """

        with self.file_path.open('rb') as f:
            raw = f.read(4096)
            detect_encoding = chardet.detect(raw).get('encoding')
            return detect_encoding or "uft-8"

    def read(self):
        encoding = self.detect_encoding()
        with self.file_path.open(encoding=encoding) as f:
            content =  f.read()

        return OriginalData(
            content=content, 
            created_at=str(datetime.fromtimestamp(self.file_path.stat().st_mtime)), 
            file_name=self.file_path.name, 
            original_file_name=self.file_path.name
            )

    def readlines(self):
        encoding = self.detect_encoding()
        with self.file_path.open(encoding=encoding) as f:
            return f.readlines()

class JSONHandler(FileHandler):
    def read(self):
        import json
        encoding = self.detect_encoding()
        with self.file_path.open('r', encoding=encoding) as f:
            return json.load()

if __name__ == "__main__":
    file = FileHandler("/workspace/Python/awesome-chunker/task/task3/src/test.txt")
    print(file.read())