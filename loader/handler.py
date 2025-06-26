from pathlib import Path

from datetime import datetime
from .dataform import OriginalData
from rich.console import Console
from rich.progress import Progress
import chardet
import sys
import importlib.util
import subprocess

class FileHandler:
    supported = [".txt", ".md"]
    def __init__(self, file_path):
        """
        Initialize a FileHandler instance with the given file path. 

        Args:
            file_path(str or Path): Path to the file to be handled.
        """

        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"File {file_path} not found")
        elif self.file_path.suffix not in self.supported:
            raise ValueError(f"File {file_path.suffix} is not supported")
        self.console = Console()

    def detect_encoding(self):
        """
        # Read the first 4096 bytes of the file to detect encoding
        """

        with self.file_path.open('rb') as f:
            raw = f.read(4096)
            detect_encoding = chardet.detect(raw).get('encoding') or "utf-8"
        self.console.print(f"[green]Detected encoding: {detect_encoding}[/green]")
        return detect_encoding

    def read(self):
        encoding = self.detect_encoding()
        with self.file_path.open(encoding=encoding) as f:
            content =  f.read()

        self.console.print(f"[green]Read file {self.file_path.name} successfully[/green]")
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

    def _shot_cut(self, text):
        """
        移除字符串中方括号及其内部的所有内容
        
        参数:
        text (str): 包含方括号的原始字符串
        
        返回:
        str: 移除方括号及其内容后的字符串
        """
        result = []
        bracket_level = 0
        
        for char in text:
            if char == '[':
                bracket_level += 1
                continue
            if char == ']' and bracket_level > 0:
                bracket_level -= 1
                continue
            if bracket_level == 0:
                result.append(char)
        
        return ''.join(result)

    def _check(self, package_name):

        if importlib.util.find_spec(self._shot_cut(package_name)) is None:
            self.console.print(
                f"[yellow]Package {package_name} not found, installing...[/yellow]")
            try:
                # 使用 subprocess 调用 pip 进行安装
                cmd = [sys.executable, "-m", "pip", "install"]
                cmd.append(package_name)
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    check=True
                )
                self.console.print(f"[green]{package_name} 安装成功[/green]")
            except subprocess.CalledProcessError as e:
                error_msg = f"安装 {package_name} 时出错: {e.stderr}"
                self.console.print(f"[red]{error_msg}[/red]")
        else:
            self.console.print(f"[green]{package_name} 已安装[/green]")

    def _check_install(self, dependencies):
        for package_name in dependencies:
            self._check(package_name)

    def type2md(self):
        pass

class MarkitDownHandler(FileHandler):
    supported = [".docx",".pdf",".xlsx",".xls",".csv"]
    dependencies = ["markitdown[all]"]

    def _save(self, content):
        new_file_path = self.file_path.with_suffix(".md")
        with new_file_path.open('w', encoding='utf-8') as f:
            f.write(content)
        return new_file_path

    def convert2md(
        self,
        keep_data_uris=False
    ):
        self._check_install(self.dependencies)
        from markitdown import MarkItDown
        result = MarkItDown().convert(
            self.file_path, 
            keep_data_uris=keep_data_uris
        )
        self.console.print(f"[bold] {self.file_path} 转换成功 [/bold]")
        return result

    def convert(self):
        return self.convert2md()

    def read(
        self
    ):
        result = self.convert()
        # self.console.print(f"[bold]content: {result.text_content} [/bold]")
        return OriginalData(
            content=result.text_content,
            created_at=str(datetime.fromtimestamp(self.file_path.stat().st_mtime)),
            file_name=self._save(result.text_content).name,
            original_file_name=self.file_path.name
        )

# class MinerUHandler(FileHandler):

if __name__ == "__main__":
    file = MarkitDownHandler("/workspace/Python/awesome-chunker/task/task2/src/user_document.docx")
    print(file.read())