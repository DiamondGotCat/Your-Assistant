
import os
import subprocess
import re
from typing import List, Optional, TypedDict, Literal
import ollama  # OpenAIからOllamaに変更
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from markdown_it import MarkdownIt

# メッセージの型定義
class ChatCompletionMessageParam(TypedDict):
    role: Literal['system', 'user', 'assistant']
    content: str

# OpenAI API クライアントの作成を削除
# client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY")  # APIキーは環境変数から取得
# )

# メッセージ履歴
history: List[ChatCompletionMessageParam] = []

# Richコンソールの初期化
console = Console()

# メッセージを履歴に追加し、コンソールに表示
def add_message(role: Literal['system', 'user', 'assistant'], content: str):
    history.append({"role": role, "content": content})

# OpenAI APIと対話をOllamaに置き換え
def chat_with_openai(messages: List[ChatCompletionMessageParam]) -> Optional[str]:
    try:
        response = ollama.chat(
            model="llama3.2:latest",  # 使用するOllamaモデルに更新
            messages=messages
        )
        response_content = response['message']['content']
        return response_content
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/red]")
        return None

# コマンドを実行
def _run_command(command: str) -> str:
    commands = command.split()
    if not commands:
        return "No command provided."
    if commands[0] == "create":
        if len(commands) < 3:
            return "Insufficient arguments for create command."
        if commands[1] == "dir":
            try:
                os.mkdir(commands[2])
                return f"Directory '{commands[2]}' created successfully."
            except Exception as e:
                return f"Failed to create directory '{commands[2]}': {e}"
        elif commands[1] == "file":
            try:
                with open(commands[2], "w") as f:
                    f.write("")
                return f"File '{commands[2]}' created successfully."
            except Exception as e:
                return f"Failed to create file '{commands[2]}': {e}"
    elif commands[0] == "pwd":
        return os.getcwd()
    elif commands[0] == "list":
        try:
            items = os.listdir()
            return "\n".join(items) if items else "Directory is empty."
        except Exception as e:
            return f"Failed to list directory: {e}"
    elif commands[0] == "delete":
        if len(commands) < 3:
            return "Insufficient arguments for delete command."
        if commands[1] == "dir":
            try:
                os.rmdir(commands[2])
                return f"Directory '{commands[2]}' deleted successfully."
            except Exception as e:
                return f"Failed to delete directory '{commands[2]}': {e}"
        elif commands[1] == "file":
            try:
                os.remove(commands[2])
                return f"File '{commands[2]}' deleted successfully."
            except Exception as e:
                return f"Failed to delete file '{commands[2]}': {e}"
    elif commands[0] == "show":
        if len(commands) < 2:
            return "No file specified for show command."
        try:
            with open(commands[1], "r") as f:
                return f"```{commands[1]}\n" + f.read() + "\n```"
        except Exception as e:
            return f"Failed to read file '{commands[1]}': {e}"
    else:
        return "Unknown command."

def run_command(command: str) -> str:
    try:
        result = _run_command(command)
        console.print(result)
        return result
    except Exception as e:
        console.print_exception()
        return str(e)

# AutoGen用のMarkdown情報
autogen_markdown_info = """
# AutoGen Integration

## AutoGen Command
To execute an AutoGen command, simply include the following syntax in assitant message:

```autogen
<write command here>
```

You can use the following commands:
- create dir <directory_name>: Create a new directory
- create file <file_name_and_extension>: Create a new file
- list: List the directory contents
- pwd: Get the current working directory
- delete dir <directory_name>: Delete a directory
- delete file <file_name>: Delete a file
- show <file_name>: Show the content of a file


## File Editing
To edit the file, simply include the following syntax in assitant message:

```<file_name_and_extension>
<file_content>
```

## Shell
To execute a shell command, simply embed the following syntax in assitant message:
    
```run
<shell_command>
```
"""

# Markdownパーサーを使い、Markdown構文を解析し、edit%とrunブロックを処理する関数
def parse_markdown_and_execute(markdown_text: str):
    md = MarkdownIt()
    tokens = md.parse(markdown_text)

    for token in tokens:
        if token.type == 'fence':
            info = token.info.strip()
            content = token.content.strip()

            # `autogen` ブロックの処理
            if info == 'autogen':
                commands = content.split('\n')
                for command in commands:
                    if command.strip():  # 空行を無視
                        result = run_command(command.strip())
                        add_message("system", result)

            # `run` ブロックの処理
            elif info == 'run':
                commands = content.split('\n')
                for command in commands:
                    command = command.strip()
                    if not command:
                        continue  # 空行を無視
                    try:
                        if command.startswith("cd"):
                            parts = command.split(maxsplit=1)
                            if len(parts) < 2:
                                add_message("system", "No directory specified for 'cd' command.")
                                continue
                            new_dir = parts[1]
                            os.chdir(new_dir)
                            message = f"Changed directory to '{new_dir}'."
                            add_message("system", message)
                        else:
                            is_confirmed = Prompt.ask(
                                Markdown(f"Are you sure you want to run the command: `{command}`?"),
                                choices=["y", "n"],
                                default="n"
                            )
                            if is_confirmed.lower() == "y":
                                result = subprocess.run(
                                    command,
                                    shell=True,
                                    text=True,
                                    capture_output=True
                                )
                                if result.stdout:
                                    add_message("system", result.stdout)
                                if result.stderr:
                                    add_message("system", f"[red]{result.stderr}[/red]")
                            else:
                                message = f"Command `{command}` was not executed."
                                add_message("system", message)
                    except Exception as e:
                        error_message = f"Error executing command `{command}`: {e}"
                        add_message("system", error_message)
            else:
                file_name = info
                try:
                    with open(file_name, 'w', encoding='utf-8') as f:
                        f.write(content)
                    message = f"File '{file_name}' updated successfully with your content."
                    add_message("system", message)
                except Exception as e:
                    error_message = f"Failed to update file '{file_name}': {e}"
                    add_message("system", error_message)

        elif token.type in ['paragraph_open', 'inline']:
            if token.type == 'inline':
                text = token.content.strip()
                if text:
                    console.print(Markdown(text))

    if tokens:

        response_content = chat_with_openai(history)
        if response_content:
            add_message("assistant", response_content)

# メイン処理
def main():
    add_message("system", autogen_markdown_info)
    while True:
        user_message = Prompt.ask("Please input message (or /exit to quit)")
        if user_message == "/exit":
            break
        add_message("user", user_message)
        response_content = chat_with_openai(history)
        if response_content:
            add_message("assistant", response_content)
            console.print(Markdown(f"### AI Assistant"))
            parse_markdown_and_execute(response_content)
        else:
            break

if __name__ == "__main__":
    main()
