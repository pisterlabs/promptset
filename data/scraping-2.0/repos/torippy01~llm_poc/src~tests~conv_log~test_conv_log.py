import pytest

from mdutils import MdUtils

from langchain.schema import AgentAction
from llama_index.response.schema import Response
from xecretary_core.conv_log.conv_log import ConversationLog


@pytest.fixture
def conv_log_prop():
    return {
        "input": "Hello",
        "output": "Hello, how are you?",
        "intermediate_steps": [
            (
                AgentAction(
                    tool="user_context_from_github wiki",
                    tool_input="プロジェクトのエンジニア",
                    log=(
                        "この情報はGitHubのwikiに記載されている可能性がある。\n"
                        "Action: user_context_from_github wiki\nAction "
                        "Input: プロジェクトのエンジニア"
                    ),
                ),
                Response(
                    "金田\u3000正太郎さん, 島\u3000鉄雄さん, 山形さん, 甲斐さん, "
                    "ケイさん, ミヤコさん"
                ),
            )
        ],
        "chat_history": None,
        "elapsed_time": 1000.00,
    }


def test_dump(conv_log_prop):
    class Mdfile:
        def __init__(self, md_file: MdUtils):
            self.md_file = md_file

    md_file_instance = Mdfile(MdUtils("test.md"))
    conv_log = ConversationLog(**conv_log_prop)
    conv_log.dump(md_file_instance.md_file)
    expect = md_file_instance.md_file.get_md_text()
    answer = "\n\n\n\n## 質問: Hello\n  \n実行時間: `1000.0`  \ntool: `user_context_from_github wiki`  \ntool input: `プロジェクトのエンジニア`  \nlog:\n\n```bash\nこの情報はGitHubのwikiに記載されている可能性がある。\nAction: user_context_from_github wiki\nAction Input: プロジェクトのエンジニア\n```  \nanswer: \n\n```bash\n金田\u3000正太郎さん, 島\u3000鉄雄さん, 山形さん, 甲斐さん, ケイさん, ミヤコさん\n```  \n  \n---  \n"
    assert expect == answer
