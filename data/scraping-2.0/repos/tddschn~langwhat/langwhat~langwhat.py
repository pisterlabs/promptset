from codecs import getdecoder


class LangWhat:
    def __init__(
        self,
        query,
        is_zh: bool = False,
        sydney: bool = False,
        bing_cookie_json_path: str | None = None,
        api_base: str | None = None,
        show_token_usage: bool = False,
    ):
        # if api_base:
        #     openai.api_base = api_base
        self.query = query
        self.is_zh = is_zh
        self.sydney = sydney
        if self.sydney and bing_cookie_json_path is None:
            raise Exception("Bing cookie json path not provided")
        self.cookie_path = bing_cookie_json_path
        self.references: str | None = None
        self.total_tokens = 0
        self.show_token_usage = show_token_usage

    def get_response(self):
        from .utils import get_llm_chain

        chain = get_llm_chain(
            is_zh=self.is_zh, sydney=self.sydney, cookie_path=self.cookie_path
        )
        from langchain.callbacks import get_openai_callback

        with get_openai_callback() as cb:
            response = chain(self.query)
            self.total_tokens = cb.total_tokens

        from .utils import parse_chain_response

        self.references, self.might_be, self.description = parse_chain_response(
            response
        )

    def show(self):
        from rich.console import Console
        from rich.table import Table
        from rich.style import Style
        from rich.text import Text

        self.get_response()

        console = Console()
        title = Text(
            "LangWhat [https://github.com/tddschn/langwhat]",
            style=Style(color="#268bd2", bold=True),
        )
        table = Table(title=title, show_lines=False, style="dim")  # type: ignore
        table.add_column("Query", style=Style(color="#b58900"))
        table.add_column("Might Be", style=Style(color="#d33682"), justify="middle")  # type: ignore
        table.add_column("Description", style=Style(color="#859900"), justify="left")
        table.add_row(self.query, self.might_be, self.description)
        console.print(table)

        if self.references:
            console.print("References:")
            console.print(self.references)

        if self.show_token_usage:
            console.print(f"Total tokens used: {self.total_tokens}")
