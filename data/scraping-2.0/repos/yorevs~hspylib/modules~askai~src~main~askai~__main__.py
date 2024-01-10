#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
   @project: hspylib
   @package: hspylib
      @file: __main__.py
   @created: Fri, 5 Jan 2024
    @author: <B>H</B>ugo <B>S</B>aporetti <B>J</B>unior"
      @site: https://github.com/yorevs/hspylib
   @license: MIT - Please refer to <https://opensource.org/licenses/MIT>

   Copyright·(c)·2024,·HSPyLib
"""
import logging as log
import sys
from textwrap import dedent

from clitt.core.tui.tui_application import TUIApplication
from hspylib.core.enums.charset import Charset
from hspylib.core.tools.commons import syserr, sysout
from hspylib.core.zoned_datetime import now
from hspylib.modules.application.argparse.parser_action import ParserAction
from hspylib.modules.application.exit_status import ExitStatus
from hspylib.modules.application.version import Version

from askai.__classpath__ import _Classpath
from askai.core.ask_ai import AskAI
from askai.core.engine.ai_engine import AIEngine
from askai.core.engine.openai.openai_engine import OpenAIEngine
from askai.core.exception.exceptions import NoSuchEngineError


class Main(TUIApplication):
    """HsPyLib Ask-AI Terminal Tools - AI on the palm of your shell."""

    # The welcome message
    DESCRIPTION = _Classpath.get_source_path("welcome.txt").read_text(
        encoding=Charset.UTF_8.val
    )

    # Location of the .version file
    VERSION_DIR = _Classpath.source_path()

    @staticmethod
    def _find_engine(engine_type: str, engine_model: str) -> AIEngine:
        match (engine_type.lower() if engine_model else "openai"):
            case "openai":
                return OpenAIEngine.of_value(
                    engine_model or OpenAIEngine.GPT_3_5_TURBO.value
                )
            case "paml":
                syserr("Google paml is not yet implemented!")
            case _:
                raise NoSuchEngineError(f"Engine type: ${engine_type}  model: ${engine_model}")
        sys.exit(ExitStatus.ABORTED.val)

    def __init__(self, app_name: str):
        version = Version.load(load_dir=self.VERSION_DIR)
        super().__init__(app_name, version, self.DESCRIPTION.format(version))
        self._ai = None

    def _setup_arguments(self) -> None:
        """Initialize application parameters and options."""
        # fmt: off
        self._with_options() \
            .option(
                "interactive", "i", "interactive",
                "whether you would like to run the program in an interactive mode or not.",
                nargs="?", action=ParserAction.STORE_TRUE, default=False)\
            .option(
                "engine", "e", "engine", "specifies which AI engine to use."
                "If not provided, 'openai' wil be used.",
                choices=['openai', 'palm'])\
            .option(
                "model", "m", "model", "specifies which AI model to use (depends on the engine).",
                nargs=1)
        self._with_arguments() \
            .argument("query_string", "what to ask to the AI engine", nargs="*")
        # fmt: on

    def _main(self, *params, **kwargs) -> ExitStatus:
        """Run the application with the command line arguments."""

        self._ai = AskAI(
            bool(self.get_arg("interactive")),
            self._find_engine(self.get_arg("engine"), self.get_arg("model")),
            self.get_arg("query_string")
        )

        log.info(
            dedent(
                f"""
        {self._app_name} v{self._app_version}

        Settings ==============================
                STARTED: {now("%Y-%m-%d %H:%M:%S")}
        """
            )
        )
        return self._exec_application()

    def _exec_application(self) -> ExitStatus:
        """Execute the application main flow."""
        self._ai.run()
        sysout("%EOL%", end="")
        return ExitStatus.SUCCESS


# Application entry point
if __name__ == "__main__":
    Main("ask-ai").INSTANCE.run(sys.argv[1:])
