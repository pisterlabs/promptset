import getpass
import re
import signal
from typing import Any

import pexpect as pexpect
import tiktoken
from langchain.utilities import BashProcess
from langchain.text_splitter import TokenTextSplitter


class SemanticTerminalProcess(BashProcess):
    model_name: str = "gpt-3.5-turbo"
    chunk_size: int = 500

    def __init__(self, pid, print_terminal_output=True, timeout=20):
        self.print_terminal_output = print_terminal_output
        self.timeout = timeout
        self.pid = pid
        self.prompt = pid
        self.process = self._initialize_persistent_process()
        self.last_command_output = ""
        self.incorrect_password_attempts = 0

    def _initialize_persistent_process(self) -> pexpect.spawn:
        process = pexpect.spawn(
            "bash",
            encoding="utf-8",
        )
        process.expect(r"\$")
        process.sendline("PS1=" + self.prompt)
        process.expect_exact(self.prompt, timeout=10)
        return process

    @staticmethod
    def _tiktoken_encoder(text: str, **kwargs: Any) -> int:
        encoder = tiktoken.encoding_for_model(SemanticTerminalProcess.model_name)
        return len(encoder.encode(text, **kwargs))

    @staticmethod
    def _get_last_n_tokens(text: str, n: int = chunk_size, overlap: int = 200) -> str:
        """Return last n tokens from the output."""
        text_splitter = TokenTextSplitter(
            model_name=SemanticTerminalProcess.model_name,
            chunk_size=n,
            chunk_overlap=overlap,
        )
        split_text = text_splitter.split_text(text)
        last = split_text[-1]
        if SemanticTerminalProcess._tiktoken_encoder(last) < n:
            return last
        else:
            return "Truncated Output: ..." + "".join(split_text[-2:])

    def process_output(self, output: str, command: str) -> str:
        """Process the output."""
        return output

    def _run_persistent(self, command: str) -> str:
        """Run commands and return final output."""
        self.command = command
        if self.process is None:
            raise ValueError("Process not initialized")
        print("semterm > " + command)
        try:
            self.process.sendline(command)
            self.process.expect([command, self.prompt], timeout=self.timeout)
            self.last_command_output = self._handle_stdout(command)
        except Exception as e:  # noqa - LLM is extremely error prone at the moment.
            self.last_command_output = (
                "The last command resulted in an error. Error: ",
                str(e),
            )
        if self.print_terminal_output:
            print(self.last_command_output)
        return SemanticTerminalProcess._get_last_n_tokens(self.last_command_output)

    def _handle_stdout(self, command):
        response = self._handle_terminal_expects(command)
        return self._handle_terminal_response(command, response)

    def _handle_terminal_response(self, command, response):
        if response == "password_request":
            return self._handle_password_request(command)
        if response == "incorrect_password":
            if self.incorrect_password_attempts > 2:
                return "Too many bad pass attempts."
            self.incorrect_password_attempts += 1
            return self._handle_password_request(command, self.incorrect_password_attempts)
        elif response == "prompt":
            return self.process.before
        elif response == "EOF":
            return f"Process exited with error status: " \
                   f"{self.process.exitstatus}"

        elif response == "TIMEOUT":
            return f"Timeout reached. Most recent output: " \
                   f"{self.process.buffer}"

    def _handle_password_request(self, command, try_count=0):
        try:
            try_text = f"{try_count} / 3 Attempts\n" if try_count > 0 else f"\n"
            signal.signal(signal.SIGINT, self.keyboard_interrupt_handler)
            try:
                self.process.expect_exact(':', timeout=1)
            except pexpect.exceptions.TIMEOUT:  # pragma: no cover
                pass
            self.process.sendline(
                getpass.getpass(
                    try_text +
                    f"semterm is requesting your password to run the following command: {command}\n"
                    f"If you trust semterm, please enter your password below:\n"
                    f"(CTRL+C to Dismiss) Password for {getpass.getuser()}: ",
                )
            )
            return self._handle_stdout(command)
        except KeyboardInterrupt:
            self.process.sendintr()
            print("KeyboardInterrupt: Password not sent.")
            return "User aborted password request."
        finally:
            signal.signal(signal.SIGINT, signal.default_int_handler)

    def _handle_terminal_expects(self, command: str) -> str:
        password_regex = re.compile(
            r"(password for|Enter password|Password:|'s password:)", re.IGNORECASE
        )
        incorrect_password_regex = re.compile(
            r"(?i)(?!.*attempts)(incorrect password|password incorrect|wrong password|try "
            r"again|wrong|incorrect)"
        )
        expect_dict = {
            "prompt": self.prompt,
            "password_request": password_regex,
            "incorrect_password": incorrect_password_regex,
            "EOF": pexpect.EOF,
            "TIMEOUT": pexpect.TIMEOUT,
        }
        list_index = self.process.expect(
            list(expect_dict.values()), timeout=self.timeout
        )

        return list(expect_dict.keys())[list_index]

    def get_most_recent_output(self):
        return self.process.buffer

    @staticmethod
    def keyboard_interrupt_handler(sig, frame):
        print("\nPassword request cancelled.")
        raise KeyboardInterrupt
