"""
Malware Analysis and Reporting Tool

This script defines a MalwareAnalyzer class that utilizes the OpenAI API to analyze malware source code and generate a threat report.
It is designed to read, analyze, and generate reports based on the source code found in specified directories.

Features:
    - Splitting text into manageable chunks for analysis.
    - Reading and concatenating the contents of all files in a directory.
    - Analyzing code fragments for potential threats using the OpenAI API. We use code fragments to avoid exceeding the token limit.
    - Analyzing the source code in a directory for malware-related threats.
    - Estimating the number of tokens required for a given text.
    - Waiting for token availability to manage API usage.
    - Summarizing detailed threat findings into concise reports.
    - Generating comprehensive reports from detailed findings.

Example:
    analyzer = MalwareAnalyzer('your-api-key')
    findings = analyzer.analyze_malware_source_code('path/to/source/code')
    report = analyzer.generate_report(findings)

Note:
    The script requires an API key from OpenAI to function. Ensure that the key is valid and has the necessary permissions.
    This script is designed for use with Python 3.x and requires external libraries `tqdm` and `openai`.
"""
import os
from typing import Optional
import time
from tqdm import tqdm
import openai
from pathlib import Path


class MalwareAnalyzer:
    def __init__(self, api_key: str):
        """
        Initializes the MalwareAnalyzer with the specified API key.

        Args:
            api_key (str): The OpenAI API key used for authentication.
        """
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.tokens_per_minute = 10000  # Token limit per minute
        self.last_request_time = time.time()
        self.tokens_used = 0

        # Constants for token and character management
        self.MAX_OUTPUT_TOKENS = 2048
        self.CHAR_NUM_PER_CHUNK = 13000
        self.MAX_NUM_CHARS_PER_MINUTE = 150000
        self.CHAR_NUM_PER_CHUNK_GPT4 = 10000

    def check_strings(self, string: str, strings: list[str]) -> bool:
        """
        Determines if any specified strings are in the provided string.

        Args:
            string: String to check.
            strings: List of substrings to search for.

        Returns:
            bool: True if any specified string is found, False otherwise.
        """
        return any(s in string for s in strings)

    def is_file_we_can_analyze(self, file_path: str) -> bool:
        """
        Determines whether a file is a Python or C# file based on its extension.

        This function checks the file extension and returns True if it's either '.py'
        for Python or '.cs' for C#, indicating that the file can be analyzed.

        Args:
            file_path (str): The path of the file to check.

        Returns:
            bool: True if the file is a Python or C# file, False otherwise.
        """
        return Path(file_path).suffix in [".py", ".cs"]

    def split_into_chunks(self, text: str, max_length: int = 4000, overlap: int = 200) -> list[str]:
        """
        Splits the text into chunks, each with a maximum length of max_length.
        This function attempts to split the text at logical breakpoints or enforces splitting if no breakpoints are found.

        Args:
            text: The text to be split into chunks.
            max_length: The maximum allowed length for each chunk.
            overlap: The number of characters to overlap between chunks.

        Returns:
            A list of text chunks, each not exceeding the specified maximum length.
        """
        chunks = []
        current_chunk = ""

        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break
            else:
                split_index = max_length

                # Look for a breakpoint in the last 200 characters
                for i in range(max_length - 1, max_length - overlap - 1, -1):
                    if text[i] in ["\n", " ", ";", "{", "}"]:
                        split_index = i
                        break

                current_chunk = text[:split_index]
                chunks.append(current_chunk)
                text = text[split_index:]

        return chunks

    def read_and_combine_files(self, directory: str) -> str:
        """
        Concatenates the contents of all readable files in a given directory.

        Args:
            directory: Path of the directory to read files from.

        Returns:
            str: Combined content of all files.
        """
        all_code = ""
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if not self.is_file_we_can_analyze(file_path):
                    continue
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                        all_code += file.read() + "\n\n"
        return all_code

    def get_findings(self, chunk: str) -> Optional[str]:
        """
        Analyzes a code fragment for potential threats using the OpenAI API.

        Args:
            chunk: Fragment of code to analyze.

        Returns:
            Findings from the analysis, if any.
        """
        initial_message = {
            "role": "system",
            "content": "You are an expert threat yet helpful analyst with a deep understanding of malware and programming languages.",
        }
        prompt = """"Perform a thorough analysis of this malware source code fragment. Focus on identifying and interpreting:
            1. All hardcoded strings and values, such as domain names, user agent strings, file paths, and unique identifiers. Discuss their potential significance in the context of malware, considering how they might be used for network communication, data exfiltration, impersonation, or evasion.
            2. Any code patterns or structures that could indicate obfuscation or evasion techniques. This includes unusual loops, complex conditional constructs, or encoded data that might be used to conceal the code’s true intent or to delay its execution.
            3. Specific string literals and data patterns that could suggest interaction with external systems or networks. Pay close attention to any elements that might be involved in making HTTP requests, handling cookies, or communicating with APIs, especially those related to social media platforms or other web services.

            Provide a detailed assessment of these elements, explaining how each could be leveraged in a malware context for various malicious objectives. If a particular element or pattern seems benign, explain how it might still be repurposed for malicious activities in different scenarios.
            Highlight examples of the values from your analysis and not just the context.
        """

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[initial_message, {"role": "user", "content": f"{prompt}\n\n{chunk}"}],
            max_tokens=self.MAX_OUTPUT_TOKENS,
        )
        return completion.choices[0].message.content if completion.choices else None

    def analyze_malware_source_code(self, directory_path: str, sleep_interval: int = 20) -> list[str]:
        """
        Analyzes source code in the specified directory for malware-related threats.

        Args:
            directory_path: Path to the directory containing source code.
            sleep_interval: How long should we sleep if there's a need to wait
                after consuming the most tokens. Defaults to 10 seconds.
        Returns:
            Detailed findings from the analysis.
        """
        all_code = self.read_and_combine_files(directory_path)
        split_chunks = self.split_into_chunks(all_code, self.CHAR_NUM_PER_CHUNK)
        detailed_findings = []
        cur_max = 0
        for code_chunk in tqdm(split_chunks):
            cur_max += len(code_chunk)
            # Reset the number of tokens processed.
            if cur_max >= self.MAX_NUM_CHARS_PER_MINUTE:
                time.sleep(sleep_interval)
                cur_max = 0
            findings = self.get_findings(code_chunk)
            if findings:
                detailed_findings.append(findings)

        return detailed_findings

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens required for processing a given text.

        Args:
            text (str): The text to be processed.

        Returns:
            int: The estimated number of tokens required.
        """
        return len(text.split()) * 4

    def wait_for_token_availability(self, tokens_needed: int):
        """
        Wait until enough tokens are available for processing.

        Args:
            tokens_needed (int): The number of tokens required for the next request.
        """
        while self.tokens_used + tokens_needed > self.tokens_per_minute:
            time_since_last_request = time.time() - self.last_request_time
            # Rate of token replenishment
            tokens_recovered = time_since_last_request * (self.tokens_per_minute / 60)
            self.tokens_used = max(0, self.tokens_used - tokens_recovered)
            if self.tokens_used + tokens_needed > self.tokens_per_minute:
                # Wait and check again
                time.sleep(5)

    def summarize_findings(self, chunk: str) -> str:
        """
        Summarize detailed threat findings into a concise report.

        Args:
            chunk (str): The text chunk containing detailed threat findings.
            client: The GPT client used for generating summaries.

        Returns:
            str: A concise summary of the threat findings.
        """

        prompt = (
            "Summarize the following detailed threat findings into a concise report, do not lose important context such as the malware's behavior:\n\n"
            + chunk
        )
        tokens_needed = self.estimate_tokens(prompt)
        self.wait_for_token_availability(tokens_needed)

        completion = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a threat analyst tasked with condensing dense information from a malware source code into concise summaries. Do not lot lose important information such as hardcoded strings or ttps.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
        )

        # Update token usage and request time
        self.tokens_used += tokens_needed
        self.last_request_time = time.time()

        return completion.choices[0].message.content if completion.choices else ""

    def generate_report(self, detailed_findings: list[str]) -> Optional[str]:
        """
        Compiles a comprehensive report from the given malware analysis findings.

        This function ensures that the generated report is based on the integrated input findings,
        avoiding the generation of separate reports for each chunk.

        Args:
            detailed_findings: Findings from the malware source code analysis.

        Returns:
            str: Generated consolidated report.
        """
        # Summarize each chunk of findings
        summarized_findings = [
            self.summarize_findings(chunk)
            for chunk in tqdm(self.split_into_chunks("\n".join(detailed_findings), 10000))
        ]
        combined_summaries = "\n".join(summarized_findings)

        # Generate the consolidated report from the summaries
        final_question = f"""Generate a comprehensive report detailing the threat actor's objectives, including observed TTPs and IOCs. 
                            Present the information in HTML format. Use the following summarized findings:\n\n" {combined_summaries}.
                            """
        system_message = {
            "role": "system",
            "content": "You are a skilled yet helpful threat intelligence analyst capable of summarizing complex findings into clear and concise reports.",
        }
        user_question = {"role": "user", "content": final_question}

        final_completion = self.client.chat.completions.create(
            model="gpt-4-1106-preview", messages=[system_message, user_question], max_tokens=self.MAX_OUTPUT_TOKENS
        )

        return final_completion.choices[0].message.content if final_completion.choices else "No report generated."
