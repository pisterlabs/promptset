import re
import openai
from typing import List


# TODO: remove this from here
import openai

with open(r'D:\GitHub\ai_adventures\API_KEY', "r") as f:
    key = f.read()

openai.api_key = key


def split_text(text, split_words) -> List[str]:
    """Split chunk of text on any lines that contain a word from split_words"""
    # Split the text into lines
    lines = text.split('\n')

    # Initialize an empty chunk and a list to hold all chunks
    chunk = []
    chunks = []

    # Prepare regex pattern with word boundaries
    patterns = [re.compile(rf'\b{word}\b') for word in split_words]

    # Iterate over the lines
    for line in lines:
        # If the line contains any of the split words, create a new chunk
        if any(pattern.search(line) for pattern in patterns):
            # Only add non-empty chunks
            if chunk:
                chunks.append('\n'.join(chunk))
                chunk = []
        if line:
            chunk.append(line)

    # Add the last chunk if non-empty
    if chunk:
        chunks.append('\n'.join(chunk))

    return chunks



class MermaidGenerator:
    """
    Generate mermaid diagrams from natural language input

    Examples:
        mg = MermaidGenerator()
        mg.mermaidify('A goes to B, B goes to C, D goes to A B C', output_type='html', output_filename='test.html')
    """
    def __init__(self):
        self.system_input = """You are an AI specialized in generating mermaid diagrams from user input. Your task is 
        to translate the user's description into a mermaid diagram code. Remember, your response should contain only 
        the mermaid code - no explanations, no additional text or context. The code should be ready to be rendered as 
        a mermaid diagram immediately.

        In case the user's input is unclear or cannot be translated into a mermaid diagram, start your response with 
        "ERROR: ", followed by a suggestion to help them provide input that can be converted into a mermaid diagram."""
        self.max_tokens = 2000

        self._mm_graph_types = [
            "graph",
            "sequenceDiagram",
            "gantt",
            "classDiagram",
            "gitGraph",
            "erDiagram",
            "journey",
        ]

        self._last_input = None
        self._last_response = None

    def _get_response(self, user_input):
        """Get the response from the llm given user input"""
        self._last_input = user_input
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_input},
                {"role": "user", "content": user_input},
            ],
            temperature=0,
            top_p=1,
            n=1,
            max_tokens=self.max_tokens,
        )
        self._last_response = response
        return response

    def display_response(self, response=None):
        """Display information about the given or last response from llm
        Mostly for debugging
        """
        response = self._last_response if response is None else response
        if response is None:
            print(f"No response to display")
            return
        print(
            "\n".join(
                [
                    f"System text:\n=====\n{self.system_input}\n======",
                    f"User text:\n=====\n{self._last_input}\n======",
                    f"Response text:\n=====\n{response.choices[0].message.content}\n======",
                    f"Response end reason: {response.choices[0].finish_reason}",
                    f"Prompt token usage: {response.usage.prompt_tokens}",
                    f"Completion token usage: {response.usage.completion_tokens}",
                    f"Total token usage: {response.usage.total_tokens}",
                    f"Cost assuming gpt-3.5-turbo: {0.002/1000*response.usage.total_tokens*100:.4f} cents",
                ]
            )
        )
        return

    def _extract_mermaid_code(self, text):
        """Extract the mermaid code part of a text response"""
        # If already formatted as only mm text
        if any([text.startswith(graph_type) for graph_type in self._mm_graph_types]):
            return text

        # If mm text inside code part of response
        if "```mermaid" in text:
            pattern = r"```mermaid(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
        elif "```" in text:
            pattern = r"```(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
        else:
            raise RuntimeError(f"Can't parse: \n {text}")

        if match:
            return match.group(1)
        else:
            raise RuntimeError(f"Can't parse: \n {text}")

    def _split_mm_graphs(self, mermaid_text) -> List[str]:
        """Split possibly multiple mermaid graphs into individual graphs"""
        mm_graph_strings = split_text(mermaid_text, self._mm_graph_types)
        return mm_graph_strings

    def _generate_html(self, mermaid_text, filename):
        """Place the mermaid code into an html file"""

        mm_graph_strings = self._split_mm_graphs(mermaid_text)
        mm_pres = [f'''
        <pre class="mermaid">
            {mm_string}
        </pre>
        ''' for mm_string in mm_graph_strings]
        mm_pre_string = '\n'.join(mm_pres)

        content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            </script>
            <script>mermaid.initialize({{startOnLoad:true, securityLevel: 'loose', theme: 'dark', 
                flowchart: {{
                    useMaxWidth: true, // Disable maximum width for better zooming
                    htmlLabels: true, // Enable HTML-based labels for better styling
                    defaultRenderer: "elk", // Makes connections linear, ugly but good for large graphs
                }},
            }});
            </script>
        </head>
        <body>
            {mm_pre_string}
        </body>
        </html>
        """
        with open(filename, "w") as f:
            f.write(content)

    def mermaidify(self, text, output_type="html", output_filename: str = "test.html"):
        """Make a mermaid graph from natural language text input
        Args:
            text: Any text to convert to mermaid graph
            output_type: 'html' to generate a file, or 'text' to get back the mermaid text only
            output_filename: if 'html' output type, then save the file in 'output_filename'

        Returns:
            mermaid graph
        """
        if output_type not in ["html", "text"]:
            raise ValueError(
                f'output_type must be one of ["html", "text"], got {output_type}'
            )

        response = self._get_response(text)
        if response.choices[0].finish_reason != "stop":
            print(f"Response did not reach good stopping point")
            self.display_response(response)
        if "ERROR" in response.choices[0].message.content:
            print(
                f"Bad input text detected, response text = {response.choices[0].message.content}"
            )
            self.display_response(response)
            return None

        mm_text = self._extract_mermaid_code(response.choices[0].message.content)
        if output_type == "html":
            self._generate_html(mm_text, output_filename)
        return mm_text
