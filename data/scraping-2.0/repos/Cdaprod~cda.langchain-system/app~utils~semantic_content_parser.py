from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import ast
import os
import yaml
import markdown2
import re
from langsmith.run_helpers import traceable
from langchain.agents import AgentExecutor, tool
from langchain.llms import OpenAI
from langchain.schema.runnable import Object, Text
from langchain.runners import Document
from langchain.prompts import StringPromptTemplate
from langchain.document_loaders import UnstructuredMarkdownLoader

class SourceCode(BaseModel):
    id: str = Field(description="Unique identifier for the source code object.")
    imports: List[str] = Field(description="List of extracted required packages.")
    classes: List[str] = Field(description="List of extracted classes from the code.")
    code: str = Field(description="Source code snippets.")
    syntax: str = Field(description="The programming language syntax/extension (e.g., Python).")
    context: str = Field(description="Any extracted text, markdown, comments, or docstrings.")
    metadata: dict = Field(description="Extracted or generated metadata tags for top-level cataloging and code object management.")

class Table(BaseModel):
    headers: List[str] = Field(description="Headers of the table")
    rows: List[Dict[str, Any]] = Field(description="Rows of the table, each row being a dictionary")

class MarkdownDocument(BaseModel):
    metadata: Dict[str, Any] = Field(description="Metadata of the document")
    tables: List[Table] = Field(description="List of tables in the document")
    code_blocks: List[SourceCode] = Field(description="List of code blocks in the document")
    content: str = Field(description="The textual content of the document")

@traceable(run_type="chain")
@tool
def parse_yaml_metadata(yaml_content: str) -> dict:
    try:
        return yaml.safe_load(yaml_content) or {}
    except yaml.YAMLError:
        return {}

@traceable(run_type="chain")
@tool
def parse_table(table_content: str) -> Table:
    # Assuming table_content is in a format that pandas can read directly
    df = pd.read_html("<table>" + table_content + "</table>")[0]
    return Table(headers=df.columns.tolist(), rows=df.to_dict(orient="records"))

@traceable(run_type="chain")
@tool
def parse_python_script(script: str) -> SourceCode:
    extracted_imports = []
    extracted_classes = []
    extracted_context = ""
    extracted_metadata = {}

    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return SourceCode(
            id="error",
            imports=[],
            classes=[],
            code=script,
            syntax="Python",
            context="Syntax error in provided script",
            metadata={"error": str(e)}
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            extracted_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            extracted_imports.append(node.module)
        elif isinstance(node, ast.ClassDef):
            extracted_classes.append(node.name)

    return SourceCode(
        id="generated_id",
        imports=extracted_imports,
        classes=extracted_classes,
        code=script,
        syntax="Python",
        context=extracted_context,
        metadata=extracted_metadata
    )

@traceable(run_type="llm")
@tool
def parse_markdown_content(markdown_path: str) -> MarkdownDocument:
    loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
    markdown_elements = loader.load()

    extracted_metadata = {}
    extracted_tables = []
    extracted_code_blocks = []
    extracted_content = []

    for element in markdown_elements:
        if element['type'] == 'yaml':
            extracted_metadata.update(parse_yaml_metadata(element['content']))
        elif element['type'] == 'table':
            extracted_tables.append(parse_table(element['content']))
        elif element['type'] == 'code' and element['language'] == 'python':
            extracted_code_blocks.append(parse_python_code_block(element['content']))
        else:
            extracted_content.append(element['content'])

    return MarkdownDocument(
        metadata=extracted_metadata,
        tables=extracted_tables,
        code_blocks=extracted_code_blocks,
        content="\\n".join(extracted_content)
    )
    

parse_yaml_metadata_tool = Tool.from_function(
    func=parse_yaml_metadata,
    name="parse_yaml_metadata",
    description="Parses YAML metadata from a string"
),

parse_table_tool = Tool.from_function(
    func=parse_table,
    name="parse_table",
    description="Parses table content into a Table object"
),

parse_python_script_tool = Tool.from_function(
    func=parse_python_script,
    name="parse_python_script",
    description="Parses a Python script into a SourceCode object"
),

parse_markdown_content_tool = Tool.from_function(
    func=parse_markdown_content,
    name="parse_markdown_content",
    description="Parses Markdown content into a MarkdownDocument object"
)

## Python Prompt Templating 

MARKDOWN_DOCUMENT_ANALYSIS_PROMPT = """
# Markdown Document Analysis

**Markdown Content for Analysis:**
{markdown_content}

**Task:**
Analyze the provided Markdown content and extract its components, mapping them to a new MarkdownDocument object. Focus on the following aspects:

**Extracted Components:**
- **Metadata**: Extract and summarize any metadata present in the document.
- **Tables**: Identify and describe tables in the document, including headers and row data.
- **Code Blocks**: Enumerate the code blocks present, especially focusing on their content and language syntax.
- **Other Content**: Highlight other significant content details such as headings, paragraphs, lists, and links.

**Instructions:**
- Provide a comprehensive analysis covering all key components of the Markdown content.
- Use a structured format to present the extracted components clearly.
- Ensure that each component is accurately represented as per its significance in the document.
- Offer additional insights or context where relevant.

---
"""

class MarkdownDocumentAnalysisPromptTemplate(StringPromptTemplate):
    def format(self, markdown_content: str) -> str:
        return MARKDOWN_DOCUMENT_ANALYSIS_PROMPT.format(markdown_content=markdown_content)

## Python Prompt Templating 

class PythonScriptPromptTemplate(StringPromptTemplate):
    def format(self, script: str) -> str:
        return f"""
Analyze the following Python script:
Script:
{script}
Extracted Components:
- Imports:
- Classes:
- Other relevant details:
"""

PYTHON_SCRIPT_ANALYSIS_PROMPT = """
# Python Script Analysis

Script for Analysis:
{script}

**Task:
Analyze the Python script and extract its components, mapping them to a new SourceCode object. Focus on the following aspects:

**Extracted Components:**
- **Imports**: List all the modules and libraries imported in the script.
- **Classes**: Enumerate the classes defined in the script along with a brief description of their purpose.
- **Functions**: Outline the functions, their roles, and interactions within the script.
- **Syntax**: Specify the programming language syntax/extension used.
- **Context**: Describe any additional context, such as markdown, comments, or docstrings.
- **Metadata**: Provide any relevant metadata extracted or generated for cataloging and code object management.

**Instructions:**
- Ensure the analysis is thorough and detailed, covering all major components of the script.
- Use clear and concise language to describe each component.
- Maintain the structure of the SourceCode object in the response.

---
"""

class PythonScriptAnalysisPromptTemplate(StringPromptTemplate):
    def format(self, script: str) -> str:
        return PYTHON_SCRIPT_ANALYSIS_PROMPT.format(script=script)

class EnhancedGeneralAnalysisPromptTemplate(StringPromptTemplate):
    def format(self, input_data: dict) -> str:
        filename = input_data.get("filename", "")
        content = input_data.get("content", "")

        if filename.endswith('.py') or filename.endswith('.ipynb'):
            # For Python script or Jupyter notebook analysis
            return f"""
Detailed Analysis of Python Script or Jupyter Notebook:
Filename: {filename}
Content:
{content}

Task:
- Analyze the structure, logic, and components of the script/notebook.
- Extract detailed information about Imports, Classes, Functions, Variables, and overall Logic.
- Identify key programming patterns, algorithms, and data structures used.
- Provide context, documentation strings, and metadata.

Expected Output Format:
- Imports: List all libraries and modules with their purposes.
- Classes: Describe each class, its methods, attributes, and role in the script.
- Functions: Detail each function, its parameters, return values, and functionality.
- Variables: Highlight significant variables, their types, and use cases.
- Structure: Outline the script's flow and logic.
- Context and Metadata: Include any comments, docstrings, and relevant metadata.
"""
        elif filename.endswith('.md'):
            # For Markdown document analysis
            return f"""
Detailed Analysis of Markdown Document:
Filename: {filename}
Content:
{content}

Task:
- Analyze the Markdown document to extract structured information.
- Focus on Metadata, Tables, Code Blocks, and textual Content.
- Provide insights into the document's layout, key themes, and structure.

Expected Output Format:
- Metadata: Extract key metadata and summarize.
- Tables: Describe each table, its headers, and row data.
- Code Blocks: Analyze any code blocks, their language, and content.
- Content: Summarize the main textual content, headings, and lists.
"""
        else:
            # Default analysis for other types of content
            return f"""
General Content Analysis:
Filename: {filename}
Content:
{content}

Task:
- Provide a comprehensive analysis of the content.
- Focus on extracting key insights, summaries, and relevant details.

Expected Output Format:
- Key Insights: Highlight the main points and insights.
- Summary: Provide a concise summary of the content.
- Details: Include any other relevant information or observations.
"""

enhanced_general_analysis_prompt_template = EnhancedGeneralAnalysisPromptTemplate()

llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

tools = [
    parse_yaml_metadata_tool,
    parse_table_tool,
    parse_python_script_tool,
    parse_markdown_content_tool
]

python_script_analysis_prompt_template = PythonScriptPromptTemplate()
markdown_document_analysis_prompt_template = MarkdownDocumentPromptTemplate()
enhanced_general_analysis_prompt_template = EnhancedGeneralAnalysisPromptTemplate()

# def agent_logic(input_data: dict):
#     filename = input_data.get("filename", "")
#     input_text = input_data.get("content", "")
#     elif filename.endswith('.py') or filename.endswith('.ipynb'):
#         return python_script_prompt_template.format(input_text) 
#     elif filename.endswith('.md'):
#         if '```python' in input_text:
#             return python_script_prompt_template.format(input_text)
#         else:
#             return markdown_document_analysis_prompt_template.format(input_text)
#     else:
#         return input
# agent = (
#     {"input": lambda x: x["input"]}  # "input" is the stringified script
#     | prompt_template
#     | llm.bind(functions=tools)
#     | (lambda output: agent_logic(output))
# )
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def agent_logic(input_data: dict):
    return enhanced_general_analysis_prompt_template.format(input_data)

agent = (
    {"input": lambda x: x["input"]}  # "input" is the input data dictionary
    | (lambda output: agent_logic(output))
    | llm.bind(functions=tools)
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
