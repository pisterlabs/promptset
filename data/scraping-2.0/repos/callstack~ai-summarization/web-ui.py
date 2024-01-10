#!/usr/bin/env python3

from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
import time

VERBOSE = True
MAX_TOKENS = 2048

STYLES = {
    "List": {
        "style": "Return your response as numbered list which covers the main points of the text and key facts and figures.",
        "trigger": "NUMBERED LIST SUMMARY WITH KEY POINTS AND FACTS",
    },
    "One sentence": {
        "style": "Return your response as one sentence which covers the main points of the text.",
        "trigger": "ONE SENTENCE SUMMARY",
    },
    "Consise": {
        "style": "Return your response as concise summary which covers the main points of the text.",
        "trigger": "CONCISE SUMMARY",
    },
    "Detailed": {
        "style": "Return your response as detailed summary which covers the main points of the text and key facts and figures.",
        "trigger": "DETAILED SUMMARY",
    },
}

LANGUAGES = ["Default", "English", "Polish", "Portuguese",
             "Spanish", "Czech", "Turkish", "French", "German", ]

# Model params
MODEL_FILE = "./models/mistral-7b-openorca.Q5_K_M.gguf"
MODEL_CONTEXT_WINDOW = 8192

# Chunk params in characters (not tokens)
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 500

llm = LlamaCpp(
    model_path=MODEL_FILE,
    n_ctx=MODEL_CONTEXT_WINDOW,
    # Don't be creative.
    temperature=0,
    max_tokens=MAX_TOKENS,
    verbose=VERBOSE,

    # Remove next two lines if NOT using macOS & M1 processor:
    n_batch=512,
    n_gpu_layers=1,
)


combine_prompt_template = """
Write a summary of the following text delimited by tripple backquotes.
{style}

```{content}```

{trigger} {in_language}:
"""

map_prompt_template = """
Write a concise summary of the following text which covers the main points and key facts and figures:
{text}

CONCISE SUMMARY {in_language}:
"""


def summarize_base(llm, content, style, language):
    """Summarize whole content at once. The content needs to fit into model's context window."""

    prompt = PromptTemplate.from_template(
        combine_prompt_template
    ).partial(
        style=STYLES[style]["style"],
        trigger=STYLES[style]["trigger"],
        in_language=f"in {language}" if language != "Default" else "",
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=VERBOSE)
    output = chain.run(content)

    return output


def summarize_map_reduce(llm, content, style, language):
    """Summarize content potentially larger that model's context window using map-reduce approach."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    split_docs = text_splitter.create_documents([content])
    print(
        f"Map-Reduce content splits ({len(split_docs)} splits): {[len(sd.page_content) for sd in split_docs]}")

    map_prompt = PromptTemplate.from_template(
        map_prompt_template
    ).partial(
        in_language=f"in {language}" if language != "Default" else "",
    )
    combine_prompt = PromptTemplate.from_template(
        combine_prompt_template
    ).partial(
        style=STYLES[style]["style"],
        trigger=STYLES[style]["trigger"],
        in_language=f"in {language}" if language != "Default" else "",
    )

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        combine_document_variable_name="content",
        verbose=VERBOSE,
    )

    output = chain.run(split_docs)
    return output


def load_input_file(input_file):
    if not input_file:
        return None

    start_time = time.perf_counter()

    if input_file.name.endswith(".pdf"):
        loader = PyPDFLoader(input_file.name)
        docs = loader.load()

        end_time = time.perf_counter()
        print(
            f"PDF: loaded {len(docs)} pages, in {round(end_time - start_time, 1)} secs")
        return "\n".join([d.page_content for d in docs])

    docs = TextLoader(input_file.name).load()

    end_time = time.perf_counter()
    print(f"Input file load time {round(end_time - start_time, 1)} secs")

    return docs[0].page_content


def summarize_text(content, style, language, progress=gr.Progress()):
    content_tokens = llm.get_num_tokens(content)

    print("Content length:", len(content))
    print("Content tokens:", content_tokens)
    print("Content sample:\n" + content[:200] + "\n\n")

    info = f"Content length: {len(content)} chars, {content_tokens} tokens."
    progress(None, desc=info)

    # Keep part of context window for models output & some buffor for the promopt.
    base_threshold = MODEL_CONTEXT_WINDOW - MAX_TOKENS - 256

    start_time = time.perf_counter()

    if (content_tokens < base_threshold):
        info += "\n"
        info += "Using summarizer: base"
        progress(None, desc=info)

        print("Using summarizer: base")
        summary = summarize_base(llm, content, style, language)
    else:
        info += "\n"
        info += "Using summarizer: map-reduce"
        progress(None, desc=info)

        print("Using summarizer: map-reduce")
        summary = summarize_map_reduce(llm, content, style, language)

    end_time = time.perf_counter()

    print("Summary length:", len(summary))
    print("Summary tokens:", llm.get_num_tokens(summary))
    print("Summary:\n" + summary + "\n\n")

    info += "\n"
    info += f"Processing time: {round(end_time - start_time, 1)} secs."
    info += "\n"
    info += f"Summary length: {llm.get_num_tokens(summary)} tokens."

    print("Info", info)
    return summary, info


with gr.Blocks() as ui:
    gr.Markdown(
        """
        # Summarization Tool
        Drop a file or paste text to summarize it!
        """,
    )

    input_file = gr.File(
        label="Drop a file here",
        file_types=["text", "pdf"],
    )

    input_text = gr.Textbox(
        label="Text to summarize",
        placeholder="Or paste text here...",
        lines=5,
        max_lines=15,
    )

    with gr.Row():
        style_radio = gr.Radio(
            choices=[s for s in STYLES.keys()],
            value=list(STYLES.keys())[0],
            label="Response style"
        )

        language_dropdown = gr.Dropdown(
            choices=LANGUAGES,
            value=LANGUAGES[0],
            label="Response language",
        )

    start_button = gr.Button("Generate Summary", variant="primary")

    with gr.Row():
        with gr.Column(scale=4):
            pass

    gr.Markdown(
        """
        ## Summary
        """
    )

    output_text = gr.Textbox(
        max_lines=25,
        show_copy_button=True,
    )

    info_text = gr.Textbox(
        label="Diagnostic info",
        max_lines=5,
        interactive=False,
        show_copy_button=True,
    )

    input_file.change(
        load_input_file,
        inputs=[input_file],
        outputs=[input_text]
    )

    start_button.click(
        summarize_text,
        inputs=[input_text, style_radio, language_dropdown],
        outputs=[output_text, info_text],
    )


ui.queue().launch(inbrowser=True)
