# RAG/TAG Tiger
# Copyright (c) 2024 Stuart Riffle

import os, argparse, time, datetime, json, pathspec
from transformers import AutoModelForCausalLM
from llama_index import VectorStoreIndex, StorageContext
from llama_index import load_index_from_storage, download_loader
from util.code_aware_directory_reader import CodeAwareDirectoryReader
from util.find_files_using_gitignore import find_files_using_gitignore

program_name = "RAG/TAG Tiger"
program_version = "0.1.0"
program_license = "MIT"
program_copyright = "Copyright (c) 2024 Stuart Riffle"

parser = argparse.ArgumentParser(
    description="Update and query a vector document store using LlamaIndex", 
    fromfile_prefix_chars='@')

arg = parser.add_argument
arg("-v", "--verbose",  help="enable extended/debug output", action="store_true")
arg("--version",        help="print the version number and exit", action="store_true")

arg = parser.add_argument_group("Vector database").add_argument
arg("--index-load",     help="Load the vector index from a given path", metavar="PATH")
arg("--index-store",    help="Save the updated vector index to a given path", metavar="PATH")

arg = parser.add_argument_group("Document indexing").add_argument
arg("--source-folder",  help="Folder of files to be indexed recursively", nargs="+", metavar="FOLDER")
arg("--source-spec",    help="Index files matching a pathspec, like \"**/*.(cpp|hpp|h)\"", nargs="+", metavar="SPEC")
arg("--source-list",    help="Text file with a list of filenames/pathspecs to index", nargs="+", metavar="FILE")
arg("--exclude-spec",   help="Exclude files matching a pathspec, like \"**/.vs/**/*\"", nargs="+", metavar="SPEC")
arg("--use-gitignore",  help="Exclude files specified by .gitignore files", action="store_true")
arg("--custom-loader",  help="Use loaders from LlamaIndex hub, specify like \"JPEGReader:jpg,jpeg\"" , nargs="+", metavar="SPEC")

arg = parser.add_argument_group("Language model").add_argument
arg("--llm-model-name", help="Path or HF model for local inference (overrides server and provider)", metavar="PATH")
arg("--llm-provider",   help="Commercial inference provider (overrides server)", choices=["openai", "claude", "bing", "gemini"])
arg("--llm-server",     help="LLM inference server URL", default="http://localhost:8080", metavar="URL")
arg("--llm-api-key",    help="API key for inference server (if needed)", default="", metavar="KEY")
arg("--llm-secret",     help="Secret for inference server (if needed)", default="", metavar="SECRET")
arg("--llm-param",      help="Inference parameter, like \"temperature=0.9\" etc", nargs="+", metavar="KVP")
arg("--context",        help="Command line context/system prompt", nargs="+", metavar="TEXT")
arg("--context-file",   help="File containing a snippet of context", nargs="+", metavar="FILE")

arg = parser.add_argument_group("Query processing").add_argument
arg("--query",          help="Command line query", nargs="+", metavar="TEXT")
arg("--query-list",     help="File containing short queries, one per line", nargs="+", metavar="FILE")
arg("--query-file",     help="File containing one long query", nargs="+", metavar="FILE")
arg("--query-log",      help="Log queries and responses to a text file", metavar="FILE")
arg("--query-log-json", help="Log queries and responses (plus some metadata) to a JSON file", metavar="FILE")
arg("--query-memory",   help="Maintain \"chat\" history between queries", action="store_true")
arg("--query-mode",     help="Query response mode", choices=["accumulate", "compact", "compact_accumulate", "generation", "no_text", "refine", "simple_summarize", "tree_summarize"], default="tree_summarize")
arg("--tag-queries",    help="The name/header in the transcript for user queries", metavar="NAME", default="Query")
arg("--tag-responses",  help="The name/header in the transcript for engine responses", metavar="NAME", default="Response")

arg = parser.add_argument_group("Interactive chat").add_argument
arg("--chat",           help="Enter chat after any query processing", action="store_true")
arg("--chat-init",      help="Extra instructions/personality for the chat LLM", nargs="+", metavar="TEXT")
arg("--chat-init-file", help="File containing a snippet of chat LLM instructions", nargs="+", metavar="FILE")
arg("--chat-log",       help="Append chat queries and responses to a text file", metavar="FILE")
arg("--chat-mode",      help="Chat response mode", choices=["best", "context", "condense_question", "simple", "react", "openai"], default="best")

print(f"{program_name} {program_version}\n")
args = parser.parse_args()
if args.version:
    exit(0)

def print_verbose(msg, **kwargs):
    if args.verbose:
        print(msg, **kwargs)

# Let's go, tiger

start_time = time.time()

# Find all the files to be indexed

files_to_load = []

for folder in args.source_folder or []:
    print(f"Searching for files under \"{folder}\"...")
    folder_files = find_files_using_gitignore(folder, args.use_gitignore)
    files_to_load.extend(folder_files)
    print_verbose(f"\t{len(folder_files)} found")

for spec in args.source_spec or []:
    print(f"Finding files matching \"{spec}\"...")
    spec = pathspec.PathSpec.from_lines('gitwildmatch', [spec])
    spec_files = os.glob(spec, recursive=True, include_files=True)
    files_to_load.extend(spec_files)
    print_verbose(f"\t{len(spec_files)} found")

for file in args.source_list or []:
    print(f"Loading file/spec list from from \"{file}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            source_list_files = f.read().splitlines()
            source_list_files = [name for name in source_list_files if name.strip()]
    except:
        print(f"\tERROR: failure while loading file list")

    for file_spec in source_list_files or []:
        print_verbose(f"\t{file_spec}")
        found_files = []
        if os.path.isfile(file_spec):
            found_files.append(file_spec)
        else:
            spec = pathspec.PathSpec.from_lines('gitwildmatch', [file_spec])
            spec_files = os.glob(spec, recursive=True, include_files=True)
            found_files.extend(spec_files)
        files_to_load.extend(found_files)

print_verbose(f"\t{len(files_to_load)} total files found to index")

# Filter out ones we don't actually want

if len(files_to_load) > 0:    
    for spec in args.exclude_spec or []:
        print(f"Excluding files matching \"{spec}\"...")
        previous_length = len(files_to_load)
        spec = pathspec.PathSpec.from_lines('gitwildmatch', [spec])
        files_to_load = [f for f in files_to_load if not spec.match_file(f)]
        if len(files_to_load) < previous_length:
            print_verbose(f"\t{previous_length - len(files_to_load)} excluded")

if len(files_to_load) > 0:
    print(f"Filtering out duplicate entries...")
    previous_length = len(files_to_load)
    files_to_load = [os.path.normpath(os.path.realpath(f)) for f in files_to_load]
    files_to_load = sorted(set(files_to_load))
    if len(files_to_load) < previous_length:
        print_verbose(f"\t{previous_length - len(files_to_load)} removed")

if len(files_to_load) > 0:
    print(f"Removing non-files...")
    previous_length = len(files_to_load)
    files_to_load = [f for f in files_to_load if os.path.isfile(f)]
    if len(files_to_load) < previous_length:
        print_verbose(f"\t{previous_length - len(files_to_load)} removed")

# Download custom loaders from the hub

default_loaders = ["JSONReader:json"]
loader_specs = default_loaders
loader_specs.extend(args.custom_loaders or [])

file_extractor_list = {}
print("Downloading file loaders...")
for loader_spec in loader_specs:
    if not ':' in loader_spec:
        print(f"\tWARNING: invalid loader spec \"{loader_spec}\"")
        continue
    loader_class, extensions = loader_spec.split(":", 1)
    print_verbose(f"\t{loader_class} ({extensions.join(', ')})")
    try:
        loader = download_loader(loader_class)
        for extension in extensions.split(","):
            file_extractor_list["." + extension.strip(". ")] = loader()
    except:
        print(f"\tWARNING: failure downloading \"{loader_class}\"")

# Load and chunk the documents

if len(files_to_load) > 0:
    print(f"Loading and chunking {len(files_to_load)} files...")
    doc_reader = CodeAwareDirectoryReader(
        input_files=files_to_load, 
        file_extractor=file_extractor_list, 
        exclude_hidden=True,
        verbose=args.verbose)
    try:
        time_before_loading_docs = time.time()
        documents_to_index = doc_reader.load_data(show_progress=args.verbose)
        print_verbose(f"\tfinished in {time.time() - time_before_loading_docs:.3f} seconds")
    except:
        print(f"\tERROR: failure while loading documents")

# Update the vector index

if args.index_load:
    print(f"Loading existing vector index from \"{args.index_load}\"...")
    storage_context = StorageContext.from_defaults(persist_dir=args.index_load)
    try:
        time_before_loading_index = time.time()
        vector_index = load_index_from_storage(storage_context, show_progress=args.verbose)    
        print_verbose(f"\tloaded in {time.time() - time_before_loading_index:.3f} seconds")
    except:
        print(f"\tERROR: failure while loading index")

if not vector_index:
    print_verbose(f"Creating a new vector index in memory...")
    vector_index = VectorStoreIndex(show_progress=args.verbose)

if len(documents_to_index or []) > 0:
    print(f"Indexing {len(documents_to_index)} documents...")
    time_before_indexing = time.time()
    for doc in documents_to_index:
        print_verbose(f"\t    {doc.file_path}", end="")
        try:
            time_before_indexing_doc = time.time()
            vector_index.add_document(doc)
            print_verbose(f" ({time.time() - time_before_indexing_doc:.3f} seconds)")
        except:
            print(f" [ERROR]")

    print_verbose(f"\tindexing complete in {time.time() - time_before_indexing:.3f} seconds")

if args.index_store:
    print(f"Storing vector index in \"{args.index_store}\"...")
    try:
        time_before_storing_index = time.time()
        vector_index.storage_context.persist(persist_dir=args.index_store, show_progress=args.verbose)
        print_verbose(f"\tstored in {time.time() - time_before_storing_index:.3f} seconds")
    except:
        print(f"\tERROR: failure while storing index")

# Construct the LLM context/system prompt from text snippets

context = ""
snippets = args.context or []

if len(snippets) > 0:
    print(f"Adding system context from the command line...")
    for snippet in snippets:
        context += snippet.strip() + "\n"
        print_verbose(snippet)

for file in args.context_file or []:
    print(f"Adding system context from \"{file}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            snippet = f.read()
            context += snippet.strip() + "\n"
            print_verbose(snippet)        
    except:
        print(f"\tERROR: failure while loading context")

# Collect all the user queries

queries = args.query or []

for file in args.query_list or []:
    print(f"Loading single-line queries from \"{file}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            short_queries = f.read().splitlines()
            short_queries = [q for q in short_queries if q.strip()]
            queries.extend(short_queries)
            print_verbose(f"\t{len(short_queries)} found")
    except:
        print(f"\tERROR: failure while loading queries")

for file in args.query_file or []:
    print(f"Loading a long query from \"{file}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            query_text = f.read().strip()
            queries.append(query_text)
    except:
        print(f"\tERROR: failure while loading query")

# Initialize the LLM query engine

query_engine_params = {
    "response_mode": args.query_mode,
    "show_progress": args.verbose,
}

if args.llm_model_name:
    print(f"Query engine will run a local instance of model \"{args.llm_model_name}\"...")
    if args.llm_provider:
        print_verbose(f"\tWARNING: overriding inference provider \"{args.llm_provider}\"")
    if args.llm_server:
        print_verbose(f"\tWARNING: overriding inference API server \"{args.llm_server}\"")

    try:
        time_before_loading_model = time.time()
        local_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name)
        print_verbose(f"\tmodel loaded in {time.time() - time_before_loading_model:.3f} seconds")
    except:
        print(f"\tERROR: failure while loading model")

    try:
        time_before_query_engine = time.time()
        query_engine = vector_index.as_query_engine(model=local_model, **query_engine_params)
        print_verbose(f"\tquery engine initialized in {time.time() - time_before_query_engine:.3f} seconds")
    except:
        print(f"\tERROR: failure while initializing query engine")

if args.llm_provider and not query_engine:
    print(f"Query engine will use LLM inference provider \"{args.llm_provider}\"...")
    if args.llm_server:
        print_verbose(f"\tWARNING: overriding inference API server \"{args.llm_server}\"")

    try:
        time_before_connecting = time.time()
        query_engine = vector_index.as_query_engine(provider=args.llm_provider, api_key=args.llm_api_key, **query_engine_params)
        print_verbose(f"\tconnected in {time.time() - time_before_connecting:.3f} seconds")
    except:
        print(f"\tERROR: failure while connecting to provider")

if args.llm_server and not query_engine:
    print(f"Query engine will use inference API server \"{args.llm_server}\"...")
    try:
        time_before_connecting = time.time()
        query_engine = vector_index.as_query_engine(server_url=args.llm_server, api_key=args.llm_api_key, **query_engine_params)
        print_verbose(f"\tconnected in {time.time() - time_before_connecting:.3f} seconds")
    except:
        print(f"\tERROR: failure while connecting to server")

if not query_engine:
    print(f"ERROR: no query engine available")
    exit(1)

# Run all the queries
    
if len(queries) > 0:
    print(f"Running {len(queries)} queries...")
    time_before_queries = time.time()
    text_log = ""
    json_log = {
        "model": query_engine.model_name,
        "timestamp": datetime.now().isoformat(),
        "context": context,
        "queries": []
    }

    for query in queries:
        preface = context
        if args.query_memory:
            preface += text_log

        try:
            query_start_time = time.time()
            response = query_engine.query(f"{preface}\n{args.tag_queries}: {query}", verbose=args.verbose)
            response_time = time.time() - query_start_time
        except:
            print(f"\tERROR: failure while running query")
            continue

        interaction = f"{args.tag_queries}: {query}\n{args.tag_responses}: {response}\n"
        print_verbose(interaction)
        
        text_log += interaction
        json_log["queries"].append({
            "query": query, 
            "response": response, 
            "latency": response_time,
        })

    # Commit the logs

    if args.output_text:
        print(f"Writing log to \"{args.output_text}\"...")
        try:
            with open(args.output_text, "w", encoding="utf-8") as f:
                f.write(text_log)
        except:
            print(f"\tERROR: failure while writing log")

    if args.output_json:
        print(f"Writing JSON log to \"{args.output_json}\"...")
        try:
            with open(args.output_json, "w", encoding="utf-8") as f:
                raw_text = json.dumps(json_log, indent=4)
                f.write(raw_text)
        except:
            print(f"\tERROR: failure while writing JSON log")

    # Queries all done!

    print(f"All queries completed in {time.time() - time_before_queries:.3f} seconds\n")

# Chat mode

if args.chat:
    chat_init = args.chat_init or []
    for file in args.chat_init_file or []:
        print_verbose(f"Loading chat context/instructions from \"{file}\"...")
        try:
            with open(file, "r", encoding="utf-8") as f:
                chat_init_text = f.read().strip()
                chat_init.append(chat_init_text)
                print_verbose(chat_init_text)
        except:
            print(f"\tERROR: failure while loading chat context")

    print(f"Entering interactive chat...")
    print(f" - The response mode is \"{args.chat_mode}\"")
    print(f" - Hit CTRL-C to interrupt a response in progress")
    print(f" - Say \"bye\" or something like that when you're done")
    print()
    
    chat_engine_params = {
        "chat_mode": args.chat_mode,
        "system_prompt": f"{context}\n{chat_init}",
        "verbose": args.verbose, 
    }

    try:
        if local_model:
            chat_engine = vector_index.as_chat_engine(model=local_model, **chat_engine_params)
        else:
            chat_engine = vector_index.as_chat_engine(server_url=args.llm_server, api_key=args.llm_api_key, **chat_engine_params)
    except:
        print(f"\tERROR: failure while initializing chat engine")
        exit(1)

    chat_lines = []
    exit_commands = ["bye", "goodbye", "exit", "quit", "done", "stop", "end", "peace"]

    while True:
        try:
            message = input("> ")
            if message.strip().lower() in exit_commands:
                break
        except KeyboardInterrupt:
            continue

        chat_lines.append(f"{args.tag_queries}: {message}")
        response_line = ""

        try:
            streaming_response = chat_engine.chat(message, streaming=True)
            for token in streaming_response.response_gen:
                response_line += token
                print(token, end="")
        except KeyboardInterrupt:
            print("[response interrupted]")

        chat_lines.append(f"{args.tag_responses}: {response_line}")
        print()

    if args.chat_log and len(chat_lines) > 0:
        write_action = "Appending" if os.path.exists(args.chat_log) else "Writing"
        print(f"{write_action} this chat log to \"{args.chat_log}\"...")
        try:
            with open(args.chat_log, "a", encoding="utf-8") as f:
                all_lines = "\n".join(chat_lines) 
                f.write(all_lines + "\n")
        except:
            print(f"\tERROR: failure while writing chat log")

# Nice job, tiger
            
print(f"Done in {time.time() - start_time:.3f} seconds")
