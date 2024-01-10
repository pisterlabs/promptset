import re
from os import path
from pickle import load as loadd, dump as dumpp

from json import load, dump
from click import group, argument, option, Choice
from rdflib import Graph
from halo import Halo
from transformers import AutoTokenizer, FalconModel
from torch import mean
from tqdm import tqdm

from llama_index import GPTVectorStoreIndex, download_loader, StorageContext, load_index_from_storage, QueryBundle

# from openai import ChatCompletion as cc

# from .OrkgContext import OrkgContext
from .similarity import compare as compare_strings, rank
from .SciQA import SciQA
from .Responder import Responder
from .util import drop_spaces
from .RDFReader import RDFReader
from .QueryEngine import QueryEngine


NEW_LINE = '\n'
MARK = '-' * 10

LABEL_LINE = re.compile(r'>(\s+)<\S+[A-Z]+[0-9]+>(\s+.+XMLSchema#string>)')


@group()
def main():
    pass


@main.command()
@argument('question', type = str, default = None, required = False)
@option('-d', '--dry-run', is_flag = True, help = 'Print generated context and exit')
@option('-f', '--fresh', is_flag = True, help = 'Don\'t use cached context entries, generate them from scratch')
@option('-c', '--cache-path', type = str, help = 'Path to the cached answers', default = 'assets/queries.pkl')
@option('-q', '--questions-path', type = str, help = 'Path to the file with questions', default = None)
@option('-g', '--graph-path', type = str, help = 'Path with .nt file with knowledge graph which should be used for generated query execution')
@option('--graph-cache', type = str, help = 'Path to the cached result of graph parsing', default = 'assets/orkg.pkl')
@option('-a', '--answers-path', type = str, help = 'Path to the output .json file with answers', default = 'assets/answers.json')
@option('-z', '--answer-cache-path', type = str, help = 'Path to file which contains cached results of generated sparql queries execution', default = 'assets/answers.pkl')
def ask(question: str, dry_run: bool, fresh: bool, cache_path: str, questions_path: str, graph_path: str, graph_cache: str, answers_path: str, answer_cache_path: str):
    graph = None

    # Parse graph or load from cache

    if graph_path is not None:
        if path.isfile(graph_cache):
            with open(graph_cache, 'rb') as file:
                graph = loadd(file)
        else:
            graph = Graph()

            print('parsing graph...')
            graph.parse(graph_path)

            with open(graph_cache, 'wb') as file:
                dumpp(graph, file)

    # Run queries, send them to the parsed graph, get answers and write them to an external file

    responder = Responder(cache_path, answer_cache_path, graph = graph)

    if questions_path is None:
        if question is None:
            raise ValueError('If questions-path is not provided, then question must be given as the first argument')

        answer = responder.ask(question, fresh = fresh, dry_run = dry_run)
        print(answer)
    else:
        with open(questions_path, 'r', encoding = 'utf-8') as file:
            content = load(file)

        n_matched_queries = 0
        n_queries = 0

        answers = []

        for i, entry in enumerate(content):
            try:
                question = entry["question"]["string"]
            except Exception:
                # print(e)
                question = entry['question']

            print(f'{i:03d}. {question}')

            # if i in (26, 53):  # these queries take a lot of time to process
            #     answers.append({
            #         'id': entry['id'],
            #         'answer': []
            #     })
            # else:
            query, answer = responder.ask(question)

            answers.append({
                'id': entry['id'],
                'answer': answer
            })

            if entry.get('query') is not None:
                if drop_spaces(query) == drop_spaces(entry['query']['sparql']):
                    n_matched_queries += 1
                else:
                    print(f'{MARK} Queries differ')
                    print(f'{MARK} Generated:')
                    print(query)
                    print(f'{MARK} Reference:')
                    print(entry['query']['sparql'])
                    print(MARK)

            # if len(answer) > 0:
            #     print(answer)

            n_queries += 1

        print('question precision: ', n_matched_queries / n_queries)

    with open(answers_path, 'w', encoding = 'utf-8') as file:
        dump(answers, file, indent = 4)

    # cache = None

    # if cache_path is not None and path.isfile(cache_path):
    #     with open(cache_path, 'rb') as file:
    #         cache = load(file)

    # if cache is not None:
    #     answer = cache.get(question)

    #     if answer is not None:
    #         print(answer)
    #         return

    # context = OrkgContext(fresh = fresh)

    # # if dry_run:
    # #     # print(context.description)
    # #     print(context.cut(question))
    # # else:
    # examples, graph = context.cut(question)

    # string_examples = []

    # for example in examples:
    #     string_examples.append(f'Also I know that for a similar question "{example.utterance}" the correct query is \n```\n{example.query}\n```.')

    # # print('\n'.join(string_examples))

    # content = f'''
    # I have a knowledge graph which includes the following fragment:

    # '
    # {graph}
    # '

    # Generate SPARQL query which allows to answer the question "{question}" using this graph

    # {NEW_LINE.join(string_examples)}
    # '''

    # if dry_run:
    #     answer = 'foo'
    # else:
    #     completion = cc.create(
    #         model = 'gpt-3.5-turbo',
    #         messages = [
    #             {
    #                 'role': 'user',
    #                 'content': content
    #             }
    #         ]
    #     )

    #     answer = completion.choices[0].message.content

    # print(answer)

    # if cache_path is not None:
    #     if cache is None:
    #         cache = {question: answer}
    #     else:
    #         cache[question] = answer

    #     with open(cache_path, 'wb') as file:
    #         dump(cache, file)


@main.command()
@argument('lhs', type = str)
@argument('rhs', type = str)
def compare(lhs: str, rhs: str):
    print(f'The similarity of strings "{lhs}" and "{rhs}" is {compare_strings(lhs, rhs)}')


@main.command()
@option('-n', '--top-n', type = int, default = 3)
def trace(top_n: int):
    # print(rank('foo', ['qux', 'o', 'fo'], top_n = 2))

    sciqa = SciQA()

    train_entries = sciqa.train.entries

    for test_utterance in sciqa.test.utterances[:1]:
        print(f'Test sample: {test_utterance}')
        print(f'Similar train samples: {rank(test_utterance, train_entries, top_n, get_utterance = lambda entry: entry.utterance)}')
        print('')

    # print(len(sciqa.train.utterances))
    # print(len(sciqa.test.utterances))


@main.command()
@option('-g', '--graph-path', help = 'path to the .nt file with input graph which should be embedded', default = 'assets/cities.nt')
@option('-c', '--cache-path', help = 'path to the resulting file with embedded graph', default = 'assets/cities')
@option('-d', '--device', help = 'device which to use for model execution', type = Choice(('cpu', 'cuda:0'), case_sensitive = True), default = 'cpu')
@option('-b', '--batch-size', help = 'how many documents to computed embeddings for at once', default = 4)
def embed(graph_path: str, cache_path: str, device: str, batch_size: int):
    # RDFReader = download_loader('RDFReader')
    tokenizer = AutoTokenizer.from_pretrained('Rocketknight1/falcon-rw-1b', device_map = device)
    tokenizer.pad_token = tokenizer.eos_token

    model = FalconModel.from_pretrained('Rocketknight1/falcon-rw-1b', device_map = device)
    # model = FalconModel.from_pretrained('Rocketknight1/falcon-rw-1b')

    def to_device(outputs: dict):
        return {
            'input_ids': outputs['input_ids'].to(device),
            'attention_mask': outputs['attention_mask'].to(device)
        }

    def embed_one(text: str):
        return mean(
            model(
                **to_device(
                    tokenizer(text, return_tensors = 'pt')
                )
            ).last_hidden_state,
            dim = (0, 1)
        ).to('cpu').tolist()

    def embed_batch(batch: list):
        texts = [document.text for document in batch]

        inputs = tokenizer(texts, return_tensors = 'pt', padding = True)
        outputs = model(**to_device(inputs))

        for embedding, document in zip(mean(outputs.last_hidden_state, dim = 1).to('cpu').tolist(), batch):
            document.embedding = embedding

    if path.isdir(cache_path):
        with Halo(text = 'Restoring index', spinner = 'dots'):
            index = load_index_from_storage(StorageContext.from_defaults(persist_dir = cache_path))
    else:
        # with Halo(text = 'Loading graph', spinner = 'dots'):
        documents = RDFReader().load_data(
            file = graph_path,
            max_document_size = 1,
            embed = embed_batch,
            batch_size = batch_size
            # embed = embed_one
        )

        # print([document.text for document in documents])

        # return

        # batch = []

        # print(documents[-2].embedding)

        # for document in documents:
        #     batch.append(document)

        #     # inputs = tokenizer(document.text, return_tensors = 'pt')

        #     # print(document.text)
        #     # print(inputs)
        #     # print(document.embedding)

        #     # outputs = model(**inputs)

        #     # print(outputs.last_hidden_state.shape)

        # if len(batch) > 0:
        #     embed_batch()

        # return

        # print(mean(outputs.last_hidden_state, dim = (0, 1)).shape)
        # print(len(mean(outputs.last_hidden_state, dim = (0, 1)).tolist()))

        # print(outputs.last_hidden_state)

        # with Halo(text = 'Generating embeddings', spinner = 'dots'):
        index = GPTVectorStoreIndex.from_documents(documents, show_progress = True)
        index.storage_context.persist(persist_dir = cache_path)

    query = 'List all places in a quoted Python array, then explain why'

    response = index.as_query_engine().query(
        QueryBundle(
            query_str = query,
            embedding = embed_one(query)
        )
    )

    # response = index.as_query_engine().query('What is the type of contribution template')
    print(response.response)


@main.command()
@argument('input-path')
@option('-o', '--output-path', help = 'path to the output file', default = 'assets/labelled-corpus.nt')
def normalize_labels(input_path: str, output_path: str):
    with open(input_path, 'r', encoding = 'utf-8') as input_file:
        with open(output_path, 'w', encoding = 'utf-8') as output_file:
            while True:
                line = input_file.readline()

                if not line:
                    break

                line = line[:-1]

                # print(line)
                output_file.write(LABEL_LINE.sub(r'>\g<1><http://www.w3.org/2000/01/rdf-schema#label>\g<2>', line) + '\n')


@main.command()
@argument('input-path', type = str)
@option('--output-path', '-o', type = str, default = 'assets/subgraph/answers.json')
def query(input_path: str, output_path: str):
    engine = QueryEngine()

    answers = []

    with open(input_path, mode = 'r', encoding = 'utf-8') as file:
        lines = [line for line in file.read().split('\n') if line]

    for line in tqdm(lines, desc = 'Running queries'):
        entry = eval(line)

        input_query = entry['llm_generated_query']

        parts = input_query.split('SELECT', maxsplit = 1)

        if len(parts) > 1:
            input_query = f'SELECT{parts[1]}'

        query = '\n'.join((
            'prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>',
            'prefix orkgc: <http://orkg.org/orkg/class/>',
            'prefix orkgp: <http://orkg.org/orkg/predicate/>',
            'prefix orkgr: <http://orkg.org/orkg/resource/>',
            '\n',
            input_query
        ))
        # query = f'\n\n\n{input_query}'  # add missing prefices

        results = engine.run(query, id_ := entry['id'])

        answers.append({'id': id_, 'answer': results})

    with open(output_path, 'w', encoding = 'utf-8') as file:
        dump(answers, file, indent = 4)


if __name__ == '__main__':
    main()
