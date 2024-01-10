from time import sleep
from os import path
from pickle import load, dump

from openai import ChatCompletion as cc
from rdflib import Graph

from .OrkgContext import OrkgContext


NEW_LINE = '\n'

# PROMPT = '''
# I have a knowledge graph which includes the following fragment:
#
# '
# {graph}
# '
#
# Generate SPARQL query which allows to answer the question "{question}" using this graph
#
# {examples}
# '''

PROMPT = '''
Generate SPARQL query which allows to answer the question "{question}".

{examples}
'''


def load_cache(cache_path: str):
    if cache_path is not None and path.isfile(cache_path):
        with open(cache_path, 'rb') as file:
            return load(file)

    return None


def dump_cache(cache_path: str, content):
    if cache_path is not None:
        with open(cache_path, 'wb') as file:
            dump(content, file)


class Responder:
    def __init__(self, query_cache_path: str, answer_cache_path: str, graph: Graph = None):
        self.query_cache_path = query_cache_path
        self.answer_cache_path = answer_cache_path

        self.query_cache = load_cache(query_cache_path)
        self.answer_cache = load_cache(answer_cache_path)

        self.graph = graph

    def _extract_query(self, answer: str):
        parts = answer.replace('```sparql', '```').split('```')

        if len(parts) != 3:
            return answer
            # raise ValueError(f'Can\'t extract query from answer "{answer}"')

        return parts[1]

    def _execute(self, question: str, answer: str, context: OrkgContext):
        query = self._extract_query(answer)

        answer_cache = self.answer_cache

        cached_results = None if answer_cache is None else answer_cache.get(question)

        if cached_results is not None:
            return query, cached_results

        results = context.get_triples(query)

        print(results)

        if answer_cache is None:
            answer_cache = {question: results}
            self.answer_cache = answer_cache
        else:
            answer_cache[question] = results

        dump_cache(self.answer_cache_path, answer_cache)

        return query, results

    def ask(self, question: str, fresh: bool = False, dry_run: bool = False):
        query_cache = self.query_cache

        context = OrkgContext(fresh = fresh, graph = self.graph)

        if query_cache is not None and not dry_run:
            answer = query_cache.get(question)

            if answer is not None:
                return self._execute(question, answer, context)

        # examples, graph = context.cut(question)
        examples, _ = context.cut(question)

        string_examples = []

        for example in examples:
            # string_examples.append(f'Also I know that for a similar question "{example.utterance}" the correct query is \n```\n{example.query}\n```.')
            string_examples.append(f'I know that for a similar question "{example.utterance}" the correct query is \n```\n{example.query}\n```.')

        # content = PROMPT.format(graph = graph, examples = NEW_LINE.join(string_examples))
        content = PROMPT.format(examples = NEW_LINE.join(string_examples), question = question)

        if dry_run:
            answer = content
        else:
            completion = cc.create(
                model = 'gpt-3.5-turbo',
                messages = [
                    {
                        'role': 'user',
                        'content': content
                    }
                ]
            )

            sleep(20)

            answer = completion.choices[0].message.content

        if query_cache is None:
            query_cache = {question: answer}
            self.query_cache = query_cache
        else:
            query_cache[question] = answer

        dump_cache(self.query_cache_path, query_cache)

        return self._execute(question, answer, context)
