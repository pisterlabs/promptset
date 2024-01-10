# Copyright (c) 2023, DjaoDjin inc.
# All rights reserved.

"""
Command to analyze ESG supportive evidence and answer questions
"""

import datetime, io, json, logging, os, shutil

import openai
import ocrmypdf
import requests
import tiktoken
from PyPDF2 import PdfReader
from dateutil.relativedelta import relativedelta
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from survey.api.sample import update_or_create_answer
from survey.models import Sample, Unit
from survey.queries import get_question_model
from survey.utils import datetime_or_now

from ...compat import urlsplit
from ...utils import get_supporting_documents

LOGGER = logging.getLogger(__name__)

MIN_ANSWER_LEN = 3
ANSWER_NOT_AVAILABLE = 'n/a'
QUESTION_NUM_NOISE_OFFSET = 3
AI_MODEL = 'gpt-4'
TOKEN_LIMIT = 8000
# derived by trial & error
OPENAPI_PROMPT_LEN = 2000


def get_tokens(contents):
    encoding = tiktoken.encoding_for_model(AI_MODEL)
    return encoding.encode(contents)

def extract_text_from_text_pdf(stream):
    contents = ''

    reader = PdfReader(stream)
    for page in reader.pages:
        text = page.extract_text()
        contents += f"{text}\n"
    return contents.strip()


def extract_text_from_pdf(stream):
    treshold = 5
    contents = extract_text_from_text_pdf(stream)

    if len(contents) > treshold:
        return contents

    stream.seek(0)
    out_io = io.BytesIO()

    ocrmypdf.ocr(stream, out_io, deskew=True, force_ocr=True,
        language='eng')
    ocr_contents = extract_text_from_text_pdf(out_io)
    return ocr_contents


def fetch_resource(url, cache_dir=None):
    if not url.lower().strip().startswith('http'):
        raise ValueError("URL %s does not starts with 'http'" % str(url))
    parts = urlsplit(url)
    filename = parts.path.strip('/').split('/')[-1]
    LOGGER.info("extracted filename '%s' from url %s", filename, url)
    if cache_dir:
        filename = os.path.join(cache_dir, filename)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
    _unused, ext = os.path.splitext(filename)
    if not ext:
        filename += '.html'

    if not os.path.exists(filename):
        resp = requests.get(url, stream=True, timeout=10) # 10 seconds
        if resp.status_code != 200:
            LOGGER.error("fetching URL %s returns status code %d",
                url, resp.status_code)
            return ""

        if filename.endswith('.pdf'):
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(resp.raw, out_file)
        else:
            with open(filename, 'wb') as out_file:
                out_file.write(resp.content)

    if filename.endswith('.pdf'):
        with open(filename, 'rb') as in_file:
            content = in_file.read()
            content = extract_text_from_pdf(io.BytesIO(content))
    else:
        with open(filename, 'r') as in_file:
            content = in_file.read()

    return content


def decode_tokens(tokens_list):
    encoding = tiktoken.encoding_for_model(AI_MODEL)
    return [encoding.decode(tokens) for tokens in tokens_list]


def maybe_split_input_into_chunks(prime, evidence):
    prime_tokens = get_tokens(prime)
    evidence_tokens = get_tokens(evidence)
    prompt_len = len(prime_tokens) + len(evidence_tokens) + OPENAPI_PROMPT_LEN
    if prompt_len <= TOKEN_LIMIT:
        tokens = [evidence_tokens]
    else:
        chunk_len = TOKEN_LIMIT - len(prime_tokens) - OPENAPI_PROMPT_LEN
        res = []
        rem = evidence_tokens[:]
        while len(rem) > 0:
            res.append(rem[0:chunk_len])
            rem = rem[chunk_len:]
        tokens = res
    return decode_tokens(tokens)


def openapi_call(prime, evidence):
    outputs = []
    for chunk in evidence:
        completion = openai.ChatCompletion.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": prime},
                {"role": "user", "content": chunk},
            ]
        )
        outputs.append(completion.choices[0].message['content'])
    return outputs


def map_responses_to_questions(responses, questions_by_path,
                               evidence_chunks, url):
    questions_list = list(questions_by_path.items())
    result = {}
    for (chunk_idx, response) in enumerate(responses):
        answers = response.split('\n')
        # TODO we can potentially still use the response in
        # situations where there's less answers than there are
        # questions
        if len(answers) != len(questions_list):
            LOGGER.error(
                "Chunck %d: cannot map %d OpenAPI responses to %d questions",
                chunk_idx, len(answers), len(questions_list))
            continue

        for (question_idx, answer) in enumerate(answers):
            question_num_str = str(question_idx + 1)

            # we are not interested in blank or n/a answers
            if (ANSWER_NOT_AVAILABLE in answer.lower() or
                len(answer.strip()) < MIN_ANSWER_LEN):
                continue

            # basic sanity checks to ensure we are mapping the right answer
            if (question_num_str not in
                answer[0:len(question_num_str) + QUESTION_NUM_NOISE_OFFSET]):
                LOGGER.error(
                    "Chunck %d: cannot map OpenAPI answer to question num %d",
                    chunk_idx, question_num_str)
                continue

            question_path = questions_list[question_idx][0]
            if question_path not in result:
                result[question_path] = []

            result[question_path].append({
                'answer': answer.strip(),
                'chunk': evidence_chunks[chunk_idx],
                'source': url,
            })
    return result


def denormalize_answers_to_str(answers):
    # TODO we are currently ignoring the file chunk where
    # the answer was found
    result = ''
    for answer in answers:
        result += answer['answer']
        result += '\n\n Source:'
        result += answer['source']
        result += '\n\n ======== \n\n'
    return result


def persist_answers(oa_answers, sample):
    """
    Presists OpenAI answers into the database
    """
    collected_by = get_user_model().objects.get(username='chatgpt')
    unit = Unit.objects.get(slug='ai-audit')
    created_at = datetime_or_now()

    results = []
    for (question_path, answers) in oa_answers.items():
        openai_answer = denormalize_answers_to_str(answers)
        question = get_question_model().objects.get(path=question_path)
        obj, _ = update_or_create_answer(
            datapoint={'measured': openai_answer, 'unit': unit},
            sample=sample, question=question, created_at=created_at,
            collected_by=collected_by)
        results.append(obj)

    return results


def process_file(url, questions_by_path, cache_dir=None, fixtures=None):
    """
    Fetches `url` from the Internet, extract the text, upload it to
    OpenAI and asks the questions.

    If you pass a `fixtures` argument, the function bypasses the OpenAI step.
    """
    questions_list = list(questions_by_path.items())
    questions_str = ""
    for (idx, question) in enumerate(questions_list):
        questions_str += f"{idx+1}. {question[1]} [ANSWER HERE]\n"

    prime = """You are an expert ESG (Environment, Social and Governance)
auditor, you are producing ESG performance report of a specific company.
The report contains a set of questions that you have to answer with "yes"
or "n/a". Each question is related to a specific ESG topic. The company
provided a document with data that has to be used to obtain the answers
to the questions. The document is a PDF file or a scanned image, you've
already extracted the text from the document. For each question, you need
to analyze the document and look for information that can be used to answer
the question. The document might have or might not have enough information
to answer the questions. If you were able to find information in the document
to answer a question, answer "yes" and include this information (and date of
the information if applicable) in your answer. If you weren't able to find
information which would've helped you to answer a question, you should answer
"n/a". Return the report, in the report each question-answer must follow this
format:


        [Number] - [Answer] - [Supportive evidence (if answer is \"yes\")]




        Here's the list of questions you should answer:

%(questions)s


Here's the extracted text from the document that you have to analyze and use
to answer the questions:


""" % {'questions': questions_str}

    text = fetch_resource(url, cache_dir=cache_dir)
    LOGGER.info(
        "extracted text from url %s, splitting evidence chuncks...", url)
    evidence_chunks = maybe_split_input_into_chunks(prime, text)
    LOGGER.info(
        "split evidence chuncks, fetching responses from OpenAI...")
    if not fixtures:
        responses = openapi_call(prime, evidence_chunks)
    else:
        responses = fixtures
    LOGGER.info(
        "fetched responses from OpenAI, mapping answers...")
    results = map_responses_to_questions(
        responses, questions_by_path, evidence_chunks, url)
    return results


class Command(BaseCommand):
    help = """Decorate answers with AI verification hints"""

    def add_arguments(self, parser):
        super(Command, self).add_arguments(parser)
        parser.add_argument('--dry-run', action='store_true',
            dest='dry_run', default=False,
            help='Do not commit results to the database')
        parser.add_argument('--fixtures', action='store',
            dest='fixtures', default=None,
            help='JSON file with fixtures to use for OpenAI responses (bypass)')
        parser.add_argument('--questions', action='store',
            dest='questions', default=None,
            help='JSON file with a dictionnary of `{path: title}` questions')
        parser.add_argument('samples', nargs='+',
            help='slugs of samples to decorate with hints')

    def handle(self, *args, **options):
        #pylint:disable=too-many-locals
        start_time = datetime.datetime.utcnow()
        dry_run = options['dry_run']
        fixtures = None
        if options['fixtures']:
            with open(options['fixtures']) as fixtures_file:
                fixtures = json.load(fixtures_file)
        questions_by_path = {}
        if options['questions']:
            with open(options['questions']) as questions_file:
                questions_by_path = json.load(questions_file)

        openai.api_key = settings.OPENAI_API_KEY
        for sample in Sample.objects.filter(slug__in=options['samples']):
            sample_start_time = datetime.datetime.utcnow()
            cache_dir = os.path.join(settings.RUN_DIR, str(sample))
            self.stdout.write("fetching URLs into %s" % cache_dir)
            public_docs, _unused = get_supporting_documents([sample])
            hints = {}
            if not questions_by_path:
                questions_by_path = {question.path: question.title
                    for question in get_question_model().filter(
                    campaign=sample.campaign)}
            for doc in public_docs:
                doc_start_time = datetime.datetime.utcnow()
                self.stdout.write('processing %s ...' % str(doc))
                results = process_file(doc, cache_dir=cache_dir,
                    questions_by_path=questions_by_path, fixtures=fixtures)
                LOGGER.info("processing %s returns %s", doc, results)
                for key, values in results.items():
                    if key not in hints:
                        hints.update({key: values})
                    else:
                        hints[key] += values

                end_time = datetime.datetime.utcnow()
                delta = relativedelta(end_time, doc_start_time)
                LOGGER.info(
                "completed document %s in %d hours, %d minutes, %d.%d seconds",
                    doc, delta.hours, delta.minutes, delta.seconds,
                    delta.microseconds)
                self.stdout.write(
                "completed document %s in %d hours, %d minutes, %d.%d seconds\n"
                    % (doc, delta.hours, delta.minutes, delta.seconds,
                       delta.microseconds))

            if not dry_run:
                persist_answers(hints, sample)

            end_time = datetime.datetime.utcnow()
            delta = relativedelta(end_time, sample_start_time)
            LOGGER.info(
                "completed sample %s in %d hours, %d minutes, %d.%d seconds",
                sample, delta.hours, delta.minutes, delta.seconds,
                delta.microseconds)
            self.stdout.write(
                "completed sample %s in %d hours, %d minutes, %d.%d seconds\n"
                % (sample, delta.hours, delta.minutes, delta.seconds,
                   delta.microseconds))

        end_time = datetime.datetime.utcnow()
        delta = relativedelta(end_time, start_time)
        LOGGER.info("completed in %d hours, %d minutes, %d.%d seconds",
            delta.hours, delta.minutes, delta.seconds, delta.microseconds)
        self.stdout.write("completed in %d hours, %d minutes, %d.%d seconds\n"
            % (delta.hours, delta.minutes, delta.seconds, delta.microseconds))
