from django.core.management import BaseCommand
from openai import OpenAI
from tqdm import tqdm

from evaluation_registry import settings
from evaluation_registry.evaluations.models import Evaluation

CHATGPT_ROLE = """
You are a plain text formatter.

You will receive badly formatted text and reformat it with:
* proper capitalization of abbreviations, proper nouns and sentences
* sensible whitespace

Do not:
* remove or add words
* change the tone of the text

Please return the reformatted text without explanation.
"""

client = OpenAI(api_key=settings.OPENAI_KEY)


def reformat_text(txt: str | None) -> str | None:
    if not txt:
        return txt

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": CHATGPT_ROLE},
            {"role": "user", "content": txt},
        ],
        model="gpt-3.5-turbo",
    )
    if not chat_completion.choices:
        raise ValueError("no data returned")
    return chat_completion.choices[0].message.content


def reformat_evaluation(evaluation: Evaluation):
    new_title = reformat_text(evaluation.title)
    if new_title and len(new_title) <= 1024:
        evaluation.title = new_title
    evaluation.brief_description = reformat_text(evaluation.brief_description)
    evaluation.save()


class Command(BaseCommand):
    help = "Use ChatGPT to reformat evaluation title and description"

    def add_arguments(self, parser):
        parser.add_argument("max_number_to_process", type=int, nargs="?", default=None)

    def handle(self, *args, **options):
        max_number_to_process = options["max_number_to_process"] or Evaluation.objects.count()

        progress_bar = tqdm(desc="Processing", total=max_number_to_process)
        for evaluation in Evaluation.objects.filter(rsm_evaluation_id__isnull=False).order_by("rsm_evaluation_id")[
            :max_number_to_process
        ]:
            progress_bar.set_description(f"updating rsm-evaluation-id: {evaluation.rsm_evaluation_id}")
            reformat_evaluation(evaluation)
            progress_bar.update(1)

        progress_bar.close()
        self.stdout.write(self.style.SUCCESS("reformatting text complete"))
