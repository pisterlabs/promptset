import datetime
import os
import traceback
from time import sleep
from typing import Optional

from govscentdotorg.models import Bill, BillTopic, BillSection
import openai

WORDS_MAX = 9800
model = "gpt-3.5-turbo-16k"

bill_save_excluded_fields = {'title', 'text', 'bill_sections', 'topics', 'smells'}
# automatically populate a list with all fields, except the ones you want to exclude
bill_fields_to_update = [f.name for f in Bill._meta.get_fields() if
                         f.name not in bill_save_excluded_fields and not f.auto_created]


def openai_with_rate_limit_handling(prompt: str, retry: Optional[bool]):
    try:
        completion = openai.ChatCompletion.create(model=model, messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        return completion
    except openai.error.RateLimitError as e:
        if retry:
            print('Got RateLimitError, waiting two minutes and trying again.', e)
            sleep(120)
            return openai_with_rate_limit_handling(prompt=prompt, retry=False)
        else:
            raise e


def extract_response_topics(bill: Bill, response: str) -> [str]:
    [top_10_index, is_single_topic, is_just_topic_list] = get_top_10_index(bill, response)
    lines = response[top_10_index:].splitlines()
    if is_single_topic:
        if len(lines[0]) > 10:
            # Example: Topic: A sunny day in canada
            return [lines[0].replace("Topic:", "").strip()]
        else:
            line = lines[1]
            if line.isnumeric():
                # Example: 1. H.R. 5889 - a bill introduced in the House of Representatives.
                first_period_index = line.find(".")
                if first_period_index > -1:
                    line_after_first_number = line[first_period_index + 1:].strip()
                    return [line_after_first_number]
                else:
                    line_after_first_number = line[1:].strip()
                    return [line_after_first_number]
            else:
                return line.strip()
    else:
        topics = []
        # lines_slice is 11 lines because the first line could be the "Top 10..." header.
        lines_slice = lines[0:] if is_just_topic_list else lines[0:11]
        for line in lines_slice:
            if len(line) > 2 and not line.startswith('Top 10'):
                if line[0].isnumeric() or line.startswith("-") or line.find(':') > -1 or is_just_topic_list:
                    # Example: 1. H.R. 5889 - a bill introduced in the House of Representatives.
                    first_period_index = line.find(".")
                    if -1 < first_period_index < 4:
                        line_after_first_number = line[first_period_index + 1:].strip()
                        final_version = trim_topic_dash_ten_on_end(line_after_first_number)
                        if final_version is not None:
                            topics.append(final_version)
                    elif line.find(':') > -1:
                        first_colon_index = line.find(':')
                        line_after_first_char = line[first_colon_index + 1:].strip()
                        final_version = trim_topic_dash_ten_on_end(line_after_first_char)
                        if final_version is not None:
                            topics.append(final_version)
                    elif line.startswith("-"):
                        line_after_first_char = line[1:].strip()
                        final_version = trim_topic_dash_ten_on_end(line_after_first_char)
                        if final_version is not None:
                            topics.append(final_version)
                    elif is_just_topic_list:
                        topics.append(line)
                elif not is_just_topic_list:
                    # end of topics
                    break
        return topics


def trim_topic_dash_ten_on_end(text: str) -> str | None:
    slash_ten_index = text.rfind('/10')
    if slash_ten_index > -1:
        # Example: "Some topic - 5/10" - We don't want the 5/10 on the end.
        if text.endswith('/10') or text.endswith('/10.'):
            # Subtract 2 to remove digit before /10.
            line_dash_ten_trimmed = text[:slash_ten_index - 2].strip()
            if line_dash_ten_trimmed.endswith('-'):
                line_dash_ten_trimmed = line_dash_ten_trimmed[:len(line_dash_ten_trimmed) - 1].strip()
            if len(line_dash_ten_trimmed) > 6:
                return line_dash_ten_trimmed
            else:
                return None
    return text


def get_topic_by_text(text: str) -> BillTopic:
    topic = BillTopic.objects.filter(name__exact=text).first()
    if topic is None:
        topic = BillTopic(name=text, created_at=datetime.datetime.now(tz=datetime.timezone.utc))
        topic.save()
        return topic
    return topic


def set_topics(bill: Bill, response: str):
    topic_texts = extract_response_topics(bill, response)
    topics = []
    for topic_text in topic_texts:
        topic = get_topic_by_text(topic_text)
        topics.append(topic)
    bill.topics.set(topics)


# Gets the index and whether we're dealing with a single topic in the response.
def get_top_10_index(bill: Bill, response: str) -> (int, bool, bool):
    index = response.find("Top 10 most important topics:")
    if index > -1:
        return index, False, False

    index = response.find("Top 10")
    if index > -1:
        return index, False, False

    if response[:2] == "1.":
        return 0, False, False

    list_start_index = response.find('1.')
    if list_start_index > -1:
        return list_start_index, False, False

    list_start_index = response.find('1:')
    if list_start_index > -1:
        return list_start_index, False, False

    list_start_index = response.find('-')
    if list_start_index > -1:
        return list_start_index, False, False

    index = response.find("Topic:")
    if index > -1:
        return index, True, False

    # In this case, probably just a raw list of topics by line.
    if len(bill.bill_sections.all()) > 1:
        return 0, False, True

    return -1, False, True


def trim_start_end_parenthesis(text: str) -> str:
    if text and text.startswith('(') and text.endswith(')'):
        text = text[1:len(text) - 1]
    return text


def set_focus_and_summary(bill: Bill, response: str):
    # if ValueError is thrown, we'll get an exception and openai response stored in the Bill and we can investigate later.
    # Example: Ranking on staying on topic: 10/10.
    # Very dirty and naughty but fast.
    topic_ranking_end_token = "/10"
    topic_ranking_index = response.find(topic_ranking_end_token)
    if topic_ranking_index == -1:
        print(f"Warning, no ranking or summary found for {bill.gov_id}.")
        return
    # now walk backward from there until we find something that's not a number or a decimal.
    topic_ranking_raw = ""
    index = topic_ranking_index - 1
    while True:
        char = response[index]
        if char.isnumeric() or char == ".":
            topic_ranking_raw = char + topic_ranking_raw
            index -= 1
        else:
            break
    # cast to int and round incase ranking like 0.5
    topic_ranking = int(float(topic_ranking_raw.strip()))
    bill.on_topic_ranking = topic_ranking
    [top_10_index, _is_single_topic, _] = get_top_10_index(bill, response)

    if top_10_index == -1:
        print(f"Warning, no ranking or summary found for {bill.gov_id}.")
        return

    summary_token = "Summary:"
    summary_token_index = response.find(summary_token)
    if summary_token_index > -1:
        summary_index = summary_token_index + len(summary_token)
        # We assume everything after topic ranking is the summary.
        bill.text_summary = trim_start_end_parenthesis(response[summary_index:top_10_index].strip())

        if summary_index < topic_ranking_index and len(response[topic_ranking_index:]) > 50:
            bill.on_topic_reasoning = response[topic_ranking_index + (len(topic_ranking_end_token)):].strip()
            if bill.on_topic_reasoning[0] == "." or bill.on_topic_reasoning[1] == "." or bill.on_topic_reasoning[
                2] == ".":
                bill.on_topic_reasoning = bill.on_topic_reasoning[bill.on_topic_reasoning.index(" "):].strip()
        return

    # Text did not contain "Summary:". So, maybe it's in the format of <topic ranking> - <summary>
    dash_index = response[topic_ranking_index + 1:topic_ranking_index + 10].find('-')
    if dash_index > -1:
        bill.text_summary = trim_start_end_parenthesis(response[topic_ranking_index + 1 + dash_index + 1:].strip())
        return

    # Maybe it's in the format of <topic ranking> . <summary>
    dot_index = response[topic_ranking_index + 1:topic_ranking_index + 10].find('.')
    if dot_index > -1:
        bill.text_summary = trim_start_end_parenthesis(response[topic_ranking_index + 1 + dot_index + 1:].strip())
        return

    # Maybe it's in the format of <topics>\n<ranking> <summary>
    beginning_text_after_ranking = response[topic_ranking_index + 1:topic_ranking_index + 5]
    if beginning_text_after_ranking.split(' ')[0].isnumeric() and len(response[topic_ranking_index + 1:]) > 10:
        bill.text_summary = trim_start_end_parenthesis(response[topic_ranking_index + 3:].strip())
        return

    # Maybe it's in the format of <topics>\n\n<ranking><summary>
    potential_summary = response[topic_ranking_index + 1:].strip()
    # It may end up just being a number.
    if not potential_summary.isnumeric():
        bill.text_summary = trim_start_end_parenthesis(potential_summary)
    else:
        # Reset if re-parsing.
        bill.text_summary = None
    # TODO set reasoning


def process_analyzed_bill_sections(bill: Bill):
    final_analyze_response = get_bill_final_analysis_response(bill)
    set_topics(bill, final_analyze_response)
    set_focus_and_summary(bill, final_analyze_response)
    bill.last_analyzed_at = datetime.datetime.now(tz=datetime.timezone.utc)
    bill.last_analyze_error = None


def create_word_sections(max_words: int, bill: Bill):
    sections = []
    pieces = bill.text.split(" ")
    for i in range(0, len(pieces), max_words):
        chunk_start = i
        chunk_end = i + max_words
        section = BillSection(
            text_start=chunk_start,
            text_end=chunk_end,
        )
        section.save()
        sections.append(section)
    bill.bill_sections.set(sections)


def create_word_sections_from_lines(max_words: int, text: str) -> [str]:
    pieces = []
    piece = ""
    for line in text.splitlines():
        if len(piece.split(" ")) + len(line.split(" ")) >= max_words:
            pieces.append(piece)
            piece = ""
        else:
            piece += line + "\n"
    if len(piece) > 0:
        pieces.append(piece)
    return pieces


def get_bill_final_analysis_response(bill: Bill) -> str | None:
    """
    Some bills are missing final_analyze_response. Re-running processing will fix that.
    """
    sections = bill.bill_sections.all()
    if bill.final_analyze_response is None:
        if len(sections) == 1:
            if sections.first().last_analyze_response is not None:
                return sections.first().last_analyze_response
    return bill.final_analyze_response


def is_ready_for_processing(bill: Bill) -> bool:
    if bill.last_analyze_response is None:
        return False
    if get_bill_final_analysis_response(bill) is None:
        return False
    sections = bill.bill_sections.all()
    for section in sections:
        if not section.last_analyze_response:
            return False
    return True


def reduce_topics(bill: Bill) -> str:
    sections_topics_text = ""
    for section in bill.bill_sections.all():
        section_topic_lines = section.last_analyze_response.splitlines()
        for line in section_topic_lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Remove bullets which might confuse AI if we go 1. 2. 3. 1. 2. 3.
            # Looking for pattern like: "1." or "1 ." or "1-" or "1 -"
            # Easier to understand/debug/step through than regex.
            if stripped[0].isnumeric():
                if stripped[1] == "." or stripped[1] == "-":
                    stripped = stripped[2:]
                elif stripped[2] == "." or stripped[2] == "-":
                    stripped = stripped[3:]
            elif stripped[0] == "-":
                stripped = stripped[1:]
            elif stripped.startswith('Topic:'):
                stripped = stripped.replace('Topic:', '')
            sections_topics_text += stripped.strip() + "\n"
    # this may still fail on very large bills, have to do recursive map reduce.
    iterations = 0
    while len(sections_topics_text.split(" ")) > WORDS_MAX and iterations <= 10_000:
        chunks = create_word_sections_from_lines(int(WORDS_MAX / 2), sections_topics_text)
        print(f"Topic list long, reduced to {len(chunks)} chunks for {bill.gov_id} (iteration {iterations}).")
        for index, chunk in enumerate(chunks):
            print(f"Summarizing chunk {index} with {len(chunk.split(' '))} words.")
            prompt = f"List the top 10 most important topics the following text:\n{chunk}"
            completion = openai_with_rate_limit_handling(prompt=prompt, retry=True)
            print(completion)
            response_text = completion.choices[0].message.content
            print(response_text)
            if not (response_text.startswith('I apologize') or response_text.startswith("I'm sorry")):
                chunks[index] = response_text
        iterations += 1
        sections_topics_text = "\n".join(chunks)
        print(f"Reduced topic summary to {len(sections_topics_text.split(' '))} words.")
    return sections_topics_text


def analyze_bill_sections(bill: Bill, reparse_only: bool):
    if not bill.bill_sections or len(bill.bill_sections.all()) == 0:
        print('Setting up bill sections.')
        create_word_sections(WORDS_MAX, bill)

    sections = bill.bill_sections.all()
    if not reparse_only:
        for index, section in enumerate(sections):
            if not section.last_analyze_response:
                print(f"Processing section {index + 1}/{len(sections)} of {bill.gov_id}")
                # If we can, this is done all in one prompt to try to reduce # of tokens.
                prompt = f"Summarize and list the top 10 most important topics the following text, and rank it from 0 to 10 on staying on topic:\n{section.get_text(bill.text)}" \
                    if len(
                    sections) == 1 else f"List the top 10 most important topics the following text:\n{section.text}"
                completion = openai_with_rate_limit_handling(prompt=prompt, retry=True)
                print(completion)
                response_text = completion.choices[0].message.content
                section.last_analyze_response = response_text
                section.last_analyze_model = model
                section.last_analyze_error = None
                section.save(update_fields=['last_analyze_response', 'last_analyze_model', 'last_analyze_error'])
                bill.last_analyze_response = response_text
                bill.last_analyze_model = model
                bill.save(update_fields=['last_analyze_response', 'last_analyze_model'])
            else:
                print(f"Section {index + 1}/{len(sections)} already processed, skipping.")
            if len(sections) == 1:
                bill.final_analyze_response = section.last_analyze_response
                bill.save(update_fields=['final_analyze_response'])
            print(f"Processed section {index + 1}/{len(sections)} of {bill.gov_id}")
        if len(sections) > 1:
            print(f"Processed {len(sections)} sections of {bill.gov_id}. Summarizing.")
            topics_list = reduce_topics(bill)
            bill.final_analyze_response = topics_list
            bill.last_analyze_response = topics_list
            bill.last_analyze_model = model
            bill.last_analyze_error = None
            bill.save(update_fields=['final_analyze_response', 'last_analyze_response', 'last_analyze_model',
                                     'last_analyze_error'])
    else:
        print(f"Processed {len(sections)} sections. Done.")
    if is_ready_for_processing(bill):
        process_analyzed_bill_sections(bill)
        # Now just save everything.
        bill.save(update_fields=bill_fields_to_update)
    else:
        print(f"Bill {bill.gov_id} not yet ready for processing!")


def get_traceback(e):
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)


def run(arg_reparse_only: str, year: str | None = None):
    reparse_only = arg_reparse_only == 'True'

    if not reparse_only:
        openai.organization = os.getenv("OPENAI_API_ORG")
        openai.api_key = os.getenv("OPENAI_API_KEY")

    print('Finding bills to analyze...')
    bills = Bill.objects.filter(is_latest_revision=True) \
        .only("id", "gov_id", "text", "bill_sections") if reparse_only else Bill.objects.filter(
        is_latest_revision=True, last_analyzed_at__isnull=True).only("id", "gov_id", "text", "bill_sections")

    bills = bills.order_by('-date')
    # bills = bills.filter(gov_id="112hjres54ih")
    # bills = bills.filter(gov_id="105hr750rfs")

    if year is not None:
        print(f"Will analyze bills for the year {year}.")
        bills = bills.filter(date__year=int(year))
    else:
        print(f"Will analyze bills for all years.")

    print(f"Will analyze {bills.count()} bills.")
    for bill in bills:
        print(F"Analyzing {bill.gov_id}")
        # print(f"Analyzing {bill.text}")
        try:
            analyze_bill_sections(bill, reparse_only)
        except Exception as e:
            print(f"Failed for {bill.gov_id}", e, get_traceback(e))
            bill.last_analyze_error = get_traceback(e)
            try:
                bill.save(update_fields=bill_fields_to_update)
            except Exception as e:
                print(f"Failed to save last_analyze_error for {bill.gov_id}", e, get_traceback(e))
