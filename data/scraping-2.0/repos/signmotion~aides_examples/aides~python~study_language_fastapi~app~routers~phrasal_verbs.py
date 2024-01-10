from fastapi import APIRouter
import g4f
import traceback
import re
import openai

from ..internal.config import *
from .context import context


router = APIRouter()


@router.get("phrasal-verbs-about")
def phrasal_verbs_about():
    return "Extracts phrasal verbs from [text] and translates these verbs."


@router.get("phrasal-verbs-tags")
def phrasal_verbs_tags():
    return ["extract", "phrasal verbs", "translate"]


@router.get("/phrasal-verbs")
def phrasal_verbs():
    count_request_errors = 0
    ignored_providers = [
        # confidently
        # "ChatAnywhere",
        # "ChatBase",
        # "ChatgptX",
        # "GptGo",
        # maybe
        # "GptForLove",
        # "Chatgpt4Online",
    ]

    responseResult = None
    improvedResult = None
    mappedResult = None
    error = None
    while True:
        try:
            # automatic selection of provider
            response = queryChatGpt() if use_chat_gpt else queryG4F(ignored_providers)

            responseResult = response

            if improve_answer:
                improvedResult = improve(responseResult)

            if map_answer:
                mappedResult = map(
                    improvedResult if improvedResult else improve(responseResult)
                )

            error = None
            break

        except Exception as ex:
            # provider_error = get_provider_from_error(ex)
            # if provider_error:
            #     ignored_providers.append(provider_error)
            #     print(f"Added `{provider_error}` to ignored list.")

            count_request_errors += 1
            print(f"ATTEMPT {count_request_errors} {ex} :: {traceback.format_exc()}")
            error = ex
            if count_request_errors >= max_count_request_errors:
                break

    return construct_answer(
        mappedResult=mappedResult,
        improvedResult=improvedResult,
        rawResult=responseResult,
        error=error,
    )


def queryChatGpt():
    if fake_response:
        return phrasal_verbs_demo_text()["result"]

    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt(),
        temperature=0.0,
        max_tokens=300,
    )

    print(response)

    return response.choices[0].text


def queryG4F(ignored_providers: list):
    if fake_response:
        return phrasal_verbs_demo_text()["result"]

    return g4f.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt()}],
        ignored=ignored_providers,
    )


def prompt():
    text = context().text

    return f"""
Write out all phrasal verbs from the text below with a translation into Ukrainian.
Phrasal verbs are a type of verb in English that consist of a verb and a preposition or an adverb, or both.
Wite down only phrasal verbs.
Take into account the context for the translation.
Don't repeat verbs if they have been written down before.

TEXT:

{text}
"""


def get_provider_from_error(ex: Exception):
    splitted = f"{ex}".split(":")
    r = None
    if len(splitted) > 1:
        r = splitted[1].strip()
        if len(r.split(" ")) != 0 or r == "RetryProvider":
            r = None

    return r


# 1. make out (translation: зрозуміти)\n
# 2. read out (translation: прочитати)\n
# 3. come out (translation: вийти)\n
# ...
# 37. think of (translation: думати про)\n
# ...
def improve(text: str):
    r = []
    lines = text.split("\n")
    for line in lines:
        try:
            line = improve_line(line)
        except Exception as ex:
            print(f"{line} :: {ex} :: {traceback.format_exc()}")
            # line = f"{line} :: {ex} :: {traceback.format_exc()}"

        if line:
            r.append(line)

    print("Removing duplicates...")
    unique_r = []
    for line in r:
        if line not in unique_r:
            unique_r.append(line)

    return "\n".join(unique_r)


def improve_line(line: str):
    print(f"\n{line}")

    print("Removing numbering...")
    line = re.sub(r"\d+\.\s*", "", line)

    print("Improving a format for translation...")
    line = re.sub("translation", "", line)
    line = re.sub(":", "", line)

    # make out (: зрозуміти)
    try:
        a, b = line.split("(")
        a = a.strip()
        b = re.sub("\)", "", b).strip()
        line = f"{a} - {b}"
    except Exception as ex:
        None

    # Make out - Розібратися
    try:
        a, b = line.split(" - ")
        a = a.strip()
        line = f"{a} - {b}"
    except Exception as ex:
        None

    print("Removing non-phrasal verbs...")
    sa = a.split(" ")
    if len(sa) < 2:
        line = None

    return line.lower() if line else line


def map(improvedText: str):
    r = {}

    lines = improvedText.split("\n")
    for line in lines:
        a, b = line.split(" - ")
        r[a] = b

    return r


def construct_answer(
    mappedResult: dict,
    improvedResult: dict,
    rawResult: dict,
    error: Exception,
):
    o = {}

    if mappedResult:
        o["mapped_result"] = mappedResult

    if improvedResult:
        o["improved_result"] = improvedResult

    if include_raw_response_in_answer or (not mappedResult and not improvedResult):
        o["raw_result"] = rawResult

    if error:
        print(f"!) {error}")
        o["error"] = {
            "key": f"{error}",
            "traceback": f"{traceback.format_exc()}",
        }

    if include_context_in_answer:
        o["context"] = context()

    return o


@router.get("/phrasal-verbs-demo-text")
def phrasal_verbs_demo_text():
    return {
        # "result": "1. make out (translation: зрозуміти)\n2. read out (translation: прочитати)\n3. come out (translation: вийти)\n4. use up (translation: використовувати)\n5. work out (translation: розібратися)\n6. figure out (translation: з'ясувати)\n7. go into (translation: увійти)\n8. read up on (translation: почитати про)\n9. set in (translation: встановлювати)\n10. make predictions (translation: робити передбачення)\n11. be good (translation: бути хорошим)\n12. be bad (translation: бути поганим)\n13. work well together (translation: добре співпрацювати разом)\n14. spend hours (translation: проводити години)\n15. love (translation: любити)\n16. hate (translation: ненавидіти)\n17. cause to pause (translation: заставляти задуматися)\n18. be useful (translation: бути корисним)\n19. be better than (translation: бути кращим, ніж)\n20. be worth it (translation: бути вартим того)\n21. figure things out (translation: розібратися в чомусь)\n22. have a disadvantage (translation: мати недолік)\n23. see something out (translation: побачити щось до кінця)\n24. know what something should do (translation: знати, що має робити щось)\n25. have a leg up on (translation: мати перевагу над)\n26. make accurate predictions (translation: робити точні передбачення)\n27. hinder from (translation: заважати чому-небудь)\n28. gloss over (translation: пропустити, замовчати)\n29. have a distinct advantage (translation: мати виразну перевагу)\n30. play a lot of (translation: грати багато в)\n31. help (translation: допомагати)\n32. have some familiarity with (translation: мати певну знайомість з)\n33. matter (translation: мати значення)\n34. provide meaningful value (translation: надавати практичну цінність)\n35. bring in (translation: привертати)\n36. get excited about (translation: захоплюватися)\n37. think of (translation: думати про)\n38. demand (translation: вимагати)\n39. learn (translation: вчитися)\n40. accomplish (translation: досягати)\n41. pull away from (translation: відводити увагу від)\n42. ask for (translation: просити)\n43. come up with (translation: придумувати)\n44. work on (translation: працювати над)\n45. create (translation: створювати)\n46. plot out (translation: розробляти схему)\n47. doubt (translation: сумніватися)\n48. happen next (translation: трапитися далі)\n49. find (translation: знаходити)\n50. struggle with (translation: боротися з)\n51. make decisions (translation: приймати рішення)\n52. understand (translation: розуміти)",
        "result": "1. Take into account - Враховувати\n2. Write down - Записати\n3. Make out - Розібратися\n4. Read out - Вголос прочитати\n5. Work out - Розібратися (про щось)\n6. Come out - Вийти (про новий набір карт)\n7. Figure out - Розібратися\n8. Read up - Почитати про щось\n9. Love these things - Обожнювати це\n10. Hate them - Ненавидіти їх\n11. Paused - Зупинитись на мить\n12. Be worth it - Бути вартим\n13. Hinder from - Заважати\n14. Gloss over - Пропустити, не зупинятись\n15. Have a leg up on - Мати перевагу над\n16. Seem good - Здаватися хорошим\n17. Work out well - Вдалий результат\n18. Over-perform expectations - Перевиконувати очікування\n19. Get in the way - Заважати, стояти на шляху\n20. Look like - Виглядати як\n21. Provide meaningful value - Надавати суттєву користь\n22. Bring in - Привертати\n23. Pull away from - Відводити увагу від\n24. Come up with - Придумати\n25. Work on - Працювати над\n26. Plot out - Складати план\n27. Doubt it - Сумніватись в цьому\n28. Make good decisions - Приймати правильні рішення\n29. Studying in the moment - Вчитися в даний момент\n30. Understand right now - Розуміти зараз",
        "context": context(),
    }


@router.get("/phrasal-verbs-demo-json")
def phrasal_verbs_demo_json():
    return {
        "result": {
            "draft (cards)": "вибирати (карти)",
            "come up with": "придумати",
            "pull away": "відволікати",
        },
        "context": context(),
    }
