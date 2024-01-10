import openai
from pydantic import NonNegativeFloat
import re
import traceback
from typing import Any, Dict

from ..config import fake_response, improve_answer, open_api_key, map_answer
from ..context import Context
from ..packages.aide_server.src.aide_server.helpers import (
    construct_and_publish,
    PublishProgressFn,
    PublishResultFn,
)
from ..packages.aide_server.src.aide_server.log import logger
from ..packages.aide_server.src.aide_server.task_progress_result import Task


async def phrasal_verbs(
    task: Task,
    publish_progress: PublishProgressFn,
    publish_result: PublishResultFn,
):
    context = Context.model_validate(task.context)

    return await construct_and_publish(
        __name__,
        task=task,
        construct_raw_result=_construct_raw_result,
        construct_improved_result=_construct_improved_result
        if improve_answer
        else None,
        construct_mapped_result=_construct_mapped_result if map_answer else None,
        publish_progress=publish_progress,
        publish_result=publish_result,
        fake_raw_result=_phrasal_verbs_demo_text(context)["result"]
        if fake_response
        else None,
    )


async def _construct_raw_result(
    task: Task,
    publish_progress: PublishProgressFn,
    publish_result: PublishResultFn,
    start_progress: NonNegativeFloat,
    stop_progress: NonNegativeFloat,
):
    openai.api_key = open_api_key
    context = Context.model_validate(task.context)
    response = openai.Completion.create(  # type: ignore
        engine="text-davinci-003",
        prompt=_prompt(context.text),
        temperature=0.0,
        max_tokens=300,
    )

    logger.info(response)

    return response.choices[0].text


async def _construct_improved_result(
    task: Task,
    raw_result: Any,
    publish_progress: PublishProgressFn,
    publish_result: PublishResultFn,
    start_progress: NonNegativeFloat,
    stop_progress: NonNegativeFloat,
):
    return _improve(raw_result)  # type: ignore[override]


async def _construct_mapped_result(
    task: Task,
    raw_result: Any,
    improved_result: Any,
    publish_progress: PublishProgressFn,
    publish_result: PublishResultFn,
    start_progress: NonNegativeFloat,
    stop_progress: NonNegativeFloat,
):
    return _map(improved_result)


def _prompt(text: str):
    return f"""
Write out all phrasal verbs from the text below with a translation into Ukrainian.
Phrasal verbs are a type of verb in English that consist of a verb and a preposition or an adverb, or both.
Wite down only phrasal verbs.
Take into account the context for the translation.
Don't repeat verbs if they have been written down before.

TEXT:

{text}
"""


# 1. make out (translation: зрозуміти)\n
# 2. read out (translation: прочитати)\n
# 3. come out (translation: вийти)\n
# ...
# 37. think of (translation: думати про)\n
# ...
def _improve(text: str) -> str:
    r = []
    lines = text.split("\n")
    for line in lines:
        try:
            line = _improve_line(line)
        except Exception as ex:
            logger.info(f"{line} :: {ex} :: {traceback.format_exc()}")
            # line = f"{line} :: {ex} :: {traceback.format_exc()}"

        if line:
            r.append(line)

    logger.info("Removing duplicates...")
    unique_r = []
    for line in r:
        if line not in unique_r:
            unique_r.append(line)

    return "\n".join(unique_r)


def _improve_line(line: str):
    logger.info(f"\n{line}")

    logger.info("Removing numbering...")
    line = re.sub(r"\d+\.\s*", "", line)

    logger.info("Improving a format for translation...")
    line = re.sub("translation", "", line)
    line = re.sub(":", "", line)

    a = ""
    b = ""

    # make out (: зрозуміти)
    try:
        a, b = line.split("(")
        a = a.strip()
        b = re.sub(")", "", b).strip()
        line = f"{a} - {b}"
    except Exception as ex:
        pass

    # Make out - Розібратися
    try:
        a, b = line.split(" - ")
        a = a.strip()
        line = f"{a} - {b}"
    except Exception as ex:
        pass

    logger.info("Removing non-phrasal verbs...")
    sa = a.split(" ")
    if len(sa) < 2:
        line = ""

    return line.lower() if line else line


def _map(improvedText: str) -> Dict[str, Any]:
    r = {}

    lines = improvedText.split("\n")
    for line in lines:
        a, b = line.split(" - ")
        r[a] = b

    return r


def _phrasal_verbs_demo_text(context: Context) -> Dict[str, Any]:
    return {
        "result":
        # "1. make out (translation: зрозуміти)\n2. read out (translation: прочитати)\n3. come out (translation: вийти)\n4. use up (translation: використовувати)\n5. work out (translation: розібратися)\n6. figure out (translation: з'ясувати)\n7. go into (translation: увійти)\n8. read up on (translation: почитати про)\n9. set in (translation: встановлювати)\n10. make predictions (translation: робити передбачення)\n11. be good (translation: бути хорошим)\n12. be bad (translation: бути поганим)\n13. work well together (translation: добре співпрацювати разом)\n14. spend hours (translation: проводити години)\n15. love (translation: любити)\n16. hate (translation: ненавидіти)\n17. cause to pause (translation: заставляти задуматися)\n18. be useful (translation: бути корисним)\n19. be better than (translation: бути кращим, ніж)\n20. be worth it (translation: бути вартим того)\n21. figure things out (translation: розібратися в чомусь)\n22. have a disadvantage (translation: мати недолік)\n23. see something out (translation: побачити щось до кінця)\n24. know what something should do (translation: знати, що має робити щось)\n25. have a leg up on (translation: мати перевагу над)\n26. make accurate predictions (translation: робити точні передбачення)\n27. hinder from (translation: заважати чому-небудь)\n28. gloss over (translation: пропустити, замовчати)\n29. have a distinct advantage (translation: мати виразну перевагу)\n30. play a lot of (translation: грати багато в)\n31. help (translation: допомагати)\n32. have some familiarity with (translation: мати певну знайомість з)\n33. matter (translation: мати значення)\n34. provide meaningful value (translation: надавати практичну цінність)\n35. bring in (translation: привертати)\n36. get excited about (translation: захоплюватися)\n37. think of (translation: думати про)\n38. demand (translation: вимагати)\n39. learn (translation: вчитися)\n40. accomplish (translation: досягати)\n41. pull away from (translation: відводити увагу від)\n42. ask for (translation: просити)\n43. come up with (translation: придумувати)\n44. work on (translation: працювати над)\n45. create (translation: створювати)\n46. plot out (translation: розробляти схему)\n47. doubt (translation: сумніватися)\n48. happen next (translation: трапитися далі)\n49. find (translation: знаходити)\n50. struggle with (translation: боротися з)\n51. make decisions (translation: приймати рішення)\n52. understand (translation: розуміти)"
        "1. Take into account - Враховувати\n2. Write down - Записати\n3. Make out - Розібратися\n4. Read out - Вголос прочитати\n5. Work out - Розібратися (про щось)\n6. Come out - Вийти (про новий набір карт)\n7. Figure out - Розібратися\n8. Read up - Почитати про щось\n9. Love these things - Обожнювати це\n10. Hate them - Ненавидіти їх\n11. Paused - Зупинитись на мить\n12. Be worth it - Бути вартим\n13. Hinder from - Заважати\n14. Gloss over - Пропустити, не зупинятись\n15. Have a leg up on - Мати перевагу над\n16. Seem good - Здаватися хорошим\n17. Work out well - Вдалий результат\n18. Over-perform expectations - Перевиконувати очікування\n19. Get in the way - Заважати, стояти на шляху\n20. Look like - Виглядати як\n21. Provide meaningful value - Надавати суттєву користь\n22. Bring in - Привертати\n23. Pull away from - Відводити увагу від\n24. Come up with - Придумати\n25. Work on - Працювати над\n26. Plot out - Складати план\n27. Doubt it - Сумніватись в цьому\n28. Make good decisions - Приймати правильні рішення\n29. Studying in the moment - Вчитися в даний момент\n30. Understand right now - Розуміти зараз",
        "context": context.model_dump_json(),
    }


def _phrasal_verbs_demo_json(context: Context) -> Dict[str, Any]:
    return {
        "result": {
            "draft (cards)": "вибирати (карти)",
            "come up with": "придумати",
            "pull away": "відволікати",
        },
        "context": context.model_dump_json(),
    }
