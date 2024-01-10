# DO NOT USE: works slow, GPT responses are unreliable
import logging
import re
from typing import Sequence, List

from anki.collection import OpChanges, Collection, OpChangesWithCount
from anki.notes import NoteId, Note
from aqt import mw
from aqt.operations import CollectionOp, ResultWithChanges
from aqt.qt import QAction, qconnect
from aqt.utils import showInfo
from openai.types.chat import ChatCompletion

from fields import english_field, examples1_generated_field, \
    examples2_generated_field, examples3_generated_field
from . import openai_client

log: logging.Logger = logging.getLogger(__name__)

fields: List[str] = [examples1_generated_field, examples2_generated_field, examples3_generated_field]


def _background_operation(col: Collection) -> ResultWithChanges:
    changes: OpChangesWithCount = OpChangesWithCount()
    field_clauses: List[str] = [f'{field}:_*' for field in fields]
    query: str = f'{" or ".join(field_clauses)}'
    log.info(f"Query: '{query}'")
    all_note_ids: Sequence[NoteId] = col.find_notes(query)
    log.info(f"Found all notes: {len(all_note_ids)}")
    all_notes: List[Note] = [col.get_note(note_id) for note_id in all_note_ids]
    notes: List[Note] = [note for note in all_notes if _note_contains_chinese(note)]
    notes_len: int = len(notes)
    log.info(f"Found chinese notes: {notes_len}")
    _show_progress(changes, notes_len)
    for note in notes:
        for field in fields:
            if mw.progress.want_cancel():
                log.info("Cancelling")
                return changes
            _update_note(col, note, field)
        changes.count += 1
        _show_progress(changes, notes_len)
    return changes


def _contains_chinese(text: str) -> bool:
    pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff\u3000-\u303f\uff00-\uffef]'
    if re.search(pattern, text):
        return True
    return False


def _note_contains_chinese(note: Note) -> bool:
    for field in fields:
        if _contains_chinese(note[field]):
            return True
    return False


def _show_progress(changes: OpChangesWithCount, notes_len: int):
    mw.taskman.run_on_main(
        lambda: mw.progress.update(
            label=f"Processed: {changes.count} from {notes_len}",
            value=changes.count,
            max=notes_len
        )
    )


def _update_note(col: Collection, note: Note, field: str):
    log.info(f'Updating text: nid={note.id}, field={field}')
    old_value: str = note[field]
    if not _contains_chinese(old_value):
        log.info(f"Skip field without Chinese characters: nid={note.id}, field={field}")
        return
    prompt: str = (
        f'I will give you an HTML snippet.\n'
        f'\n'
        f'You need to perform these operations on the snippet:\n'
        f'1. Remove any information related to CSS.\n'
        f'2. Remove Chinese, Japanese or Korean characters.\n'
        f'3. Remove tags that became empty.\n'
        f'4. Convert it to a flat HTML list without `div` tags.\n'
        f'5. If any list element starts with number duplicating `li` tag, remove the number.\n'
        f'6. Surround word `{note[english_field]}` with `em` tag if it is not surrounded already.\n'
        f'\n'
        f'Your response must contain strictly only resulting snippet without any comments (it is very important).\n'
        f'The text is:\n'
        f'```\n'
        f'{old_value}\n'
        f'```'
    )
    log.debug(f"Prompt:\n{prompt}")
    chat_completion: ChatCompletion = openai_client.get_completion(prompt, model='gpt-3.5-turbo-1106')
    message: str = chat_completion.choices[0].message.content
    log.debug(f"Message:\n{message}")
    message = message.replace('```\n', '').replace('\n```', '')
    log.debug(f"Message without MarkDown:\n{message}")
    if _contains_chinese(message):
        raise RuntimeError(f'Cleaned text still contains Chinese characters: nid={note.id}, text="{message}"')
    note[field] = message
    log.info(f"Updating note: nid={note.id}, english='{note[english_field]}', field='{field}',\n"
             f"old='{old_value}',\n"
             f"new='{note[field]}'")
    # col.update_note(note)


def on_success(changes: ResultWithChanges) -> None:
    showInfo(f"Finished: updated {changes.count} notes")


def on_failure(e: Exception) -> None:
    showInfo(f"Failed: {e}")


def _ui_action():
    op: CollectionOp[OpChanges] = CollectionOp(parent=mw, op=lambda col: _background_operation(col))
    op.success(on_success)
    op.failure(on_failure)
    op.run_in_background()


def menu_action() -> QAction:
    action = QAction('Remove Chinese characters from examples', mw)
    qconnect(action.triggered, _ui_action)
    return action
