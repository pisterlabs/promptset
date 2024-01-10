import logging
from csv import DictReader
from io import StringIO
from typing import Sequence, List

from anki.collection import OpChanges, Collection, OpChangesWithCount
from anki.notes import NoteId, Note
from aqt import mw
from aqt.operations import CollectionOp, ResultWithChanges
from aqt.qt import QAction, qconnect
from aqt.utils import showInfo
from openai.types.chat import ChatCompletion

from fields import english_field, part_of_speech_field, description_field
from tags import unit_tag
from . import openai_client

log: logging.Logger = logging.getLogger(__name__)


class WantCancelException(Exception):
    pass


def _background_operation(col: Collection) -> ResultWithChanges:
    changes: OpChangesWithCount = OpChangesWithCount()
    query: str = f'{description_field}: -tag:{unit_tag}'
    log.info(f"Query: '{query}'")
    note_ids: Sequence[NoteId] = col.find_notes(query)
    log.info(f"Found notes: {len(note_ids)}")
    notes: List[Note] = [col.get_note(note_id) for note_id in note_ids]
    slice_size: int = 25
    log.info(f"Slice size: {slice_size}")
    notes_len: int = len(notes)
    _show_progress(changes, notes_len)
    for i in range(0, notes_len, slice_size):
        log.info(f"Processed notes: {i} from {notes_len}")
        sublist: List[Note] = notes[i:i + slice_size]
        try:
            _update_notes(col, sublist, changes)
        except WantCancelException:
            log.info("Cancelling")
            break
        _show_progress(changes, notes_len)
    return changes


def _show_progress(changes: OpChangesWithCount, notes_len: int):
    mw.taskman.run_on_main(
        lambda: mw.progress.update(
            label=f"Updated: {changes.count} from {notes_len}",
            value=changes.count,
            max=notes_len
        )
    )


def _update_notes(col: Collection, notes: List[Note], changes: OpChangesWithCount):
    log.info(f"Notes to update in slice: {len(notes)}")
    nid_column: str = 'ID'
    english_column: str = 'English Word'
    pos_column: str = 'Part Of Speech'
    definition_column: str = 'Definition'
    lines: List[str] = []
    for note in notes:
        word: str = note[english_field]
        if not note[part_of_speech_field]:
            raise RuntimeError(f"Part of speech is missing: nid={note.id}")
        pos: str = note[part_of_speech_field]
        line: str = f'"{note.id}","{word}","{pos}"'
        lines.append(line)
    lines_str: str = '\n'.join(lines)
    prompt: str = (
        f'I will provide you a CSV snippet. '
        f'You need to fill column "{definition_column}" in the snippet with simple and short definition of the word. '
        f'Your response must contain strictly only raw CSV content (it is very important). '
        f'The CSV snippet is:\n'
        f'```\n'
        f'"{nid_column}","{english_column}","{pos_column}","{definition_column}"\n'
        f'{lines_str}\n'
        f'```'
    )
    log.debug(f"Prompt:\n{prompt}")
    chat_completion: ChatCompletion = openai_client.get_completion(prompt)
    message: str = chat_completion.choices[0].message.content
    log.debug(f"Message:\n{message}")
    message = message.replace('```\n', '').replace('\n```', '')
    log.debug(f"Message without MarkDown:\n{message}")
    with StringIO(message) as csv_file:
        reader: DictReader = DictReader(csv_file, doublequote=True)
        for row in reader:
            if mw.progress.want_cancel():
                raise WantCancelException()
            log.debug(f"Row: {row}")
            nid_int: int = int(row[nid_column])
            note: Note = col.get_note(NoteId(nid_int))
            if note[english_field] != row[english_column]:
                raise RuntimeError(f"Wrong English word: note={note[english_field]}, row={row[english_column]}")
            description_old: str = note[description_field]
            if description_old != '':
                raise RuntimeError(f'Field {description_field} is not empty: nid={nid_int}, value="{description_old}"')
            description_new: str = row[definition_column]
            if description_new == '':
                raise RuntimeError(f"Empty definition: nid={nid_int}")
            note[description_field] = description_new
            log.info(f"Updating note: nid={note.id}, english='{note[english_field]}', "
                     f"pos='{note[part_of_speech_field]}', "
                     f"definition='{note[description_field]}'")
            col.update_note(note)
            changes.count += 1


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
    action = QAction('Fill "Description-my" field', mw)
    qconnect(action.triggered, _ui_action)
    return action
