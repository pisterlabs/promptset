from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING, Any, Callable

from anki.collection import Collection, OpChanges, OpChangesWithCount
from aqt.operations import CollectionOp, QueryOp
from aqt.utils import showWarning, tooltip

from . import consts
from .gpt import OpenAI

if TYPE_CHECKING:
    from anki.notes import Note
    from aqt.main import AnkiQt
    from aqt.qt import QWidget


class BulkCompleter:
    "This class is used in the editor or browser to process given notes according to the config and update the notes."

    def __init__(self, mw: AnkiQt, parent: QWidget, config: dict) -> None:
        self.mw = mw
        self.parent = parent
        self.config = config
        self.completer = OpenAI(config["service_options"]["openai"])

    def _on_failure(self, exc: Exception) -> None:
        showWarning(str(exc), self.parent, title=consts.ADDON_NAME)

    def fill_note(
        self,
        options: dict,
        note: Note,
        on_success: Callable,
        fill_prompt: bool = False,
    ) -> None:
        "This wraps _fill_note() to show a progress dialog and show error messages. It's used in the editor."

        def op(col: Collection) -> None:
            self._fill_note(options, note, fill_prompt)

        def success(_: Any) -> None:
            return on_success()

        # FIXME: maybe only call with_progress on 2.1.50+ because it was broken in older versions
        QueryOp(parent=self.parent, op=op, success=success).failure(
            self._on_failure
        ).with_progress(label="Processing GPT prompts...").run_in_background()

    def _fill_note(self, options: dict, note: Note, fill_prompt: bool = False) -> bool:
        """This calls the OpenAI API and parses the result according to `options`, which corresponds to the settings of some field/preset button."""

        result_field_lists: list[list[str]] = []
        # Here we ignore parsing fields that do not exist in the given note
        for field_list in options["result_fields"]:
            matched = []
            for field in field_list:
                if field in note:
                    matched.append(field)
            result_field_lists.append(matched)

        # If `fill_prompt` is True, we only paste the prompt to the configured prompt field and return
        if fill_prompt:
            for field in options["prompt_fields"]:
                if field in note:
                    note[field] = options["prompt"]
                    return True
            return False
        prompt = "\n".join(
            note[field]
            for field in options["prompt_fields"]
            if field in note and note[field].strip()
        )

        if not any(l for l in result_field_lists):
            return False
        # Here we send the prompt to the API and get the result
        # TODO: test rate limits
        service_options = options.get("service_options", {}).get("openai", {})
        if self.config["use_chat_api"]:
            result = self.completer.chat_complete(prompt, **service_options)
        else:
            result = self.completer.complete(prompt, **service_options)
        if not result:
            return False
        # Remove all occurences of specified regex from result
        if options["remove_pattern"]:
            result = re.sub(options["remove_pattern"], "", result)
        # Here we split the result according to the configured separator
        sections = (
            result.split(options["result_separator"])
            if options["result_separator"]
            else [result]
        )
        for j, result_section in enumerate(sections):
            result_section = result_section.strip()
            result_fields = (
                result_field_lists[j]
                if j < len(result_field_lists)
                else result_field_lists[-1]
            )
            # We write each section to a set of fields, optionally keeping old contents
            for field in result_fields:
                contents = (
                    note[field] if not options.get("override_contents", True) else ""
                )
                if contents:
                    contents += "<br>"
                contents += result_section
                note[field] = contents
        return True

    def process_notes(
        self, options: dict, notes: list[Note], fill_prompt: bool = False
    ) -> None:
        """This takes a list of notes (e.g. selected browser notes) and process them according to given options.
        A progress dialog is shown for the duration of the process.
        """

        want_cancel = False

        def update_progress(i: int) -> None:
            nonlocal want_cancel
            want_cancel = self.mw.progress.want_cancel()
            self.mw.progress.update(
                ("Updating" if fill_prompt else "Processing")
                + f" prompt {i} of {len(notes)}...",
                value=i,
                max=len(notes),
            )

        def op(col: Collection) -> OpChanges:
            self.mw.taskman.run_on_main(
                lambda: self.mw.progress.set_title(consts.ADDON_NAME)
            )
            updated_notes = []
            for i, note in enumerate(notes):
                self.mw.taskman.run_on_main(functools.partial(update_progress, i=i))
                if want_cancel:
                    break
                if self._fill_note(options, note, fill_prompt):
                    updated_notes.append(note)
            return OpChangesWithCount(
                count=len(updated_notes), changes=col.update_notes(updated_notes)
            )

        def success(changes: OpChangesWithCount) -> None:
            msg = (
                "Updated" if fill_prompt else "Processed"
            ) + f" prompts in {changes.count} notes"
            tooltip(msg, parent=self.parent)

        CollectionOp(parent=self.parent, op=op).success(success).failure(
            self._on_failure
        ).run_in_background()
