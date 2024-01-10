from typing import Tuple

from ovos_plugin_manager.templates.transformers import DialogTransformer

from ovos_solver_openai_persona import OpenAIChatCompletionsSolver


class OpenAIDialogTransformer(DialogTransformer):
    def __init__(self, name="ovos-dialog-transformer-openai-plugin", priority=10, config=None):
        super().__init__(name, priority, config)
        self.solver = OpenAIChatCompletionsSolver({
            "key": self.config.get("key"),
            'api_url': self.config.get('api_url', 'https://api.openai.com/v1'),
            "enable_memory": False,
            "initial_prompt": "your task is to rewrite text as if it was spoken by a different character"
        })

    def transform(self, dialog: str, context: dict = None) -> Tuple[str, dict]:
        """
        Optionally transform passed dialog and/or return additional context
        :param dialog: str utterance to mutate before TTS
        :returns: str mutated dialog
        """
        prompt = context.get("prompt") or self.config.get("rewrite_prompt")
        if not prompt:
            return dialog, context
        return self.solver.get_spoken_answer(f"{prompt} : {dialog}"), context
