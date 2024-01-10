from typing import Dict, Any, Optional
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler


class MarkdownHtmlCallback(BaseCallbackHandler):
    """
    Callback ton convert a text from Markdown to html
    """

    def __init__(self, input_field: str = "answer", output_field: str = "answer_html"):
        """Constructor

        Args:
            input_field (str): the outputs key to take the Markdown text from
            output_field (str): the outputs key to put the html value in
        """
        super().__init__()
        self._input_field = input_field
        self._output_field = output_field

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """This callback method in automatically called when a chain end

        Args:
            outputs: dictionary, where to take Markdown from and put HTML in
            run_id: unique identifier for the chain run
            parent_run_id: unique identifier of a parent chain run
            **kwargs:


        """

        # we need to ensure we are on the root chain end call
        if parent_run_id is None and self._input_field in outputs:  # last chain_end
            import markdown

            # extract the Markdown value
            markdown_txt = outputs.get(self._input_field)

            # convert it to html
            html_output = markdown.markdown(markdown_txt)

            # put it back to the dictionary
            outputs[self._output_field] = html_output
