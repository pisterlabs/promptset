"""For inspecting webpages."""

from contextlib import suppress
from pathlib import Path
from dataclasses import dataclass

from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatAnthropic

from hivemind.config import TEST_DIR
from hivemind.toolkit.models import query_model
from hivemind.toolkit.text_formatting import dedent_and_strip, extract_blocks


class ZoomError(Exception):
    """Error when zooming in or out of a page."""


class MissingSectionError(Exception):
    """Error when a section is missing from the current zoom view."""


@dataclass
class WebpageInspector:
    """Agent to inspect the contents of a webpage with varying degrees of detail."""

    html: str
    """HTML of the webpage."""

    _breadcrumbs: list[str] | None = None
    """Breadcrumbs for where the agent is focused on the page."""

    _section_outlines: dict[tuple[str, ...], str] | None = None
    """All currently generated section outlines."""

    @property
    def role_instructions(self) -> str:
        """Main role instructions for the agent role."""
        return dedent_and_strip(
            """
            # PURPOSE
            Behave like a self-assembling program whose purpose is to represent the contents of the page (given to you as an HTML) in a concise, distilled format to a blind user at various levels of detail specified by the user.
            """
        )

    @property
    def html_context(self) -> str:
        """Instructions for the HTML."""
        return dedent_and_strip(
            """
            # HTML
            Here is the HTML of the page you are representing:

            ```html
            {html}
            ```
            """
        ).format(html=self.html)

    @property
    def base_instructions(self) -> str:
        """Base instructions for the agent."""
        return dedent_and_strip(
            """
            # INSTRUCTIONS
            You are a read-only program, and do NOT have the ability to interact with the page.

            Your goal is to run the remainder of the chat as a fully functioning program that is ready for user input once this prompt is received.
            """
        )

    @property
    def model(self) -> ChatAnthropic:
        """The model for the agent."""
        return ChatAnthropic(
            temperature=0, model="claude-2", max_tokens_to_sample=50000, verbose=False
        )  # hardcoded for now; agent must process large amounts of html text

    def extract_page_outline(self) -> str:
        """Extract the outline of the page."""
        instructions = dedent_and_strip(
            # "navigational structure" prompts claude to give a hierarchical outline of the page; "content outline" prompts it to focus on the textual contents
            """
            Please give me a summary outline of the navigational structure of the page. Include any navigation, main, footer, and any other sections that are relevant for skimming the overall page contents.
            Enclose the outline in a markdown block:
            ```markdown
            {{outline}}
            ```
            """
            # """
            # Please give me a high-level, hierarchical outline of the high-level sections of the page. Include any navigation, main, footer, and any other sections that are relevant for skimming the overall page contents.
            # Enclose the outline in a markdown code block:
            # ```markdown
            # {{outline}}
            # """
            # """
            # Please give me a top-level, hierarchical outline of the contents on the page; the outline should be a tree structure that includes:
            # - the TITLE of the page.
            # - all TOP-LEVEL sections on the page—e.g. navigation, main, footer, etc.
            # - selected KEY interactive elements on the page, and their element type (e.g. <a>, <input>, <button>, etc.).
            # - selected KEY text elements on the page.
            # Enclose the outline in a markdown code block:
            # ```text
            # {{outline}}
            # ```
            # """
        )
        messages = [
            SystemMessage(content=self.role_instructions),
            SystemMessage(content=self.html_context),
            SystemMessage(content=self.base_instructions),
            HumanMessage(content=instructions),
        ]
        result = query_model(self.model, messages, printout=False).strip()
        if result := extract_blocks(result, block_type="markdown"):
            return result[0].strip()
        raise ValueError("Could not extract page outline.")

    def update_page(self, html: str) -> None:
        """Update the HTML of the webpage."""
        self.html = html

    @property
    def title(self) -> str:
        """Return the title of the page."""
        return self.html.split("<title>")[1].split("</title>")[0]

    @property
    def breadcrumbs(self) -> tuple[str, ...]:
        """Breadcrumbs for where the agent is focused on the page."""
        return () if self._breadcrumbs is None else tuple(self._breadcrumbs)

    @property
    def page_outline(self) -> str:
        """Outline of the page."""
        return self.extract_page_outline()

    @property
    def full_page_context(self) -> str:
        """Context for the current view of the page."""
        return dedent_and_strip(
            """
            # CURRENT VIEW
            You are currently viewing the full page. Here is the high-level outline of the page:
            ```
            {page_outline}
            ```
            """
        ).format(page_outline=self.page_outline)

    @property
    def current_section_name(self) -> str:
        """Return the name of the current section.
        Possibly not a unique identifier, so only use for display purposes."""
        return self.breadcrumbs[-1] if self.breadcrumbs else "Root page"

    @property
    def section_outlines(self) -> dict[tuple[str, ...], str]:
        """All currently generated section outlines."""
        if self._section_outlines is None:
            return {
                (): self.page_outline,
            }
        return self._section_outlines

    @property
    def section_outline(self) -> str:
        """Outline of the current section."""
        return self.section_outlines[self.breadcrumbs]

    @property
    def breadcrumb_display(self) -> str:
        """Display of the breadcrumb trail to the current section."""
        return " -> ".join(("Root page", *self.breadcrumbs))

    @property
    def section_context(self) -> str:
        """Context for the current section of the page."""
        subsection_context = dedent_and_strip(
            """
            # CURRENT VIEW
            You are currently viewing the `{section}` section of the page. Here is the high-level outline of the `{section}` section:
            ```
            {section_outline}
            ```

            Here is the section breadcrumb trail, from the root of the page to the current section:
            {breadcrumbs}
            """
        )
        root_page_context = dedent_and_strip(
            """
            # CURRENT VIEW
            You are currently viewing the full page (the "root" level). Here is the high-level outline of the page:
            ```
            {section_outline}
            ```
            """
        )
        return (
            subsection_context.format(
                section=self.current_section_name,
                section_outline=self.section_outline,
                breadcrumbs=self.breadcrumb_display,
            )
            if self.breadcrumbs
            else root_page_context.format(section_outline=self.section_outline)
        )

    @property
    def current_view_context(self) -> str:
        """Context for the current view of the page."""
        return self.section_context if self.breadcrumbs else self.full_page_context

    def subsection_exists(self, subsection: str) -> bool:
        """Validate that a section exists in the current section outline."""
        return subsection in self.section_outline

    def extract_section_outline(self, subsection: str) -> str:
        """Extract the outline of a section of the page."""
        if not self.subsection_exists(subsection):
            raise MissingSectionError(
                f"Section {subsection} does not exist on the current zoom view: {self.breadcrumb_display}"
            )
        instructions = dedent_and_strip(
            """
            Please give me a high-level, hierarchical outline of the contents of the `{subsection}` SUBSECTION of the section you are viewing:
            - If there are subsections nested WITHIN the `{subsection}` subsection, include the next-level-down subsections.
              - If there are no subsections nested within but there is text content, give a concise outline (think "Mordin Solus" style) of the text content in the `{subsection}` subsection. Do not add information.
            - If there are important INTERACTIVE elements within the `{subsection}` subsection, include them, and their element type (e.g. <a>, <input>, <button>, etc.).

            Enclose the subsection outline in a markdown code block:
            ```markdown
            {{outline}}
            ```
            """
        )
        messages = [
            SystemMessage(content=self.role_instructions),
            SystemMessage(content=self.html_context),
            SystemMessage(content=self.current_view_context),
            SystemMessage(content=self.base_instructions),
            HumanMessage(content=instructions.format(subsection=subsection)),
        ]
        result = query_model(self.model, messages, printout=False).strip()
        if result := extract_blocks(result, block_type="markdown"):
            return result[0].strip()
        raise ValueError("Could not extract section outline.")

    def zoom_in(self, section: str) -> None:
        """Zoom in on a section of the page."""
        try:
            new_section_outline = self.extract_section_outline(section)
        except MissingSectionError as error:
            raise ZoomError(
                f"Section {section} does not exist on the current zoom view: `{self.breadcrumb_display}`"
            ) from error
        self._breadcrumbs = [*self.breadcrumbs, section]
        self._section_outlines = {
            **self.section_outlines,
            self.breadcrumbs: new_section_outline,
        }

    def zoom_out(self) -> None:
        """Zoom out of the current section of the page."""
        if not self._breadcrumbs:
            raise ZoomError(
                "Already at highest level of zoom (root) for the page. Cannot zoom out any further."
            )
        self._breadcrumbs = self._breadcrumbs[:-1]


def test_zoom_out() -> None:
    """Test webpage inspector ability to zoom out of a section of the page."""
    page = Path(TEST_DIR / "cleaned_page.html").read_text(encoding="utf-8")
    oracle = WebpageInspector(html=page)
    oracle.zoom_in("Readme")
    print(oracle.section_outline)  # expect hierarchical outline of Readme section
    oracle.zoom_out()
    print(oracle.section_outline)  # expect hierarchical outline of page
    with suppress(ZoomError):
        oracle.zoom_out()
        print(oracle.section_outline)  # expect ZoomError
        assert False, "Expected ZoomError"


def test_zoom_in() -> None:
    """Test webpage inspector ability to zoom in on a section of the page."""
    page = Path(TEST_DIR / "cleaned_page.html").read_text(encoding="utf-8")
    oracle = WebpageInspector(html=page)
    oracle.zoom_in("Readme")
    print(oracle.section_outline)  # expect hierarchical outline of Readme section
    oracle.zoom_in("Installation")
    print(oracle.section_outline)  # expect concise distillation of Installation section


def test_section_outline() -> None:
    """Test webpage inspector ability to generate section outline."""
    page = Path(TEST_DIR / "cleaned_page.html").read_text(encoding="utf-8")
    oracle = WebpageInspector(html=page)
    print(
        oracle.extract_section_outline("Readme")
    )  # expect hierarchical outline of Readme section


def test_page_outline() -> None:
    """Test webpage inspector ability to generate page outline."""
    page = Path(TEST_DIR / "cleaned_page.html").read_text(encoding="utf-8")
    oracle = WebpageInspector(html=page)
    print(oracle.extract_page_outline())  # expect hierarchical outline of page
