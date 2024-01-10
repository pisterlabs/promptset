from ..models import Content

content_aw = Content(
    title="ActivityWatch",
    description="The free and open-source automated time-tracker",
    url="https://activitywatch.net/",
    source_url="https://www.reddit.com/r/coolgithubprojects/comments/x2fqvq/activitywatch_an_opensource_automated_timetracker/",
)
content_gpt3 = Content(
    title="OpenAI GPT-3",
    description="GPT-3 is a language model from OpenAI",
    url="https://openai.com/blog/gpt-3/",
    source_url="https://openai.com/blog/gpt-3/",
)
content_st_support = Content(
    title="Many conflicts? Remove synced folder and add it back - solved it for me",
    description="I sync a folder (with a bunch of subfolders across 3 devices, 2 on android and 1 on pc. For a week or so I had an annoying amount of sync conflicts...",
    url="https://www.reddit.com/r/Syncthing/comments/y0bj6h/many_conflicts_remove_synced_folder_and_add_it/",
    source_url="https://www.reddit.com/r/Syncthing/comments/y0bj6h/many_conflicts_remove_synced_folder_and_add_it/",
)

example_content: list[tuple[Content, bool, str]] = [
    (
        content_aw,
        True,
        "It is about a time-tracker. Matches user interest in (quantified self) and (open source).",
    ),
    (
        content_gpt3,
        True,
        "It is about GPT-3, a large language model. Matches user interest in (artificial intelligence).",
    ),
    (
        content_st_support,
        False,
        "It is about Syncthing support. Matches user dislike in (support questions).",
    ),
]
