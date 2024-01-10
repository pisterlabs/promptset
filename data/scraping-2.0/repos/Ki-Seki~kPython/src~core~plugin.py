"""
This file store all plugins.
"""


# Register new plugin here. One line for each plugin.
__all__ = [
    'gen_shortcut', 
    'focus',
    'chat',
    'arxiv_yesterday'
    ]


def gen_shortcut(dst="."):
    """
    Create a shortcut for kPython in directory `dst`.

    If `dst` is in the system PATH, then you can 
    use kPython by typing `kPython` instead of 
    changing directory and typing `python kPython.py`.
    """

    import os

    dst_path = f"{dst}/kPython" + (".cmd" if os.name == 'nt' else ".sh")
    with open(dst_path, "w") as f:
        f.write(f"cd {os.getcwd()}/src/\n")
        f.write("python kPython.py")


def focus(m=45, r=15):
    """
    Focus on working for `m` minutes, and then rest for `r` minutes.

    Requirements:
    * Windows platform only
    * `pip install winsound`
    """

    from time import sleep
    import winsound

    while True:
        sleep(m * 60)
        winsound.Beep(1000, 3000)
        sleep(r * 60)
        winsound.Beep(1000, 3000)


def chat(prompt):
    """
    ChatGPT in terminal.

    Requirements:
    * Add your own OPENAI_API_KEY to the OS environment variables.
    * `pip install openai`

    More:
    * Visit: https://platform.openai.com/docs/api-reference/authentication
    """

    import os
    import openai

    openai.api_key = os.getenv('OPENAI_API_KEY')
    completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        timeout=1000,
    )
    return completion.choices[0].text

def arxiv_yesterday(kw:str, max_results=-1, show_abstract=True, download=True, path='.'):
    """
    Fetch the arXiv paper whose title or abstract contains `kw` yesterday.

    * Fetch papers firstly submitted yesterday.
    * Fetch at most `max_results` papers. Fetch all possible papers if `max_results=-1`.
    * Paper abstract will be printed if `show_abstract=True`
    * Papers will be downloaded if `download=True`
    * Downloaded papers will be saved in `path`.

    Requirements:
    * `pip install arxiv`

    More:
    * arXiv official API: https://info.arxiv.org/help/api/index.html
    """

    import os
    import datetime
    import arxiv

    query = f'ti:{kw} OR abs:{kw}'  # Title or abstract contains `kw`
    search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.SubmittedDate)
    print('Fetching papers...')
    papers = list(search.results())

    # Ensure the published date is yesterday
    year, month, yesterday = datetime.date.today().year, \
                       datetime.date.today().month, \
                       datetime.date.today().day - 1
    papers = filter(lambda x: (x.published.year==year 
                               and x.published.month==month 
                               and x.published.day==yesterday)
                               , papers)
    
    papers = list(papers)
    papers = papers[:max_results] if max_results != -1 else papers
    cnt = len(papers)
    print(f'Found {cnt} papers.')

    for i, paper in enumerate(papers):
        print(f'TITLE: {paper.title}')
        print(f'LINK: {paper.links[0]}')
        print(f'ABSTRACT: {paper.summary}') if show_abstract else ()
        if download:
            os.mkdir(path) if not os.path.exists(path) else ()
            paper.download_pdf(dirpath=path, filename=f'{paper.get_short_id()}.pdf')
            print(f'Paper downloaded {i+1}/{cnt}.')
        print()

        
if __name__ == '__main__':
    arxiv_yesterday('LLM', path='./papers/')
