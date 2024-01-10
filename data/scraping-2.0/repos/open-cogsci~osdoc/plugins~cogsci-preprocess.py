# encoding=utf-8

import os
import re
import sys
import sys
import jinja2
sys.path.insert(0, '/home/sebastiaan/git/academicmarkdown')
import yaml
import json
from pathlib import Path
from collections import OrderedDict
from pelican import signals
from pelican.readers import MarkdownReader
from markdown import Markdown
from markdown.extensions import codehilite
from markdown.extensions.toc import TocExtension
from markdown.extensions.tables import TableExtension
from academicmarkdown import build, HTMLFilter, _FigureParser
from translation_tools.translation_utils import LOCALES, parse_metadata
from langchain.text_splitter import MarkdownHeaderTextSplitter
from datamatrix import functional as fnc
from langchain.chat_models import ChatOpenAI
from translation_tools import translation_utils
if 'publishconf.py' in sys.argv:
    from publishconf import *
else:
    from pelicanconf import *

_FigureParser.figureTemplate[u'jekyll'] = u"""
![%(source)s](/""" + BRANCH + """%(source)s)

__Figure %(nFig)d.__ %(caption)s\n{: .fig-caption #%(id)s}\n
"""

ITEM_TYPES = [
    'SKETCHPAD', 'FEEDBACK', 'SEQUENCE', 'LOOP', 'SAMPLER', 'SYNTH', 'LOGGER',
    'INLINE_SCRIPT', 'RESET_FEEDBACK', 'COROUTINES', 'KEYBOARD_RESPONSE',
    'MOUSE_RESPONSE', 'JOYSTICK', 'SRBOX', 'TEXT_DISPLAY', 'FORM_BASE',
    'FORM_TEXT_INPUT', 'FORM_TEXT_DISPLAY', 'FORM_MULTIPLE_CHOICE',
    'FORM_CONSENT', 'FORM', 'MEDIA_PLAYER_VLC', 'MEDIA_PLAYER_GST',
    'MEDIA_PLAYER_MPY', 'MOUSETRAP', 'SOUND_START_RECORDING',
    'SOUND_STOP_RECORDING', 'TOUCH_RESPONSE', 'PYGAZE_INIT', 'PYGAZE_LOG',
    'PYGAZE_WAIT', 'PYGAZE_DRIFT_CORRECT', 'PYGAZE_STOP_RECORDING',
    'PYGAZE_START_RECORDING', 'THIS_STYLE', 'NOTEPAD', 'INLINE_JAVASCRIPT',
    'INLINE_HTML'
]
DEFAULT_SUMMARY_PROMPT = '''Write a comprehensive summary for the documentation below.

```markdown
{content}
```
'''
SUMMARY_PROMPT_PATTERN = r"<summary_prompt>(.*?)</summary_prompt>"

root = os.path.dirname(os.path.dirname(__file__)) + '/content'

with open('constants.yaml') as f:
    const = yaml.load(f, Loader=yaml.SafeLoader)


def orderedLoad(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):

    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


class AcademicMarkdownReader(MarkdownReader):

    enabled = True

    def read(self, source_path):

        """Parse content and metadata of markdown files"""
        self._source_path = source_path
        self._md = Markdown(
            extensions=[
                'markdown.extensions.toc',
                'markdown.extensions.tables',
                'markdown.extensions.meta',
                'markdown.extensions.extra',
                codehilite.CodeHiliteExtension(css_class='highlight'),
                ],
            )
        with open(source_path) as fd:
            text = fd.read()
        print(f'****** {source_path}')
        metadata = parse_metadata(text)
        print(metadata)
        # Fix things that have been incorrectly translated. This is a hacky
        # list that is created based on compilation errors.
        #
        # French
        text = text.replace(':manuel/', ':manual/')
        text = text.replace('api/clavier', 'api/keyboard')
        text = text.replace('api/souris', 'api/mouse')
        text = text.replace('api/horloge', 'api/clock')
        text = text.replace('coroutine%', 'coroutines%')
        text = text.replace('syntaxe :', 'syntax:')
        text = text.replace('inclure/', 'include/')
        text = text.replace('inclure:', 'include:')
        text = text.replace('inclure :', 'include:')
        text = text.replace('réponses.md', 'responses.md')
        # Chinese
        text = text.replace('包含/', 'include/')
        text = text.replace('包括/', 'include/')
        text = text.replace('figure：', 'figure:')
        text = text.replace('id：', 'id:')
        text = text.replace('source：', 'source:')
        text = text.replace('caption：', 'caption:')
        text = text.replace('width：', 'width:')
        text = text.replace('height：', 'height:')
        text = text.replace('视频:', 'video:')
        text = text.replace('视频：', 'video:')
        text = text.replace('来源:', 'source:')
        text = text.replace('来源：', 'source:')
        text = text.replace('源:', 'source:')
        text = text.replace('源：', 'source:')
        text = text.replace('视频id:', 'videoid:')
        text = text.replace('视频id：', 'videoid:')
        text = text.replace('宽度:', 'width:')
        text = text.replace('宽度：', 'width:')
        text = text.replace('高度:', 'height:')
        text = text.replace('高度：', 'height:')
        text = text.replace('标题:', 'caption:')
        text = text.replace('标题：', 'caption:')
        text = text.replace('图像:', 'figure:')
        text = text.replace('图像：', 'figure:')
        text = text.replace('图示:', 'figure:')
        text = text.replace('图示：', 'figure:')
        text = text.replace('图:', 'figure:')
        text = text.replace('图：', 'figure:')
        # German
        text = text.replace('Abbildung:', 'figure:')
        text = text.replace('Bedienungsanleitung/', 'manual/')
        text = text.replace('Quelle:', 'source:')
        text = text.replace('Quelle:', 'source:')
        text = text.replace('source: Youtube', 'source: youtube')
        text = text.replace('source: YouTube', 'source: youtube')
        text = text.replace('Bildunterschrift:', 'caption:')
        text = text.replace('Untertitel:', 'caption:')
        text = text.replace('Höhe:', 'height:')
        text = text.replace('Breite:', 'width:')
        text = text.replace('Video:', 'video:')
        text = text.replace('Id:', 'id:')
        text = text.replace('ID:', 'id:')
        text = text.replace('Videoid:', 'videoid:')
        text = text.replace('Code:', 'code:')
        text = text.replace('Syntax:', 'syntax:')
        # Spanish
        text = text.replace('fuente:', 'source:')
        text = text.replace(':descarga', ':download')
        text = text.replace(':texto', ':text')
        text = text.replace('ancho:', 'width:')
        text = text.replace('alto:', 'height:')
        text = text.replace('leyenda:', 'caption:')
        text = text.replace('ID del video:', 'videoid:')
        text = text.replace('incluir/', 'include/')
        text = text.replace('link:diseñoexperimental',
                            'link:experimentaldesign')
        # General
        text = text.replace('figure:', 'figure: ')
        text = text.replace('source:', 'source: ')
        text = text.replace('caption:', 'caption: ')
        text = text.replace('width:', 'width: ')
        text = text.replace('height:', 'height: ')
        text = text.replace('video:', 'video: ')
        text = text.replace('id:', 'id: ')
        # Recode include links based onthe locale
        if 'locale' in metadata:
            text = text.replace(' include/', f' include/{metadata["locale"]}/')
        # Extract links and paths
        links = locale_links[metadata.get('locale', None)]
        img_path = os.path.dirname(source_path) + '/img/' \
            + os.path.basename(source_path)[:-3]
        lst_path = os.path.dirname(source_path) + '/lst/' \
            + os.path.basename(source_path)[:-3]
        tbl_path = os.path.dirname(source_path) + '/tbl/' \
            + os.path.basename(source_path)[:-3]
        # The include path depends on the locale, because the included files
        # should also be translated
        for language, code in LOCALES:
            if code in Path(source_path).parts:
                include_path = f'include/{code}'
                break
        else:
            include_path = 'include'
        print(f'include path: {include_path}')
        build_path = build.path
        build.path = [include_path, img_path, lst_path, tbl_path] + build_path
        text = build.MD(text)
        text = jinja2.Template(text).render()
        # Process internal links
        for m in re.finditer('%link:(?P<link>[\w/-]+)%', text):
            full = m.group(0)
            link = m.group('link')
            print('link', link, full)
            if link not in links:
                raise Exception(u'%s not a key in %s' % (link, links))
            text = text.replace(full, '<%s/%s>' % (SITEURL, links[link]))
        for m in re.finditer('%url:(?P<link>[\w/-]+)%', text):
            full = m.group(0)
            link = m.group('link')
            print('url', link, full)
            if link not in links:
                raise Exception(u'%s not a key in %s' % (link, links))
            text = text.replace(full, '%s/%s' % (SITEURL, links[link]))
        for m in re.finditer('%static:(?P<link>[\w/.-]+)%', text):
            full = m.group(0)
            link = m.group('link')
            print('static', link, full)
            text = text.replace(full, '<%s/%s>' % (SITEURL, link))
        text = text.replace(root, u'')
        text = HTMLFilter.DOI(text)
        content = self._md.convert(re.sub(SUMMARY_PROMPT_PATTERN, '', text,
                                   flags=re.DOTALL))
        for var, val in const.items():
            content = content.replace(u'$%s$' % var, str(val))
        for item_type in ITEM_TYPES:
            content = content.replace(item_type,
                u'<span class="item-type">%s</span>' % item_type.lower())
        build.path = build.path[3:]
        metadata = self._parse_metadata(self._md.Meta)
        build.path = build_path
        # Write content to a JSON file that is ready for indexing by
        # sigmund. We do this only for english, which is the only language
        # with out a locale in the metadata. We also don't include pages aren't
        # translated, because those tend to be workshop pages etc.
        includes = ['manual/', 'items/', 'beginner.md', 'intermediate.md',
                    'intermediate-javascript.md']
        if 'locale:' not in text and 'translate: false' not in text and \
                any(include in source_path for include in includes):
            url = SITEURL + source_path[len(build_path[0]) + 14:-3]
            # A summary prompt can be embedded in the text in pseudo tags. If
            # so, then we use that to summarize the page, otherwise we use a
            # default summary tool. For chunks of text that are relatively
            # short, we don't summarize at all.
            match = re.search(SUMMARY_PROMPT_PATTERN, text, re.DOTALL)
            if match:
                print("Summary prompt is defined")
                summary_prompt = match.group(1).strip()
                text = re.sub(SUMMARY_PROMPT_PATTERN, '', text,
                              flags=re.DOTALL).strip()
            elif len(text.split()) > 800 and len(text.split()) < 5000:
                print("No summary prompt is defined")
                summary_prompt = DEFAULT_SUMMARY_PROMPT
            else:
                summary_prompt = None
            with open('sigmund-sources.jsonl', 'a') as fd:
                if summary_prompt is not None:
                    query = summary_prompt.format(content=text)
                    summary = predict(query)
                    fd.write(json.dumps(
                        {'content': summary,
                         'url': url,
                         'title': metadata['title']}) + '\n')
                else:
                    for chunk in markdown_split(text):
                        print(f'{url} {len(chunk)}')
                        fd.write(json.dumps(
                            {'content': chunk,
                             'url': url,
                             'title': metadata['title']}) + '\n')
        return content, metadata


def markdown_split(md, maxlen=5000):
    while len(md) > maxlen:
        split_point = md[:len(md) // 2].rfind('\n#')
        yield md[:split_point + 1]
        md = md[split_point + 1:]
    yield md


@fnc.memoize(persistent=True)
def predict(query):
    llm = ChatOpenAI(model='gpt-4-1106-preview')
    answer = llm.predict(query)
    print(answer)
    return answer


def init_academicmarkdown(sender):

    with open('sigmund-sources.jsonl', 'w') as fd:
        pass
    build.postMarkdownFilters = []
    build.figureTemplate = 'jekyll'
    build.tableTemplate = 'kramdown'
    build.figureSourcePrefix = SITEURL
    build.extensions.remove('toc')
    build.extensions.insert(0, 'toc')
    
    global links, duplicate_names, locale_links
    locale_links = {}
    for locale in [None] + [code for _, code in LOCALES]:
        links = {}
        duplicate_names = []
        if locale is not None:
            print(f'loading sitemap-{locale}.yaml')
            with open(f'sitemap/sitemap-{locale}.yaml') as f:
                d = orderedLoad(f)
        else:
            print('loading sitemap.yaml')
            with open('sitemap/sitemap.yaml') as f:
                d = orderedLoad(f)
        process_links(d)
        locale_links[locale] = links


def isseparator(pagename):

    for ch in pagename:
        if ch != '_':
            return False
    return True


def process_links(d):

    for pagename, entry in d.items():
        if pagename.startswith('__'):
            continue
        if isinstance(entry, list):
            entry = entry[0]
        if isseparator(pagename) or entry in [None, '']:
            continue
        if isinstance(entry, dict):
            process_links(entry)
            continue
        name = entry.split('/')[-1]
        url = entry
        if not name.strip():
            continue
        clean_entry = entry
        for language, code in LOCALES:
            if entry.startswith(code + '/'):
                entry = entry[len(code) + 1:]
                break
        links[entry] = url
        links['/' + entry] = url
        if entry == name or name in duplicate_names:
            continue
        if name not in links:
            links[name] = url
            links['/' + name] = url
            continue
        duplicate_names.append(name)
        del links[name]
        del links['/' + name]
        print('Duplicate name: %s' % name)


def add_reader(readers):

    readers.reader_classes['md'] = AcademicMarkdownReader


def register():

    signals.readers_init.connect(add_reader)
    signals.initialized.connect(init_academicmarkdown)
