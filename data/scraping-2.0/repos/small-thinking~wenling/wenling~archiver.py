"""
"""
import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from bs4 import BeautifulSoup, Tag

from wenling.common.model_utils import OpenAIChatModel
from wenling.common.notion_utils import NotionStorage
from wenling.common.utils import *


class ArchiverOrchestrator:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = Logger(logger_name=os.path.basename(__file__), verbose=verbose)
        self.archivers: List[Dict[str, Any]] = [
            {
                # Match url has mp.weixin.qq.com in it.
                "match_regex": r"^https://mp\.weixin\.qq\.com/s/.*$",
                "archiver": WechatArticleArchiver(verbose=verbose),
            },
        ]
        self.default_archiver = WebPageArchiver(verbose=verbose)

    async def archive(self, url: str) -> str:
        """Match the url with pattern and find the corresponding archiver."""
        for archiver in self.archivers:
            if re.match(pattern=archiver["match_regex"], string=url):
                if self.verbose:
                    self.logger.info(f"Archive url with archiver {archiver['archiver'].name}...")
                page_id = await archiver["archiver"].archive(url)
                return page_id
        # Match to general web archiver by default.
        if self.verbose:
            self.logger.info(f"Archive url with archiver general web archiver...")
        page_id = await self.default_archiver.archive(url)
        return page_id


class Archiver(ABC):
    """Archiver is a tool used to archive the bookmarked articles."""

    def __init__(self, vendor_type: str = "openai", verbose: bool = False, **kewargs):
        load_env()
        self.api_key = os.getenv("ARCHIVER_API_KEY")
        self.verbose = verbose
        self.logger = Logger(logger_name=os.path.basename(__file__), verbose=verbose)
        self.notion_store = NotionStorage(verbose=verbose)
        if vendor_type == "openai":
            self.model = OpenAIChatModel()
        else:
            raise NotImplementedError
        self.name = self._set_name()
        self._extra_setup()

    def _extra_setup(self):
        pass

    @abstractmethod
    def _set_name(self) -> str:
        pass

    def _consolidate_content(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge the consecutive texts or code blocks into one. The max size of each block should be less than 2000."""
        if self.verbose:
            self.logger.info("Consolidate the consecutive texts or code blocks into one...")
        consolidated_content: List[Dict[str, Any]] = []
        for block in content:
            if block["type"] in ["text", "code"]:
                if consolidated_content and consolidated_content[-1]["type"] == block["type"]:
                    if len(consolidated_content[-1]["text"]) + len(block["text"]) < 2000:
                        consolidated_content[-1]["text"] += "\n" + block["text"]
                    else:
                        consolidated_content.append(block)
                else:
                    consolidated_content.append(block)
            else:
                consolidated_content.append(block)
        return consolidated_content

    async def archive(self, url: str) -> str:
        if not check_url_exists(url):
            raise ValueError(f"The url {url} does not exist.")
        article_json_obj = await self._archive(url)
        return await self.notion_store.store(json_obj=article_json_obj)

    @abstractmethod
    async def _archive(self, url: str) -> Dict[str, Any]:
        pass

    def list_archived(self) -> List[str]:
        return []


class WechatArticleArchiver(Archiver):
    """
    WechatArticleArchiver is a tool used to archive the bookmarked wechart articles.
    """

    def __init__(self, vendor_type: str = "openai", verbose: bool = False, **kwargs):
        super().__init__(vendor_type=vendor_type, verbose=verbose)
        self.root_css_selector = "div#img-content.rich_media_wrp"

    def _set_name(self) -> str:
        return "WechatArticleArchiver"

    def _parse_title(self, element_bs: BeautifulSoup) -> str:
        """Get the title from the first h1 element, put it into {"type": "h1", "text": <title>}."""
        title_element = element_bs.select_one("h1")
        if not title_element:
            self.logger.warning("Cannot find title element.")
            title = "Untitled"
        else:
            title = title_element.get_text().strip()
        return title

    def _parse_author(self, element_bs: BeautifulSoup) -> str:
        """Get the author name from a sub element with class "rich_media_meta rich_media_meta_text",
        put it into {"type": "h2", "text": <author_name>}.
        """
        author_element = element_bs.select_one(".rich_media_meta.rich_media_meta_text")
        if not author_element:
            raise ValueError("Cannot find author element.")
        author = author_element.get_text().strip()
        return author

    def _parse_publish_time(self, element_bs: BeautifulSoup) -> Dict[str, str]:
        """Get the publish time from a sub element (not direct sub) with class "detail-time",
        and put it into {"type": "h2", "text": <publis_time>}
        """
        publish_time_element = element_bs.select_one(".detail-time")
        if not publish_time_element:
            raise ValueError("Cannot find publish time element.")
        publish_time = publish_time_element.get_text().strip() if publish_time_element else "Not available"
        return {"type": "h2", "text": publish_time}

    def _parse_tags(self, element_bs: BeautifulSoup) -> List[str]:
        """Get the tags from a sub elements (not direct sub) each with class "article-tag__item",
        and put them into {"type": "text", "text": <comma separated tags>}
        """
        tags_element = element_bs.select(".article-tag__item")
        if not tags_element:
            raise ValueError("Cannot find tags element.")
        tags = [tag.get_text().strip() for tag in tags_element]
        return tags

    def _parse_paragraph(self, paragraph_tag: Tag, cache: Dict[str, Any]) -> List[Dict[str, str]]:
        # 1. Each <p style="visibility: visible;">content</> is a paragraph, and will be put as an individual dictionary.
        # 1. 1. If the content a simple text, it will be stored as {"type": "text", "text": <content>}.
        # 1. 2. If the content is an image, it will be stored as {"type": "image", "url": <url>}.
        # 2. A <blockquote style="visibility: visible;">...</blockquote>, then store as {"type": "quote", "text": <content>}.
        # 3. A <span data-vw...>...</span> indicate a video, and store as {"type": "video", "url": <url>}. The url can be obtained from the attr data-src.
        parsed_elements = []
        try:
            if paragraph_tag.name == "p":
                if "text-align: center;" in paragraph_tag.get("style", "") and paragraph_tag.find("strong"):
                    # <strong> directly wrapped by <p style="text-align: center;">
                    strong_text = paragraph_tag.find("strong").get_text().strip()
                    if strong_text not in cache:
                        parsed_elements.append({"type": "h2", "text": strong_text})
                        cache[strong_text] = True
                elif paragraph_tag.find("img"):  # Check for images
                    image_url = paragraph_tag.find("img")["data-src"]
                    if image_url not in cache:
                        parsed_elements.append({"type": "image", "url": image_url})
                        cache[image_url] = True
                else:  # Regular text content, including <strong> not in center-aligned <p>
                    for child in paragraph_tag.contents:
                        if child.name == "strong":
                            strong_text = child.get_text().strip()
                            if strong_text not in cache:
                                parsed_elements.append({"type": "h2", "text": strong_text})
                                cache[strong_text] = True
                        elif child.string:
                            text = child.string.strip()
                            if text:
                                if text not in cache:
                                    parsed_elements.append({"type": "text", "text": text})
                                    cache[text] = True
            elif paragraph_tag.name == "blockquote":
                text = paragraph_tag.get_text().strip()
                if text not in cache:
                    parsed_elements.append({"type": "quote", "text": text})
                    cache[text] = True
            elif paragraph_tag.name == "span":
                text = paragraph_tag.get_text().strip()
                if text:
                    if text not in cache:
                        parsed_elements.append({"type": "text", "text": text})
                        cache[text] = True
            elif paragraph_tag.name == "span" and paragraph_tag.get("data-vw"):
                video_url = paragraph_tag.get("data-src")
                if video_url not in cache:
                    parsed_elements.append({"type": "video", "url": video_url})
                    cache[video_url] = True
            elif paragraph_tag.name in ["ul", "ol"]:
                for li in paragraph_tag.find_all("li"):
                    li_text = li.get_text().strip()
                    if li_text:
                        if li_text not in cache:
                            parsed_elements.append({"type": "text", "text": li_text})
                            cache[li_text] = True
            elif paragraph_tag.name == "em":
                em_text = paragraph_tag.get_text().strip()
                if em_text not in cache:
                    parsed_elements.append({"type": "text", "text": em_text})
                    cache[em_text] = True
            elif paragraph_tag.name == "figure":
                image_url = paragraph_tag.find("img")["data-src"]
                if image_url not in cache:
                    parsed_elements.append({"type": "image", "url": image_url})
                    cache[image_url] = True
            else:
                content = paragraph_tag.get_text().strip()
                if content:
                    if content not in cache:
                        parsed_elements.append({"type": f"{paragraph_tag.name}", "text": content})
                        cache[content] = True
        except Exception as e:
            raise ValueError(f"Error parsing content.")

        return parsed_elements

    def _parse_section(self, section_tag: Tag, cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        content_list = []
        try:
            has_descendants = False
            for tag in section_tag.descendants:
                if tag.name in ["p", "blockquote", "span", "ul", "ol", "figure"]:
                    has_descendants = True
                    blob = self._parse_paragraph(tag, cache)
                    if blob:
                        content_list.extend(blob)
            if not has_descendants:
                content = section_tag.get_text().strip()
                if content:
                    content_list.append({"type": "text", "text": content})
        except Exception as e:
            raise ValueError(f"Error parsing content.")
        return content_list

    def _parse_content(self, element_bs: BeautifulSoup) -> List[Dict[str, Any]]:
        """Get the content from a sub element with id "js_content",
        pass the entire blob as string to function _parse_content.
        """
        content_element = element_bs.select_one("#js_content")
        content_json_obj: List[Dict[str, Any]] = []

        if not content_element:
            raise ValueError("The content element is not found.")

        # Initialize cache here
        cache: Dict[str, Any] = {}
        try:
            # Process all <section> tags first
            for section in content_element.find_all("section", recursive=False):
                content_json_obj.extend(self._parse_section(section, cache))

            # Process <p>, <ul>, <ol> tags that are direct children of the content_element
            for tag in content_element.find_all(["p", "blockquote", "span", "ul", "ol", "figure"], recursive=False):
                blob = self._parse_paragraph(tag, cache)
                if blob:
                    content_json_obj.extend(blob)
        except Exception as e:
            raise ValueError(f"Error parsing content.")

        return content_json_obj

    async def _archive(self, url: str) -> Dict[str, Any]:
        """Get the content block from the web page with the path div#img-content.rich_media_wrp.
        Parse the elements and put them into a json object with list of elements.
        """
        element = fetch_url_content(url=url, css_selector=self.root_css_selector)
        element_bs = BeautifulSoup(element, "html.parser")
        if not element:
            raise ValueError(f"The url {url} does not have the element {self.root_css_selector}.")
        try:
            paragraphs = self._parse_content(element_bs=element_bs)
            # Consolidate the consecutive texts or code blocks into one.
            paragraphs = self._consolidate_content(paragraphs)
            article_json_obj: Dict[str, Any] = {
                "properties": {},
                "children": paragraphs,
            }
            article_json_obj["properties"]["url"] = url
            article_json_obj["properties"]["title"] = self._parse_title(element_bs=element_bs)
            article_json_obj["properties"]["type"] = "微信"
            article_json_obj["properties"]["datetime"] = get_datetime()
            tags = self._parse_tags(element_bs=element_bs) + [self._parse_author(element_bs=element_bs)]
            tags = [tag.replace("#", "") for tag in tags if len(tag) > 1]
            article_json_obj["properties"]["tags"] = tags
            if self.verbose:
                json_object_str = json.dumps(article_json_obj, indent=2)
                self.logger.info(f"Archived article: {json_object_str}")
        except Exception as e:
            raise ValueError(f"Error parsing content. Details: {str(e)}")
        finally:
            return article_json_obj


class WebPageArchiver(Archiver):
    """
    WebPageArchiver is a tool used to archive the bookmarked web pages.
    """

    def __init__(self, vendor_type: str = "openai", verbose: bool = False, **kwargs):
        super().__init__(vendor_type=vendor_type, verbose=verbose)
        self.root_css_selector = "html"
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.temperature = kwargs.get("temperature", 0.0)
        self._extra_setup()

    def _extra_setup(self):
        self._set_name()

    def _set_name(self) -> str:
        return "WebPageArchiver"

    def _parse_title(self, element_bs: BeautifulSoup) -> str:
        """Get the title from the head -> title element, og:title, or twitter:title meta tags."""
        if self.verbose:
            self.logger.info(f"Extracting the title from the html...")

        # First, try to get the title from the <title> tag
        title_element = element_bs.select_one("head > title")
        if title_element and title_element.get_text().strip():
            return title_element.get_text().strip()

        # If not found or empty, try the og:title meta tag
        og_title_element = element_bs.select_one("meta[property='og:title']")
        if og_title_element and og_title_element.get("content", "").strip():
            return og_title_element["content"].strip()

        # If still not found, try the twitter:title meta tag
        twitter_title_element = element_bs.select_one("meta[property='twitter:title']")
        if twitter_title_element and twitter_title_element.get("content", "").strip():
            return twitter_title_element["content"].strip()

        # If none of these are found, default to "Untitled"
        if self.verbose:
            self.logger.warning("Cannot find title element.")
        return "Untitled"

    async def _parse_segment(self, batch_id: int, segment_tags: List[Tag]) -> List[Dict[str, Any]]:
        """Extract the image urls and all texts from the given list of segment tags.
        The returned list of extracted blocks will be merged into a long list.
        """
        texts = ""
        for tag in segment_tags:
            texts += tag.get_text().strip() + "\n"
        texts = texts[: self.max_tokens]
        prompt = f"""
        Please help organize the raw contents extracted from a html into strucutred json blob.
        The returned json keyed by "content" will be merged into a long list.
        
        Please note:
        1. The html can contain titles (h1, h2, h3), texts, codes, images (<img>), and videos. 
        2. If the texts themselves have ", please replace it with '.
        3. Please merge consecutive code lines or paragraphs into one entry.
        
        The json blob is expected to have the following structure:
        {{
            "content": [
                {{
                    "type": "h1",
                    "text": "Section title",
                }},
                {{
                    "type": "h2",
                    "text": "Subsection title",
                }},
                {{
                    "type": "text",
                    "text": "Paragraph text",
                }},
                {{
                    "type": "code",
                    "text": "Code",
                }},
                {{
                    "type": "image",
                    "url": "https://image_url",
                }},
                {{
                    "type": "video",
                    "url": "https://video_url",
                }}
            ]
        }}
        
        The contents are below:
        {texts}
        """
        json_response_str = self.model.inference(
            user_prompt=prompt, max_tokens=self.max_tokens, temperature=self.temperature, response_format="json_object"
        )
        try:
            json_obj = json.loads(json_response_str)
            paragraphs = json_obj.get("content", [])
            return paragraphs
        except Exception as e:
            self.logger.error(f"Error parsing the content. Details: {str(e)}. Return empty paragraphs.")
            return []

    async def _parse_content(self, element_bs: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract the image urls and all texts.
        And then leverage LLM to structurize the content into a json blob.
        """
        if self.verbose:
            self.logger.info("Extract the image urls and all texts...")
        body_tag = element_bs.select_one("body")

        segments_to_include = ["p", "div", "h1", "h2", "img", "pre"]
        segment_tags = body_tag.find_all(segments_to_include)
        if self.verbose:
            self.logger.info(f"Total number of segments: {len(segment_tags)}")

        # Filter out nested elements
        unique_segments = []
        for segment in segment_tags:
            # Check if the segment is not a child of any other segments
            if not any(parent in segment.parents for parent in segment_tags):
                unique_segments.append(segment)

        if self.verbose:
            self.logger.info(f"Total number of unique segments: {len(unique_segments)}")

        paragraphs: List[Dict[str, Any]] = []
        for batch_id in range(0, len(unique_segments), 20):
            segment_tags_batch = unique_segments[batch_id : batch_id + 20]
            if self.verbose:
                self.logger.info(f"Processing segment batch {batch_id}...")
            paragraphs.extend(await self._parse_segment(batch_id=batch_id, segment_tags=segment_tags_batch))
        # Consolidate.
        paragraphs = self._consolidate_content(paragraphs)

        return paragraphs

    async def _parse_tags(self, paragraphs: List[Dict[str, Any]]) -> List[str]:
        """Leverage the LLM to auto-generate the tags based on the contents."""
        if self.verbose:
            self.logger.info(f"Generate tags based on the contents...")
        contents_str = "\n".join([paragraph.get("text", "") for paragraph in paragraphs])
        prompt = f"""
        Please help generate the tags based on the contents below:
        ---
        {contents_str}
        ---
        
        Please generate the tags in the same language as the contents, and return in below json format:
        {{
            "tags": ["tag1", "tag2", "tag3"]
        }}
        """
        json_response_str = await self.model.inference(
            user_prompt=prompt, max_tokens=256, temperature=0.0, response_format="json_object"
        )
        try:
            json_obj = json.loads(json_response_str)
            tags = json_obj.get("tags", [])
            return tags
        except Exception as e:
            self.logger.error(f"Error parsing the tags. Details: {str(e)}. Return empty tags.")
            return []

    async def _archive(self, url: str) -> Dict[str, Any]:
        """Get the content block from the web page.
        Parse the elements and put them into a json object with list of elements.
        """
        # TODO:
        # 1. Get the content from the body block.
        element = fetch_url_content(url=url, css_selector=self.root_css_selector)
        element_bs = BeautifulSoup(element, "html.parser")
        if not element:
            raise ValueError(f"The url {url} does not have the element {self.root_css_selector}.")
        article_json_obj: Dict[str, Any] = {
            "properties": {},
            "children": {},
        }
        try:
            article_json_obj["properties"]["url"] = url
            article_json_obj["properties"]["title"] = self._parse_title(element_bs=element_bs)
            article_json_obj["properties"]["type"] = "网页"
            article_json_obj["properties"]["datetime"] = get_datetime()
            contents = await self._parse_content(element_bs=element_bs)
            article_json_obj["children"] = contents
            # Leverage LLM to generate the tags based on the article json obj contents.
            tags = await self._parse_tags(contents)
            tags = [tag.replace("#", "") for tag in tags if len(tag) > 1]
            article_json_obj["properties"]["tags"] = tags
            if self.verbose:
                json_object_str = json.dumps(article_json_obj, indent=2)
                self.logger.info(f"Archived article: {json_object_str}")
        except Exception as e:
            raise ValueError(f"Error parsing content. Details: {str(e)}")
        finally:
            return article_json_obj
