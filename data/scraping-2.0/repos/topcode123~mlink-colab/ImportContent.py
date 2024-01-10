import datetime
import random

from newspaper import Config
import html.parser
from urllib.parse import urlparse
import requests
import base64
import html
import html.parser
from Settings import *
from bson import ObjectId
from pymongo import MongoClient
import time
from PIL import Image
import io

from SpinService import SpinService
from extract import ContentExtractor
import re
import logging
import openai

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.ERROR)

logger = logging.getLogger(__name__)


def no_accent_vietnamese(s):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s


spinService = SpinService()

config = Config()
campaign_root = MongoClient(CONNECTION_STRING_MGA1).campaigns.mlink
comment_queue = MongoClient(CONNECTION_STRING_MGA1).campaigns.comment_queue
keywords = MongoClient(CONNECTION_STRING_MGA1).campaigns.mlinkkeywords
mlink_report_posts = MongoClient(CONNECTION_STRING_MGA1).campaigns.mlinkreportposts

contentExtractor = ContentExtractor(config)
from bs4 import BeautifulSoup, Tag


def replace_attr(soup, from_attr: str, to_attr: str):
    if from_attr in str(soup):
        soup[to_attr] = soup[from_attr]
        del soup[from_attr]

        return soup
    else:
        return soup


def process_content(article, url):
    print("process content")
    article.article_html = str(html.unescape(article.article_html))

    soup = BeautifulSoup(article.article_html, 'html.parser')
    self_url = str(no_accent_vietnamese(url["keyword"].replace("\r", ""))) + ' ' + \
               str(time.time()).split(".")[0]
    self_url = self_url.replace(" ", "-")
    self_url = self_url.replace("--", "-")
    self_url = self_url.replace(".", "")

    self_url = self_url.replace("\n", "")
    self_url = self_url.replace(",", "")
    self_url = self_url.replace("(", "")
    self_url = self_url.replace(")", "")
    self_url = self_url.replace("<", "")
    self_url = self_url.replace(">", "")
    self_url = self_url.replace("[", "")
    self_url = self_url.replace("]", "")
    domain = urlparse(url["link"]).netloc
    images = soup.find_all("img")
    src_img = []
    for image in images:
        try:
            if image.has_attr("src"):
                if image["src"] in src_img:
                    image.decompose()
                else:
                    if "http" in image["src"]:
                        src_img.append(image["src"])
                    elif image["src"] == "":
                        image.decompose()
                    elif image["src"][:2] == "//" and "." in image["src"][3:].split("/")[0] and "jpg" not in \
                            image["src"][3:].split("/")[0] and "png" not in image["src"][3:].split("/")[0]:
                        image["src"] = "http:" + image["src"]
                        src_img.append(image["src"])
                    else:
                        image["src"] = "http://" + domain + image["src"]
                        src_img.append(image["src"])

            elif image.has_attr("srcset"):
                if image["srcset"] in src_img:
                    image.decompose()
                else:
                    if "http" in image["srcset"]:
                        src_img.append(image["srcset"])
                    elif image["srcset"] == "":
                        image.decompose()
                    elif image["srcset"][:2] == "//" and "." in ["srcset"][3:].split("/")[0] and "jpg" not in \
                            image["srcset"][3:].split("/")[0] and "png" not in image["srcset"][3:].split("/")[0]:
                        image["srcset"] = "http" + image["srcset"]
                        src_img.append(image["srcset"])
                    else:
                        image["srcset"] = "http://" + domain + image["srcset"]
                        src_img.append(image["srcset"])
        except Exception as e:
            print(e)

    thumb = None
    if len(src_img) > 0:
        try:
            for _ in range(5):
                thumb = random.choice(src_img)
                if ".PNG" in thumb or ".JPG" in thumb:
                    break
        except Exception as e:
            print(e)
            pass
    # if url["campaign"]["Top10url"]:
    #     if len(url["campaign"]["Top10url"]) > 0:
    #         internal_link_total = random.choice(url["campaign"]["Top10url"])
    #         internal_link = internal_link_total["link"]
    #         internal_link_title = internal_link_total["name"]
    #         internal_link_total2 = random.choice(url["campaign"]["Top10url"])
    #         internal_link2 = internal_link_total2["link"]
    #         internal_link_title2 = internal_link_total2["name"]
    # else:
    #     internal_link = None
    #     internal_link_title = None
    #     internal_link2 = None
    #     internal_link_title2 = None
    internal_link = None
    internal_link_title = None
    internal_link2 = None
    internal_link_title2 = None

    # todo: temporary comment for debugging
    # if url["campaign"]["CategoryId"] is not None and url["campaign"]["CategoryName"] is not None and url["campaign"][
    #     "CategoryLink"] is not None:
    #     cate_name = url["campaign"]["CategoryName"]
    #     cate_link = url["campaign"]["CategoryLink"]
    # else:
    #     cate_name = None
    #     cate_link = None
    cate_name = None
    cate_link = None
    article.article_html = str(soup)
    paper = html.unescape(article.article_html)
    paper = BeautifulSoup(paper, "html.parser")
    for elem in paper.find_all(['a']):
        elem.unwrap()
    domain = domain.split(".")
    domain[-2] = list(domain[-2])
    domain[-2][0] = ".?"
    domain[-2][-1] = ".?"
    domain[-2][2] = ".?"
    domain[-2][-2] = ".?"
    domain[-2] = "".join(domain[-2])
    domain = ".".join(domain)
    article.title = re.sub(re.compile(domain), url["web_info"]["Website"], article.title)
    titles = []
    for i in article.title.split(" "):
        if ".com" in i or ".org" in i or ".vn" in i or ".us" in i or ".mobi" in i or ".gov" in i or ".net" in i or ".edu" in i or ".info" in i:
            titles.append(url["web_info"]["Website"])
        else:
            titles.append(i)
    article.title = " ".join(titles)

    for elem in paper.find_all(["img"], {"alt": re.compile("https://" + domain)}):
        elem['alt'] = re.sub(re.compile("https://" + domain), url["web_info"]["Website"], elem['alt'])
    for elem in paper.find_all(text=re.compile("https://" + domain)):
        elem = elem.replace_with(re.sub(re.compile("https://" + domain), url["web_info"]["Website"], elem))
    heading_p = []
    for heading in soup.find_all(["h1", "h2", "h3"]):
        for p in heading.find_all("p"):
            heading_p.append(p)
    thepp = paper.find_all('p')
    list_p_tag = []
    for i in thepp:
        if i not in heading_p:
            list_p_tag.append(i)

    if "campaign" in url and len(url["campaign"]["Top10url"]) > 0:
        if url["campaign"]["language"] == "vi":
            if internal_link and internal_link_title:
                internal_link_p_tag1 = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Xem thêm: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> </p></div>'.format(
                    internal_link, internal_link_title, internal_link_title)
                internal_link_p_tag1 = BeautifulSoup(internal_link_p_tag1, "html.parser")
                try:
                    list_p_tag[int(len(list_p_tag) / 2)].append(internal_link_p_tag1)
                except:
                    pass

            self_link_p_tag = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Bạn đang đọc: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> </p></div>'.format(
                url["web_info"]["Website"] + '/' + self_url, article.title, article.title)
            if internal_link2 and internal_link_title2:
                internal_link_p_tag2 = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Xem thêm: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a></p></div>'.format(
                    internal_link2, internal_link_title2, internal_link_title2)
                internal_link_p_tag2 = BeautifulSoup(internal_link_p_tag2, "html.parser")
                try:
                    list_p_tag[len(list_p_tag) - 4].append(internal_link_p_tag2)
                except Exception as e:
                    print(e)
                    pass

            self_link_p_tag = BeautifulSoup(self_link_p_tag, "html.parser")
            try:
                list_p_tag[min(len(list_p_tag), 3)].append(self_link_p_tag)
            except Exception as e:
                print(e)
                pass
        else:
            if internal_link and internal_link_title:
                internal_link_p_tag1 = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Read more : <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> </p></div>'.format(
                    internal_link, internal_link_title, internal_link_title)
                internal_link_p_tag1 = BeautifulSoup(internal_link_p_tag1, "html.parser")
                try:
                    list_p_tag[int(len(list_p_tag) / 2)].append(internal_link_p_tag1)
                except Exception as e:
                    print(e)
                    pass

            self_link_p_tag = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Reading: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> </p></div>'.format(
                url["web_info"]["Website"] + '/' + self_url, article.title, article.title)
            if internal_link2 and internal_link_title2:
                internal_link_p_tag2 = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Read more : <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a></p></div>'.format(
                    internal_link2, internal_link_title2, internal_link_title2)
                internal_link_p_tag2 = BeautifulSoup(internal_link_p_tag2, "html.parser")
                try:
                    list_p_tag[len(list_p_tag) - 4].append(internal_link_p_tag2)
                except:
                    pass

            self_link_p_tag = BeautifulSoup(self_link_p_tag, "html.parser")
            try:
                list_p_tag[min(len(list_p_tag), 3)].append(self_link_p_tag)
            except:
                pass

    if cate_link and cate_name:
        nguon = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Source: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> <br> Category : <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> </p></div>'.format(
            url["web_info"]["Website"] + '/', url["web_info"]["Website"], url["web_info"]["Website"], cate_link,
            cate_name, cate_name)

        nguon = BeautifulSoup(nguon, "html.parser")
    else:
        nguon = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Source: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a>'.format(
            url["web_info"]["Website"] + '/', url["web_info"]["Website"], url["web_info"]["Website"])
        nguon = BeautifulSoup(nguon, "html.parser")

    paper.append(nguon)
    # listp = [{"ptag": m, "keywords": url["keyword"], "language": url["language"]} for m in
    #          paper.find_all("p")]
    listp = []
    for p_tag in paper.find_all("p"):
        listp.append({"ptag": p_tag, "keywords": url["keyword"], "language": url["language"]})

    resultp = []
    replaced = {
        "is_replaced": False
    }

    keyword_replace = url["keyword"]
    anchor_text = url["anchortext"]
    base_url = url["baseUrl"]
    for i in listp:
        if i["language"] == "vi":
            spinned = spinService.spin_paragraph(i["ptag"], i["keywords"], replaced, keyword_replace, anchor_text, base_url)
            resultp.append(spinned)
        else:
            resultp.append(spinService.spin_paragraph_en(i["ptag"], i["keywords"], replaced, keyword_replace, anchor_text, base_url))
    

    for index in range(0, len(resultp)):
        if 3 == index:
            soup = BeautifulSoup(f"open_blockquote{str(resultp[index])}", 'html.parser')
            resultp[index] = soup
        
        if index == len(resultp) - 3:
            soup = BeautifulSoup(f"{str(resultp[index])}close_blockquote", 'html.parser')
            resultp[index] = soup


    for k1, k2 in zip(listp, resultp):
        k1["ptag"].replace_with(k2)
    paper = str(paper)
    paper = paper.replace("&lt;", "<")
    paper = paper.replace("&gt;", ">")
    paper = paper.replace(" . ", ". ")
    paper = paper.replace(" , ", ", ")
    paper = paper.replace("open_blockquote", """<blockquote style="border: none !important;padding: 0 !important;text-align: justify !important;">""")
    paper = paper.replace("close_blockquote", "</blockquote>")

    try:
        if url["web_info"]["Email_replace"] != '':
            match = re.findall(r'[\w\.-]+@[\w\.-]+', paper)
            email = url["web_info"]["Email_replace"]
            for i in match:
                paper = paper.replace(i, email)

        if len(url["web_info"]["Text_replace_doc"].keys()) > 0:
            for i in url["web_info"]["Text_replace_doc"].keys():
                paper = paper.replace(i, url["web_info"]["Text_replace_doc"][i])
    except Exception as e:
        print(e)
        pass

    content = {
        "user": url,
        "title": article.title,
        "content": str(paper),
        "category": url["category"],
        "url_img": thumb,
        "src_img": src_img,
        "slug": self_url

    }
    return content


def rest_image_url(website, user, password, url_img, src_img):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
        # This is another valid field
    }
    new_id = None
    if url_img is None:
        return new_id
    else:
        try:
            path_files = url_img.split("/")[-1].split("?")[0]
            with requests.get(url_img, stream=True, allow_redirects=False, verify=False, timeout=50,
                              headers=headers) as response:
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    image = image.resize((900, 603))
                    output = io.BytesIO()
                    credentials = user + ':' + password
                    token = base64.b64encode(credentials.encode())
                    if "JPG" in path_files.upper():
                        image.save(output, format='JPEG', optimize=True, quality=30)
                        headers = {'Authorization': 'Basic ' + token.decode('utf-8'), 'Content-Type': 'image/jpeg',
                                   'Content-Disposition': 'attachment; filename=%s' % path_files}

                    elif "PNG" in path_files.upper():
                        image.save(output, format='PNG', optimize=True, quality=30)
                        headers = {'Authorization': 'Basic ' + token.decode('utf-8'), 'Content-Type': 'image/png',
                                   'Content-Disposition': 'attachment; filename=%s' % path_files,
                                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
                    image = output.getvalue()

                    with requests.post(website,
                                       data=image,
                                       headers=headers, timeout=10, verify=False) as repsonse_post:
                        res = repsonse_post.json(encoding="utf-8")
                        new_id = res.get('id')
                        return new_id
        except Exception as e:
            print(e)
            return None


def replace_nth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string, re.IGNORECASE)][n - 1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    return before + after


def import_content(content, keyword_object):
    keyword = keyword_object["keyword"]
    anchor_text = keyword_object["anchortext"]
    base_url = keyword_object["baseUrl"]

    cl = content['user']["web_info"]
    website = cl["WebsitePost"]
    websiteimg = cl["Website"] + "/wp-json/wp/v2/media"
    user = cl["UserWP"]
    password = cl["PasswordWP"]

    idthump = rest_image_url(websiteimg, user, password, content['url_img'], content["src_img"]) or None
    if idthump is None:
        idthump = content['user']["web_info"]["imageid"]
    content["content"] = content["content"].replace("Bất động sản", "")
    content["content"] = content["content"].replace("bất động sản", "")
    # find keyword and replace anchor text for test
    # replace here

    # replace_anchortext(anchor_text, base_url, content, keyword)
    # content["content"] = content.get("content").replace(str(keyword["keyword"]), anchor_link, 1)
    anchor_link = f"""<a href='{base_url}'>{anchor_text}</a>"""
    print(content["content"].find("replace__anchor_link"))
    if content["content"].find("replace__anchor_link") != -1:
        content["content"] = content["content"].replace("replace__anchor_link", anchor_link, 1)
    else:
        raise "not found keyword"

    credentials = user + ':' + password
    token = base64.b64encode(credentials.encode())
    header = {'Authorization': 'Basic ' + token.decode('utf-8'), 'Content-Type': 'application/json',
              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    # todo: temporary delete category
    post = {
        'status': 'publish',
        "title": content["title"],
        "content": content["content"],
        'categories': content["category"],
        'featured_media': int(idthump),
        'slug': content['slug']
    }

    print("post artical")
    with requests.post(website, headers=header, json=post, verify=False) as response:
        res = response.status_code
        # try:
        # campaign = campaign_root.find_one({"_id": ObjectId(content['user']["campaign"]["_id"])})
        # if campaign.get("isautocomment") is True and len(campaign["listlinkyoutube"]) > 0:
        #     response_body = response.json(encoding="utf-8")
        #     comment_queue.insert_one({
        #         "id": response_body.get("id"),
        #         "guid": response_body.get("guid"),
        #         "campaign_id": content['user']["campaign"]["_id"],
        #         "status": response.status_code,
        #         "created_at": datetime.datetime.now()
        #     })
        # except Exception as e:
        #     print(e)
    if res is not None:
        print(res)
        print(post["slug"])
        # url = content['user']["campaign"]
        # if not url["Top10url"]:
        #     url["Top10url"] = [
        #         {"link": content['user']["web_info"]["Website"] + "/" + post['slug'], "name": content["title"]}]
        # elif len(url["Top10url"]) < 10:
        #     url["Top10url"].append(
        #         {"link": content['user']["web_info"]["Website"] + "/" + post['slug'], "name": content["title"]})
        # else:
        #     url["Top10url"] = [{"link": content['user']["web_info"]["Website"] + "/" + post['slug'],
        #                         "name": content["title"]}] + url["Top10url"][1:10]
        #
        # campaign_root.update_one({"_id": ObjectId(content['user']["campaign"]["_id"])},
        #                          {"$set": {"Top10url": url["Top10url"]}})
        # keywords.update_one(
        #     {"_id": ObjectId(content['user']["keyword"]["_id"])},
        #     {"$set": {"status": "done", "link": content['user']["web_info"]["Website"] + "/" + post['slug']}})
        #     todo: update status of keyword
        keyword_object["status"] = "success"
        keyword_object["post_url"] = f'{keyword_object["web_info"]["Website"]}/{post["slug"]}'
        mlink_report_posts.insert_one(keyword_object)
    return True


def get_contents(article, keyword_object):
    content_process = process_content(article, keyword_object)
    print("url data: ", keyword_object)

    content = import_content(content_process, keyword_object)
    print("-----------------------------------------------------------------------------------------------------------")
    return content
