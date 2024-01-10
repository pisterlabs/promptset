import datetime

from newspaper import Config
from sys import prefix
from pymongo import MongoClient
import time
import html.parser
import requests
from requests.models import HTTPBasicAuth
from Settings import *
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from SpinService import *
import json
from unidecode import unidecode
from Title_fix import *
from aiohttp import request
import requests
import json
import base64
import html
import html.parser
from Settings import *
from bson import ObjectId
from pymongo import MongoClient
import time
from PIL import Image
import io
from unidecode import unidecode
from aiohttp import request
from extract import ContentExtractor
from lxml.html import tostring
import re
import openai


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
campaign_root = MongoClient(CONNECTION_STRING_MGA1).campaigns.data
comment_queue = MongoClient(CONNECTION_STRING_MGA1).campaigns.comment_queue
keywords = MongoClient(CONNECTION_STRING_MGA1).keywords

contentExtractor = ContentExtractor(config)
from bs4 import BeautifulSoup


def replace_attr(soup, from_attr: str, to_attr: str):
    if from_attr in str(soup):
        soup[to_attr] = soup[from_attr]
        del soup[from_attr]

        return soup
    else:
        return soup


def process_content(article, url):
    # print(url)
    article.article_html = str(html.unescape(article.article_html))

    # # article.article_html = tostring(article.top_node,encoding="unicode")
    soup = BeautifulSoup(article.article_html, 'html.parser')
    self_url = str(no_accent_vietnamese(url["keyword"]["Keyword"].replace("\r", ""))) + ' ' + \
               str(time.time()).split(".")[0]
    self_url = self_url.replace(" ", "-")
    self_url = self_url.replace("--", "-")
    self_url = self_url.replace(".", "")

    self_url = self_url.replace("\n", "")
    domain = urlparse(url["link"]).netloc
    img = soup.find_all("img")
    print(f"length img: {len(img)}")
    src_img = []
    pre_link = None
    for i in img:
        # try:
        #     i.replace_with(replace_attr(i,'data-src', 'src'))
        #     i.replace_with(replace_attr(i,'data-lazy-src', 'src'))
        #     i.replace_with(replace_attr(i,'lazy-src', 'src'))
        #     i.replace_with(replace_attr(i,'data-srcset', 'srcset'))
        #     i.replace_with(replace_attr(i,'data-lazy-srcset', 'srcset'))
        #     i.replace_with(replace_attr(i,'lazy-srcset', 'srcset'))
        #     i.replace_with(replace_attr(i,'data-original', 'src'))
        # except Exception as e:
        #     print(str(e))

        # try:
        #     a = re.findall("lazy.*=\".*\"",str(i))
        #     if len(a)>0:
        #         for i in a:
        #             hhh= i.split(" ")[0].split("=")[-1]
        #             if ".JPG" in hhh.upper() or ".PNG" in hhh.upper():
        #                 i["src"] = hhh
        #                 print(hhh)
        #                 break
        # except Exception as e:
        #     print(str(e))

        # if pre_link!= None or str(pre_link) == "br" and i!=None:
        #     try:
        #         if i.has_attr("src") and pre_link.has_attr("src") and i.has_attr("alt") and pre_link.has_attr("alt"):
        #             if pre_link["src"] == i["src"] or pre_link["alt"] == i["alt"]:
        #                 if "base64" in pre_link["src"]:
        #                     aa = soup.new_tag("br")
        #                     pre_link.replace_with(aa)
        #                 else:
        #                     aa = soup.new_tag("br")
        #                     i.replace_with(aa)
        #                     pre_link = i
        #                     continue
        #     except:
        #         pass
        # pre_link = i
        # i['style'] ="width:100%"
        print(i)
        try:
            if i.has_attr("src"):
                if i["src"] in src_img:
                    i.decompose()
                else:
                    if "http" in i["src"]:
                        src_img.append(i["src"])
                    elif i["src"] == "":
                        i.decompose()
                    elif i["src"][:2] == "//" and "." in i["src"][3:].split("/")[0] and "jpg" not in \
                            i["src"][3:].split("/")[0] and "png" not in i["src"][3:].split("/")[0]:
                        i["src"] = "http:" + i["src"]
                        src_img.append(i["src"])
                    else:
                        i["src"] = "http://" + domain + i["src"]
                        src_img.append(i["src"])

            elif i.has_attr("srcset"):
                if i["srcset"] in src_img:
                    i.decompose()
                else:
                    if "http" in i["srcset"]:
                        src_img.append(i["srcset"])
                    elif i["srcset"] == "":
                        i.decompose()
                    elif i["srcset"][:2] == "//" and "." in i["srcset"][3:].split("/")[0] and "jpg" not in \
                            i["srcset"][3:].split("/")[0] and "png" not in i["srcset"][3:].split("/")[0]:
                        i["srcset"] = "http" + i["srcset"]
                        src_img.append(i["srcset"])
                    else:
                        i["srcset"] = "http://" + domain + i["srcset"]
                        src_img.append(i["srcset"])
        except Exception as e:
            print(e)

    thumb = None
    print(f"src_img {src_img}")
    if len(src_img) > 0:
        try:
            for iii in range(5):
                thumb = random.choice(src_img)
                if ".PNG" in thumb or ".JPG" in thumb:
                    break
        except:
            pass
    # print(src_img)
    if url["campaign"]["Top10url"] != None and url["campaign"]["Top10url"] != []:
        if len(url["campaign"]["Top10url"]) > 0:
            internal_link_total = random.choice(url["campaign"]["Top10url"])
            internal_link = internal_link_total["link"]
            internal_link_title = internal_link_total["name"]
            internal_link_total2 = random.choice(url["campaign"]["Top10url"])
            internal_link2 = internal_link_total2["link"]
            internal_link_title2 = internal_link_total2["name"]
    else:
        internal_link = None
        internal_link_title = None
        internal_link2 = None
        internal_link_title2 = None

    if url["campaign"]["CategoryId"] != None and url["campaign"]["CategoryName"] != None and url["campaign"][
        "CategoryLink"] != None:
        # print(acate_name)
        cate_name = url["campaign"]["CategoryName"]
        cate_link = url["campaign"]["CategoryLink"]
    else:
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
    thep = []
    for i in thepp:
        if i not in heading_p:
            thep.append(i)

    if len(url["campaign"]["Top10url"]) > 0:
        if url["campaign"]["language"] == "vi":
            if internal_link and internal_link_title:
                internal_link_p_tag1 = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Xem thêm: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> </p></div>'.format(
                    internal_link, internal_link_title, internal_link_title)
                internal_link_p_tag1 = BeautifulSoup(internal_link_p_tag1, "html.parser")
                try:
                    thep[int(len(thep) / 2)].append(internal_link_p_tag1)
                except:
                    pass

            self_link_p_tag = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Bạn đang đọc: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> </p></div>'.format(
                url["web_info"]["Website"] + '/' + self_url, article.title, article.title)
            if internal_link2 and internal_link_title2:
                internal_link_p_tag2 = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Xem thêm: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a></p></div>'.format(
                    internal_link2, internal_link_title2, internal_link_title2)
                internal_link_p_tag2 = BeautifulSoup(internal_link_p_tag2, "html.parser")
                try:
                    thep[len(thep) - 4].append(internal_link_p_tag2)
                except:
                    pass

            self_link_p_tag = BeautifulSoup(self_link_p_tag, "html.parser")
            try:
                thep[min(len(thep), 3)].append(self_link_p_tag)
            except:
                pass
        else:
            if internal_link and internal_link_title:
                internal_link_p_tag1 = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Read more : <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> </p></div>'.format(
                    internal_link, internal_link_title, internal_link_title)
                internal_link_p_tag1 = BeautifulSoup(internal_link_p_tag1, "html.parser")
                try:
                    thep[int(len(thep) / 2)].append(internal_link_p_tag1)
                except:
                    pass

            self_link_p_tag = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Reading: <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a> </p></div>'.format(
                url["web_info"]["Website"] + '/' + self_url, article.title, article.title)
            if internal_link2 and internal_link_title2:
                internal_link_p_tag2 = '<div style="margin-bottom:15px;margin-top:15px;"><p style="padding: 20px; background: #eaf0ff;">Read more : <a target="_blank" href="{}" rel="bookmark" title="{}">{}</a></p></div>'.format(
                    internal_link2, internal_link_title2, internal_link_title2)
                internal_link_p_tag2 = BeautifulSoup(internal_link_p_tag2, "html.parser")
                try:
                    thep[len(thep) - 4].append(internal_link_p_tag2)
                except:
                    pass

            self_link_p_tag = BeautifulSoup(self_link_p_tag, "html.parser")
            try:
                thep[min(len(thep), 3)].append(self_link_p_tag)
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
    listp = [{"ptag": m, "keywords": url["keyword"]["Keyword"], "language": url["campaign"]["language"]} for m in
             paper.find_all("p")]
    resultp = []
    for p_tag in listp:
        if p_tag["language"] == "vi":
            resultp.append(spinService.spin_paragraph(p_tag["ptag"], p_tag["keywords"], url["web_info"]["UserId"]))
        else:
            resultp.append(spinService.spin_paragraph_en(p_tag["ptag"], p_tag["keywords"], url["web_info"]["UserId"]))

    for index in range(0, len(resultp)):
        if 3 == index:
            # soup = BeautifulSoup(f"open_blockquote{str(resultp[index])}", 'html.parser')
            resultp[index] = soup
        
        if index == len(resultp) - 3:
            # soup = BeautifulSoup(f"{str(resultp[index])}close_blockquote", 'html.parser')
            resultp[index] = soup

    for k1, k2 in zip(listp, resultp):
        k1["ptag"].replace_with(k2)
        
    paper = str(paper)
    paper = paper.replace("&lt;", "<")
    paper = paper.replace("&gt;", ">")
    paper = paper.replace(" . ", ". ")
    paper = paper.replace(" , ", ", ")
    # paper = paper.replace("open_blockquote", """<blockquote style="border: none !important;padding: 0 !important;text-align: justify !important;">""")
    # paper = paper.replace("close_blockquote", "</blockquote>")
    try:
        if url["web_info"]["Email_replace"] != '':
            match = re.findall(r'[\w\.-]+@[\w\.-]+', paper)
            email = url["web_info"]["Email_replace"]
            for i in match:
                paper = paper.replace(i, email)

        if len(url["web_info"]["Text_replace_doc"].keys()) > 0:
            for i in url["web_info"]["Text_replace_doc"].keys():
                paper = paper.replace(i, url["web_info"]["Text_replace_doc"][i])
    except:
        pass
    content = {
        "user": url,
        "title": article.title,
        "content": str(paper),
        "category": url["campaign"]["CategoryId"],
        "url_img": thumb,
        "src_img": src_img,
        "slug": self_url

    }
    return content


def restImgUL(website, user, password, urlimg, src_img):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
        # This is another valid field
    }
    newID = None
    if urlimg == None:
        return newID
    else:
        try:
            path_files = urlimg.split("/")[-1].split("?")[0]
            with requests.get(urlimg, stream=True, allow_redirects=False, verify=False, timeout=50,
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
                                       headers=headers, timeout=10, verify=False) as response:
                        res = response.json()
                        newID = res.get('id')
                        return newID
        except Exception as e:
            print(str(e))
            return None


def importcontent(content):
    # cl = await clientt.user["userdatabase"].find_one({'_id':ObjectId(content['UserId'])})
    cl = content['user']["web_info"]
    website = cl["WebsitePost"]
    websiteimg = cl["Website"] + "/wp-json/wp/v2/media"
    user = cl["UserWP"]
    password = cl["PasswordWP"]
    idthump = None

    idthump = restImgUL(websiteimg, user, password, content['url_img'], content["src_img"])
    if idthump == None:
        idthump = content['user']["web_info"]["imageid"]
    # if len(a_link) == len(content['src_img']):
    #     for i,j in zip(content['src_img'],a_link):
    #         content["content"] = content["content"].replace(i,j)
    content["content"] = content["content"].replace("Bất động sản", "")
    content["content"] = content["content"].replace("bất động sản", "")
    credentials = user + ':' + password
    token = base64.b64encode(credentials.encode())
    header = {'Authorization': 'Basic ' + token.decode('utf-8'), 'Content-Type': 'application/json',
              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    post = {
        'status': 'publish',
        "title": content["title"],
        "content": content["content"],
        'categories': content["category"],
        'featured_media': int(idthump),
        'slug': content['slug']
    }

    with requests.post(website, headers=header, json=post, verify=False) as response:
        res = response.status_code
        try:
            campaign = campaign_root.find_one({"_id": ObjectId(content['user']["campaign"]["_id"])})
            if campaign.get("isautocomment") is True and len(campaign["listlinkyoutube"]) > 0:
                response_body = response.json(encoding="utf-8")
                comment_queue.insert_one({
                    "id": response_body.get("id"),
                    "guid": response_body.get("guid"),
                    "campaign_id": content['user']["campaign"]["_id"],
                    "status": response.status_code,
                    "created_at": datetime.datetime.now()
                })
        except Exception as e:
            print(str(e))
    if res != None:
        print(res)
        print(post["slug"])
        url = content['user']["campaign"]
        if url["Top10url"] == None or url["Top10url"] == []:
            url["Top10url"] = [
                {"link": content['user']["web_info"]["Website"] + "/" + post['slug'], "name": content["title"]}]
        elif len(url["Top10url"]) < 10:
            url["Top10url"].append(
                {"link": content['user']["web_info"]["Website"] + "/" + post['slug'], "name": content["title"]})
        else:
            url["Top10url"] = [{"link": content['user']["web_info"]["Website"] + "/" + post['slug'],
                                "name": content["title"]}] + url["Top10url"][1:10]

        campaign_root.update_one({"_id": ObjectId(content['user']["campaign"]["_id"])},
                                 {"$set": {"Top10url": url["Top10url"]}})
        keywords[content['user']['campaign']["WebsiteId"]].update_one(
            {"_id": ObjectId(content['user']["keyword"]["_id"])},
            {"$set": {"status": "done", "link": content['user']["web_info"]["Website"] + "/" + post['slug']}})
    return True


def ImportContents(article, url):
    dataprocess = process_content(article, url)
    import_content = importcontent(dataprocess)

    return import_content
