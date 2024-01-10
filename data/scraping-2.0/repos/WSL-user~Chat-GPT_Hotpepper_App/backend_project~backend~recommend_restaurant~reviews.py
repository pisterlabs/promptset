from bs4 import BeautifulSoup
import re
import requests
from concurrent import futures
import openai
openai.api_key='YOUR OPENAI API KEY HERE'
#指定されたURLからレビューを取得
def parse(url):
    result={}
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "lxml")
    reviews=soup.find_all("div",{"class":"reportText"})
    links=soup.find_all("a",{"class":"arrowLink"})
    for review,link in zip(reviews,links):
        result["https://www.hotpepper.jp/"+link.get('href')]=review.text.replace("\n", "").replace("\u3000", "").replace("\xa0", "")
    return result

#全てのレビューを入手
def get_reviews(id:str , max_num:int)->list:
    #スクレイピングの準備
    url = 'https://www.hotpepper.jp/str'+id+'/report/'
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "lxml")

    #レビュー数を取得
    num=soup.find_all("p",{"class":"recommendReportNum"})
    pattern = r"\d+"
    matches = re.findall(pattern, num[0].text)
    num = int("".join(matches))

    #スクレイピングするURLを列挙
    paths=[]
    for i in range(min(-1*int(((-num+6)//5)),max_num-1)):
        url_="list_"+str(6+5*(i))
        url = 'https://www.hotpepper.jp/str'+id+'/report/'+url_+"/"
        paths.append(url)
    #並行処理でスクレイピング
    with futures.ThreadPoolExecutor() as executor:
        rets = list(executor.map(parse, paths))
    #スクレイピング結果を一つのリストに
    result={}
    reviews=soup.find_all("div",{"class":"reportText"})
    links=soup.find_all("a",{"class":"arrowLink"})
    for review,link in zip(reviews,links):
        result["https://www.hotpepper.jp/"+link.get('href')]=review.text.replace("\n", "").replace("\u3000", "").replace("\xa0", "")
    for ret in rets:
        result.update(ret)
    return result


def get_detail_all(reviews):
    paths=[]
    for k,v in reviews.items():
        if v[-1]=="…":
            paths.append(k)
    with futures.ThreadPoolExecutor() as executor:
        rets = list(executor.map(review_detail, paths))
    for k,v in rets:
        reviews[k]=v
    return reviews
def review_detail(url:str,):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "lxml")
    reviews=soup.find_all("div",{"class":"reportText"})[0].text.replace("\n", "").replace("\u3000", "").replace("\xa0", "")
    return url,reviews

#レビューをランク順&。で区切った時の文章数を制限。
def get_reviews_rank(id:str , max_num:int , mode:str , modify:bool , page=2)->list:
    reviews=get_reviews(id,max_num)
    all={}
    out={}
    safe={}
    if mode=="time":
        if modify:
            reviews=get_detail_all(reviews)
        for k ,v in reviews.items():
            if v[-1]=="…":
                all[k]=v
                out[k]=v
            else:
                all[k]=v
                safe[k]=v
        return  all,out,safe
    elif mode=="length":
        if modify:
            reviews=get_detail_all(reviews)
        for k ,v in sorted(reviews.items(), key=lambda item: len(item[1]),reverse=True):
            if v[-1]=="…":
                all[k]=v
                out[k]=v
            else:
                all[k]=v
                safe[k]=v
        return all,out,safe
    elif mode=="hybrid":
        #評価関数作るのは時間かかるので簡単な処理。
        """
        ①まず、スクレイピングしたページごとにグループを作る。
        グループ1:レビュー1~2ページ目
        グループ2:レビュー3~4ページ目
        グループ3:レビュー5~6ページ目
        ②その後、各グループで文字数順に並び替える。
        """
        reviews=get_detail_all(reviews)
        num=page*5
        for i in range(-1*(-1*len(reviews)//num)):
            dicts={k: reviews[k] for k in list(reviews.keys())[i*num:min((i+1)*num,len(reviews))]}
            for k ,v in sorted(dicts.items(), key=lambda item: len(item[1]),reverse=True):
                if v[-1]=="…":
                    all[k]=v
                    out[k]=v
                else:
                    all[k]=v
                    safe[k]=v
        return all,out,safe

def get_short_review(reviews,num_sentence=4):
    results=[]
    for url in reviews:
        review_list=reviews[url].split("。")[:num_sentence]
        
        results.append("。".join(review_list))
    return results
def limit_length(reviews,num=100):
    sums=0
    result=[]
    for review in reviews:
        result.append(review)
        sums+=len(reviews)
        if sums > num:
            return result
    return result

def review_responce(shop_id):
    reviews,out,safe=get_reviews_rank(shop_id,10000,"hybrid",True,100)
    reviews=get_short_review(reviews,4)
    reviews=limit_length(reviews,300)
    content="Please create one new review in Japanese from the following reviews.\n 「"
    for i,review in enumerate(reviews):
        content+="・"+review+"\n"
    content+="」"
    print(content)
    message=[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":content}
    ]
    ans=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message)["choices"][0]["message"]["content"]
    return ans