# fastapiとその他必要なライブラリをインポートします。
from fastapi import APIRouter, Depends
from typing import List
from config.index import get_db
from sqlalchemy.orm import Session
from services.crawling.dev_to.get_articles import GetArticlesInDevTo
from services.markdown.changer import ConvertUrlToMarkdown
from services.openai.gpt.model import GPTModelFacade
from models.index import Tag, Article
import openai
import re


# /api/dev-to というプレフィックスと"Dev-To-Crawling"というタグのついたAPIRouterを作成します。
dev_to_router = APIRouter(
    prefix="/api/dev-to",
    tags=["Dev-To-Crawling"]
)

# /latestエンドポイントを作成します。このエンドポイントは、dev.toから最新の記事を取得しスプレッドシートに保存します。
@dev_to_router.get("/latest", response_model=List[str])
async def get_dev_to_latest_articles(
    page: int = 10,
    db: Session = Depends(get_db),
):
    """
    引数
        page (int)： 検索する最新記事の数。デフォルトは10。
        db (セッション)： データベースとやりとりするためのセッションオブジェクト。

    戻り値
        リスト[str]： 成功メッセージを含むリスト。

    """
    # dev.toから最新記事を取得するインスタンスを作成します。
    dev_to_get_article_facade = GetArticlesInDevTo(f"https://dev.to/api/articles/latest{1 if page is None else page}")
    dev_to_get_article_facade.get_article()

    model = GPTModelFacade()

    count = 0
    for article in dev_to_get_article_facade.json:
        positive_reactions_count = article.get("positive_reactions_count", 0)
        if positive_reactions_count < 5:
            continue
        if count > 3:
            break

        # すでに記事が存在するか確認
        dev_to_id = article.get("id", "")
        my_article = db.query(Article).filter(Article.dev_to_id == dev_to_id).one_or_none()
        if not my_article is None:
            continue

        # URLに基づいてMarkdown形式のテキストを取得します。
        link = article.get("url", 0)
        convert_markdown = ConvertUrlToMarkdown.convert(link)

        try:
            answer = await model.listen_markdown_prompt(convert_markdown)
        except openai.error.APIConnectionError:
            answer = convert_markdown

        answer = answer.replace("```markdown", "")

        answer+=f"<br><br>こちらの記事はdev.toの良い記事を日本人向けに翻訳しています。<br>[{article.get('url', '')}]({article.get('url', '')})"

        title = article.get("title", "")
        match = re.search(r'#\s*(.*?) -', answer)
        if match:
            title = match.group(1)

        article_model = Article(
            story=answer.replace(" - DEVコミュニティ", "").replace(" - DEV Community", "").replace("フルスクリーンモードに入る", ""),
            description=article.get("description", ""),
            title=title,
            dev_to_id=article.get("id", ""),
            og_image_url=article.get("social_image", "https://thepracticaldev.s3.amazonaws.com/i/6hqmcjaxbgbon8ydw93z.png")
        )

        # タグのデータと記事のデータをDBに入れる
        for tag_name in article.get("tag_list", [""]):
            tag = db.query(Tag).filter(Tag.name == tag_name).one_or_none()
            if tag is None:
                tag = Tag(name=tag_name)
                db.add(tag)
                db.commit()
            article_model.tags.append(tag)
        db.add(article_model)
        db.commit()
        count+=1

    # 成功メッセージを返します。
    return [f'成功しました: 計{len(count)}件保存']


# /popular エンドポイントを作成します。使用は未定義で、まだ何も行いません。
@dev_to_router.get("/popular", response_model=List[str])
async def get_dev_to_popular_articles(
    rank: int,
    db: Session = Depends(get_db),
):
    """
    引数
        page (int)： 検索する最新記事の数。デフォルトは10。
        db (セッション)： データベースとやりとりするためのセッションオブジェクト。

    戻り値
        リスト[str]： 成功メッセージを含むリスト。

    """
    # このパラメータを使用すると、クライアントは過去数日間で最も人気のある記事を返すことができますN。
    # top返された記事の公開からの日数を示します。
    # このパラメータは と組み合わせて使用できますtag。
    dev_to_get_article_facade = GetArticlesInDevTo(
        f'https://dev.to/api/articles?top={1 if rank is None else rank}')
    dev_to_get_article_facade.get_article()

    model = GPTModelFacade()
    print(dev_to_get_article_facade.json)
    count = 0
    for article in dev_to_get_article_facade.json:
        positive_reactions_count = article.get("positive_reactions_count", 0)
        if positive_reactions_count < 5:
            continue
        if count > 3:
            break
        # すでに記事が存在するか確認
        dev_to_id = article.get("id", "")
        my_article = db.query(Article).filter(Article.dev_to_id == dev_to_id).one_or_none()
        if not my_article is None:
            continue

        # URLに基づいてMarkdown形式のテキストを取得します。
        link = article.get("url", 0)
        convert_markdown = ConvertUrlToMarkdown.convert(link)

        try:
            answer = await model.listen_markdown_prompt(convert_markdown)
        except openai.error.APIConnectionError:
            answer = convert_markdown

        answer = answer.replace("```markdown", "")
        answer+=f"<br><br>こちらの記事はdev.toの良い記事を日本人向けに翻訳しています。<br>[{article.get('url', '')}]({article.get('url', '')})"

        title = article.get("title", "")
        match = re.search(r'#\s*(.*?) -', answer)
        if match:
            title = match.group(1)

        article_model = Article(
            story=answer.replace(" - DEVコミュニティ", "").replace(" - DEV Community", "").replace("フルスクリーンモードに入る", ""),
            description=article.get("description", ""),
            title=title,
            dev_to_id=article.get("id", ""),
            og_image_url=article.get("social_image", "https://thepracticaldev.s3.amazonaws.com/i/6hqmcjaxbgbon8ydw93z.png")
        )
        # タグのデータと記事のデータをDBに入れる
        for tag_name in article.get("tag_list", [""]):
            tag = db.query(Tag).filter(Tag.name == tag_name).one_or_none()
            if tag is None:
                tag = Tag(name=tag_name)
                db.add(tag)
                db.commit()
            article_model.tags.append(tag)
        db.add(article_model)
        db.commit()
        count+=1

    # 成功メッセージを返します。
    return [f'成功しました{count}']


@dev_to_router.get("/tag/{tag}", response_model=List[str])
async def get_dev_to_popular_articles(
    tag: str,
    page: str,
    db: Session = Depends(get_db),
):
    """
    引数
        page (int)： 検索する最新記事の数。デフォルトは10。
        db (セッション)： データベースとやりとりするためのセッションオブジェクト。
        get_bearer_token（TokenDataSchema）： ベアラートークンのデータスキーマ。

    戻り値
        リスト[str]： 成功メッセージを含むリスト。

    """

    # dev.toからタグで検索して記事を取得するインスタンスを作成します。
    dev_to_get_article_facade = GetArticlesInDevTo(
        f'https://dev.to/api/articles?tag={tag}&page={1 if page is None else page}')
    dev_to_get_article_facade.get_article()

    model = GPTModelFacade()

    count = 0
    for article in dev_to_get_article_facade.json:
        positive_reactions_count = article.get("positive_reactions_count", 0)
        if positive_reactions_count < 5:
            continue
        if count > 3:
            break

        # すでに記事が存在するか確認
        dev_to_id = article.get("id", "")
        my_article = db.query(Article).filter(Article.dev_to_id == dev_to_id).one_or_none()
        if not my_article is None:
            continue

        # URLに基づいてMarkdown形式のテキストを取得します。
        link = article.get("url", 0)
        convert_markdown = ConvertUrlToMarkdown.convert(link)

        try:
            answer = await model.listen_markdown_prompt(convert_markdown)
        except openai.error.APIConnectionError:
            answer = convert_markdown

        answer = answer.replace("```markdown", "")
        answer+=f"<br><br>こちらの記事はdev.toの良い記事を日本人向けに翻訳しています。<br>[{article.get('url', '')}]({article.get('url', '')})"

        title = article.get("title", "")
        match = re.search(r'#\s*(.*?) -', answer)
        if match:
            title = match.group(1)

        article_model = Article(
            story=answer.replace(" - DEVコミュニティ", "").replace(" - DEV Community", "").replace("フルスクリーンモードに入る", ""),
            description=article.get("description", ""),
            title=title,
            dev_to_id=article.get("id", ""),
            og_image_url=article.get("social_image", "https://thepracticaldev.s3.amazonaws.com/i/6hqmcjaxbgbon8ydw93z.png")
        )
        # タグのデータと記事のデータをDBに入れる
        for tag_name in article.get("tag_list", [""]):
            tag = db.query(Tag).filter(Tag.name == tag_name).one_or_none()
            if tag is None:
                tag = Tag(name=tag_name)
                db.add(tag)
                db.commit()
            article_model.tags.append(tag)
        db.add(article_model)
        db.commit()
        count+=1

    # 成功メッセージを返します。
    return [f'成功しました: 計{count}件保存']
