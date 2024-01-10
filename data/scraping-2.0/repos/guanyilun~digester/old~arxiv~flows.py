from prefect import flow, get_run_logger
from typing import Optional


@flow(name="arxiv-metadata-ingest")
def ingest_metadata(
    db_url: str = "sqlite:////home/yilun.guan/.digester/arxiv/metadata.db",
    category: str = "astro-ph.CO",
    max_results: Optional[int] = 100
    ):
    import arxiv
    from sqlalchemy.orm import sessionmaker
    from lib import create_db, Article, Author, arxiv_url_to_id_and_ver  # Assuming the revised script is saved as arxiv_db.py

    logger = get_run_logger()

    # Set up the database
    engine = create_db(db_url)

    # Fetch articles from arXiv
    articles = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    # Store articles in the database
    Session = sessionmaker(bind=engine)
    session = Session()

    new_records = 0
    for result in articles.results():
        # check if article already exists
        arxiv_id, _ = arxiv_url_to_id_and_ver(result.entry_id)
        if session.query(Article).filter_by(id=arxiv_id).first():
            continue
        title = result.title
        submitted_date = result.published
        category = "astro-ph.CO"
        abstract = result.summary
        authors = [author.name for author in result.authors]

        logger.info(f"Adding article: {arxiv_id}")

        article = Article(
            id=arxiv_id,
            title=title,
            submitted_date=submitted_date,
            category=category,
            abstract=abstract,
            abstract_digested=False,
            source_file_digested=False
        )
        session.add(article)

        for author_name in authors:
            author = Author(name=author_name, article=article)
            session.add(author)
        new_records += 1

    session.commit()
    logger.info(f"Added {new_records} new records to the database.")


@flow(name="arxiv-abstract-embed")
def embed_abstract(
    limit: int = 1000,
    embedding_method: str = "huggingface",
    db_url: str = "sqlite:////home/yilun.guan/.digester/arxiv/metadata.db",
    chroma_db_path: str = "/home/yilun.guan/.digester/arxiv/chroma/"
) -> str:
    from langchain.vectorstores import Chroma
    from sqlalchemy.orm import sessionmaker
    from lib import create_db, Article

    logger = get_run_logger()

    # get abstract from database
    engine = create_db(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # get first `limit` articles that have not been digested
    articles = session.query(Article).filter_by(abstract_digested=False).limit(limit).all()

    if len(articles) == 0:
        logger.info("No new articles to digest")
        return

    # get embeddings
    if embedding_method == "huggingface":
        from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings()
    elif embedding_method == "openai":
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    else:
        raise ValueError(f"Unknown algorithm: {embedding_method}")
    logger.info("Using embedding method: " + embedding_method)
    
    # separate collection for different algorithmm
    collection_name = f"arxiv-abstract-{embedding_method}"
    logger.info("Using collection name: " + collection_name)
    vectordb = Chroma(
        collection_name=collection_name, 
        embedding_function=embeddings,
        persist_directory=chroma_db_path
    )

    # build a list of ab    stracts and associated ids and metadata
    abstracts = [article.abstract for article in articles]
    ids = [article.id for article in articles]

    for id_, abs_ in zip(ids, abstracts):
        vectordb.add_texts(texts=[abs_], ids=id_)
        logger.debug("Embedded:", id_)

    # update database
    for article in articles:
        article.abstract_digested = True
    session.commit()
    
    logger.info(f"Embedded {len(ids)} new abstracts to the database")

if __name__ == "__main__":
    embed_abstract(limit=100)