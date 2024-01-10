from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from opencopilot import ContextInput
from opencopilot import OpenCopilot

import extract_query_context
from data_API import get_financials
from data_API import get_historical_market_data
from data_API import get_info
from news_API import get_news


def main():
    copilot = OpenCopilot(
        copilot_name="Stock Analyst Copilot",
        llm="gpt-4",
        prompt_file="prompt.txt",
        log_level="debug"
    )

    def load_news(company_name: str) -> List[Document]:
        news = get_news(company_name)
        documents: List[Document] = []
        for n in news:
            scraped_document = Document(
                page_content=n["content"],
                metadata={
                    "source": n["url"],
                    "title": n["title"],
                    "description": n["description"],
                    "image_url": n["urlToImage"],
                    "time_published": n["publishedAt"]
                }
            )
            documents.append(scraped_document)
        return documents

    def load_info(ticker: str) -> List[Document]:
        try:
            info = get_info(ticker)
            documents: List[Document] = []
            scraped_document = Document(
                page_content=info,
                metadata={"source": f"https://finance.yahoo.com/quote/{ticker}"}
            )
            documents.append(scraped_document)
            return documents
        except Exception as e:
            print(f"Failed to load info for {ticker}, {e}")
            return []

    def load_historical_market_data(ticker: str) -> List[Document]:
        try:
            historical_market_data = get_historical_market_data(
                ticker,
                "1mo",  # ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
            )
            documents: List[Document] = []
            scraped_document = Document(
                page_content=historical_market_data,
                metadata={
                    "source": f"https://finance.yahoo.com/quote/{ticker}",
                    "title": f"Historical Market Data for {ticker}"
                }
            )
            documents.append(scraped_document)
            return documents
        except Exception as e:
            print(f"Failed to load historical market data for {ticker}, {e}")
            return []

    def load_financial_data(ticker: str) -> List[Document]:
        try:
            financial_data = get_financials(ticker)
            documents: List[Document] = []
            for f in financial_data:
                for key, value in f.items():
                    scraped_document = Document(
                        page_content=value,
                        metadata={
                            "source": f"https://finance.yahoo.com/quote/{ticker}",
                            "title": key
                        }
                    )
                    documents.append(scraped_document)
            return documents
        except Exception as e:
            print(f"Failed to financial info for {ticker}, {e}")
            return []

    @copilot.context_builder
    async def query_company_info(context_input: ContextInput) -> List[Document]:
        # TODO: use async stuff here
        extracted_company = extract_query_context.extract(context_input.message)
        if not extracted_company or len(extracted_company) != 2:
            return []
        docs = []
        ticker = extracted_company[1]
        if ticker != "0":
            docs.extend(load_historical_market_data(ticker))
            docs.extend(load_info(ticker))
            docs.extend(load_financial_data(ticker))
        if extracted_company[0] != "0":
            docs.extend(load_news(extracted_company[0]))
        return docs

    copilot()


if __name__ == '__main__':
    load_dotenv()
    main()
