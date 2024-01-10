""" importing necessary libs and methods """
import os
from celery import shared_task
from elasticsearch import Elasticsearch
from .web_scraper import WebScrapping, ApolloCompany
from .open_ai import LangChainAI


class ScrapeDataTask:
    """A scrape data class """

    def __init__(self, website_url):
        self.website_url = website_url
        self.es_search = Elasticsearch(os.environ.get('ELASTIC_DB_URL'),)

    def query_should_list(self, field_name, key_list):
        ''' queries to involve'''
        resp = []
        for k in key_list:
            resp.append({
                "match": {
                    field_name: k
                }
            })
        return resp

    def run(self):
        """the delayed task to scraped data"""
        # scraping all urls from the companies webpage
        company_urls = WebScrapping.get_url_website(self.website_url)
        # get domain domain-related pages and scrape the data
        company_related_urls = WebScrapping.domain_related_route(
            urls=company_urls,
            target_domain=self.website_url
        )
        # linkedin_data = WebScrapping.apolo_request_sesion(self.linkedin_url)
        company_data_content = WebScrapping.get_all_content(
            company_related_urls
        )

        # using apollo search to get companies information
        apolo = ApolloCompany(self.website_url)
        organization_data = apolo.get_data()
        # using langchain ai to analyze the company data
        lang_chain = LangChainAI()
        ai_analysis, campaign_data = lang_chain.get_ai_data(
            company_data_content,
            organization_data
        )

        resp_data = None
        job_titles = campaign_data.get('Job Title', [])
        countries = campaign_data.get('Countries', [])
        keywords = campaign_data.get('Keywords', [])

        # Check if job titles, countries
        # and keywords lists are not empty before constructing queries
        job_title_should = self.query_should_list(
            "job_title",
            job_titles
        ) if job_titles else []

        countries_should = self.query_should_list(
            "location_country",
            countries
        ) if countries else []

        keywords_should = self.query_should_list(
            "summary",
            keywords
        ) if keywords else []

        # Construct the Elasticsearch query
        # with conditions only if lists are not empty
        if job_title_should or countries_should or keywords_should:
            search_query = {
                "bool": {
                    "should": [
                        {"bool": {
                            "should": job_title_should
                        }},
                        {"bool": {
                            "must": countries_should
                        }},
                        {
                            "bool": {
                                "should": keywords_should
                            }
                        }
                    ]
                }
            }
        else:
            search_query = None

        resp_data = self.es_search.search(
            index="linked-in",
            request_timeout=60,
            query=search_query,
            from_=0, size=10)['hits']['hits']

        return {
            "ai_analysis": ai_analysis,
            "matched_users": resp_data if search_query else [],
            "campaign_data": campaign_data
        }


# Define a shared_task decorator for the run method
@shared_task
def scrape_data(website_url):
    '''a celery task to scrape and analyze company data'''
    task = ScrapeDataTask(website_url)
    return task.run()
