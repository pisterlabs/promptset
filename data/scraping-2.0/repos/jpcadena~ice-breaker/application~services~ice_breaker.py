"""
A module for icebreaker in the application.services package.
"""
from typing import Any

from langchain.chains import LLMChain

from application.agents.linkedin_lookup_agent import (
    lookup as linkedin_lookup_agent,
)
from application.agents.twitter_lookup_agent import (
    lookup as twitter_lookup_agent,
)
from application.chains.custom_chains import (
    get_ice_breaker_chain,
    get_interests_chain,
    get_summary_chain,
)
from application.schemas.output_parsers import (
    IceBreaker,
    Summary,
    TopicOfInterest,
    ice_breaker_parser,
    summary_parser,
    toi_parser,
)
from application.services.linkedin import scrape_linkedin_profile
from application.services.twitter import scrape_user_tweets


def ice_break_with(
    name: str
) -> tuple[Summary, IceBreaker, TopicOfInterest, str]:
    """
    Ice break with name from LLM service
    :param name: The name of the persona to search for
    :type name: str
    :return: The IceBreaker summary for the username
    :rtype: tuple[Summary, IceBreaker, TopicOfInterest, str]
    """
    linkedin_username: str = linkedin_lookup_agent(name)
    linkedin_data: dict[str, Any] = scrape_linkedin_profile(linkedin_username)
    twitter_username: str = twitter_lookup_agent(name)
    tweets: list[dict[str, str]] = scrape_user_tweets(twitter_username)
    summary_chain: LLMChain = get_summary_chain()
    summary_and_facts_run: str = summary_chain.run(
        information=linkedin_data, twitter_posts=tweets
    )
    summary_and_facts: Summary = summary_parser.parse(summary_and_facts_run)
    interests_chain: LLMChain = get_interests_chain()
    interests_run: str = interests_chain.run(
        information=linkedin_data, twitter_posts=tweets
    )
    interests: TopicOfInterest = toi_parser.parse(interests_run)
    ice_breaker_chain: LLMChain = get_ice_breaker_chain()
    ice_breakers_run: str = ice_breaker_chain.run(
        information=linkedin_data, twitter_posts=tweets
    )
    ice_breakers: IceBreaker = ice_breaker_parser.parse(ice_breakers_run)
    return (
        summary_and_facts,
        ice_breakers,
        interests,
        str(linkedin_data.get("profile_pic_url", "")),
    )
