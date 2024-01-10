import os
import re
import openai
import backoff
from concurrent.futures import ThreadPoolExecutor

from django.utils import timezone

from scrapper.models import Video, BrandDeal, BlackList
from scrapper.utils import get_domain, print_exception
from scrapper.limits import MAX_OPENAI_WORKERS, MAX_VIDEOS_PER_CHANNEL, COUNTRIES
from scrapper.csvs.utils import accept_domain

openai.api_key = os.getenv("OPENAI_API_KEY")


PREVIOUS_VIDEOS_COUNT = 10
VIDEOS_BATCH_SIZE = 100

# Use GPT-3.5 and extract brand deal links
system_prompt = "You help extract brand deal or sponsored segment links"
user_prompt = """
Brand Deals are sponsored segments where a YouTuber shares endorses brands by sharing a discounted URL to the brands product or service.  
The following is a YouTube description. It may contain 0-6 brand deals.
Extract the section that contains the brand deal/deals. If there is no brand deal, return "".
Here's the description
'''
{description}
'''
"""


def on_backoff(details):
    print(f"{timezone.now()} Backing off {details['wait']} seconds")


# # 3,500 RPM - Paid users after 48 hours
# # https://platform.openai.com/docs/guides/rate-limits/overview
# @backoff.on_exception(backoff.expo, openai.error.RateLimitError, on_backoff=on_backoff)
# def extract_brand_deal_links(description):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt,
#             },
#             {
#                 "role": "user",
#                 "content": user_prompt.format(description=description),
#             },
#         ],
#         temperature=1,
#         max_tokens=1024,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#     )
#     brand_deals_section = response.choices[0]["message"]["content"]
#     # Extract urls from the section using regex
#     urls = re.findall(r"(?P<url>https?://[^\s]+)", brand_deals_section) or [""]
#     # Remove all "" strings
#     urls = [url for url in urls if url]
#     return urls


def extract_brand_deal_links(description):
    all_urls = re.findall(r"(?P<url>https?://[^\s]+)", description) or [""]
    all_domains = [get_domain(url) for url in all_urls]

    return_urls = []
    for url, domain in zip(all_urls, all_domains):
        if not accept_domain(domain):
            continue
        return_urls.append(url)

    return return_urls


def create_brand_deal_links():
    # Detailed videos which already don't have BrandDeal
    detailed_videos = Video.objects.filter(
        status=Video.DETAILED, brand_deals__isnull=True, channel__country__in=COUNTRIES
    ).order_by("-published_at")

    batches = [
        detailed_videos[i : i + VIDEOS_BATCH_SIZE]
        for i in range(0, len(detailed_videos), VIDEOS_BATCH_SIZE)
    ]
    for batch in batches:
        with ThreadPoolExecutor(max_workers=MAX_OPENAI_WORKERS) as executor:
            futures_dict = {}
            for video in batch:
                if (
                    video.channel.videos.filter(status=Video.FILTERED).count()
                    > MAX_VIDEOS_PER_CHANNEL
                ):
                    log_string = (
                        f"{timezone.now()} MAX VIDEOS PER CHANNEL {video.video_id}"
                    )
                    print(log_string)
                    print_exception(log_string)
                    continue
                # Optimization - if the next 10 videos of the channel don't have any brand deals, then skip this video
                channel = video.channel
                previous_videos = channel.videos.filter(
                    published_at__gte=video.published_at
                ).order_by("published_at")[:PREVIOUS_VIDEOS_COUNT]
                if previous_videos.count() == PREVIOUS_VIDEOS_COUNT:
                    brand_deals_count = BrandDeal.objects.filter(
                        video__in=previous_videos
                    ).count()
                    if brand_deals_count == 0:
                        video.status = Video.SKIPPED
                        video.save(update_fields=["status"])
                        log_string = f"{timezone.now()} SKIPPING {video.video_id} for OPTIMIZATION"
                        print(log_string)
                        continue

                # Extract brand deal links
                description = video.description
                # Find the first URL in the description using regex - http or https
                url_index = re.search(r"(?P<url>https?://[^\s]+)", description)
                if url_index is None:
                    continue
                url_index = url_index.start()
                # Limit the description to 2000 characters after the first URL
                description = description[0 : url_index + 2000]
                print(
                    f"{timezone.now()} Extracting brand deal links for {video.video_id}"
                )
                futures_dict[video] = executor.submit(
                    extract_brand_deal_links, description
                )

            for video, future in futures_dict.items():
                urls = future.result()
                # Remove duplicates
                urls = list(set(urls))
                # Filter the domains that are blacklisted
                urls = [
                    url
                    for url in urls
                    if not BlackList.objects.filter(domain=get_domain(url)).exists()
                ]
                print(timezone.now(), video.video_id, urls)
                for url in urls:
                    BrandDeal.objects.get_or_create(video=video, initial_url=url)
                # Update the video status
                video.status = Video.FILTERED
                video.save(update_fields=["status"])
    # detailed_videos.update(status=Video.FILTERED)
