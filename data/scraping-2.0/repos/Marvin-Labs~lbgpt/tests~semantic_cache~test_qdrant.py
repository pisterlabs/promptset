# -*- coding: utf-8 -*-

import random
import string
import time
import uuid

import numpy
import pytest
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client.http.models import PointStruct
from sklearn.metrics.pairwise import cosine_similarity

from lbgpt.semantic_cache import QdrantSemanticCache

# for qdrant we want to get truly random names. However, it may be that the random seed is set somewhere else,
# so we have to create an instance of random.Random with a new seed here.
rng = random.Random(time.time())


TEST_EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="bert-base-uncased")


@pytest.fixture()
def messages() -> list[dict[str, str]]:
    raw = [
        {
            "role": "user",
            "content": "Apple’s board of directors has declared a cash dividend of $0.63 per share of the Company’s common stock. The dividend is payable on February 15, 2018 to shareholders of record as of the close of business on February 12, 2018.\n",
        },
        {
            "role": "user",
            "content": "Apple will provide live streaming of its Q1 2018 financial results conference call beginning at 2:00 p.m. PST on February 1, 2018 at www.apple.com/investor/earnings-call/. This webcast will also be available for replay for approximately two weeks thereafter.\n",
        },
        {
            "role": "user",
            "content": "This press release contains forward-looking statements, within the meaning of the Private Securities Litigation Reform Act of 1995. These forward-looking statements include without limitation those about the Company’s estimated revenue, gross margin, operating expenses, other income/(expense), tax rate, and plans for return of capital. These statements involve risks and uncertainties, and actual results may differ. Risks and uncertainties include without limitation: the effect of global and regional economic conditions on the Company's business, including effects on purchasing decisions by consumers and businesses; the ability of the Company to compete in markets that are highly competitive and subject to rapid technological change; the ability of the Company to manage frequent product introductions and transitions, including delivering to the marketplace, and stimulating customer demand for, new products, services and technological innovations on a timely basis; the effect that product introductions and transitions, changes in product pricing and product mix, and increases in component and other costs could have on the Company’s gross margin; the dependency of the Company on the performance of distributors of the Company's products, including cellular network carriers and other resellers; the inventory and other asset risks associated with the Company’s need to order, or commit to order, product components in advance of customer orders; the continued availability on acceptable terms, or at all, of certain components, services and new technologies essential to the Company's business, including components and technologies that may only be available from sole or limited sources; the dependency of the Company on manufacturing and logistics services provided by third parties, many of which are located outside of the U.S. and which may affect the quality, quantity or cost of products manufactured or services rendered to the Company; the effect of product and service quality problems on the Company’s financial performance and reputation; the dependency of the Company on third-party intellectual property and digital content, which may not be available to the Company on commercially reasonable terms or at all; the dependency of the Company on support from third-party software developers to develop and maintain software applications and services for the Company’s products; the impact of unfavorable legal proceedings, such as a potential finding that the Company has infringed on the intellectual property rights of others; the impact of changes to laws and regulations that affect the Company’s activities, including the Company’s ability to offer products or services to customers in different regions; the ability of the Company to manage risks associated with its international activities, including complying with laws and regulations affecting the Company’s international operations; the ability of the Company to manage risks associated with the Company’s retail stores; the ability of the Company to manage risks associated with the Company’s investments in new business strategies and acquisitions; the impact on the Company's business and reputation from information technology system failures, network disruptions or losses or unauthorized access to, or release of, confidential information; the ability of the Company to comply with laws and regulations regarding data protection; the continued service and availability of key executives and employees; war, terrorism, public health issues, natural disasters, and other business interruptions that could disrupt supply or delivery of, or demand for, the Company’s products; financial risks, including risks relating to currency fluctuations, credit risks and fluctuations in the market value of the Company’s investment portfolio; and changes in tax rates and exposure to additional tax liabilities. More information on these risks and other potential factors that could affect the Company’s financial results is included in the Company’s filings with the SEC, including in the “Risk Factors” and “Management’s Discussion and Analysis of Financial Condition and Results of Operations” sections of the Company’s most recently filed periodic reports on Form 10-K and Form 10-Q and subsequent filings. The Company assumes no obligation to update any forward-looking statements or information, which speak as of their respective dates.\n",
        },
        {
            "role": "user",
            "content": "NOTE TO EDITORS: For additional information visit Apple Newsroom (www.apple.com/newsroom), or call Apple’s Media Helpline at (408) 974-2042.\n",
        },
        {
            "role": "user",
            "content": "© 2018 Apple Inc. All rights reserved. Apple and the Apple logo are trademarks of Apple. Other company and product names may be trademarks of their respective owners.\n",
        },
        {
            "role": "user",
            "content": "  Three Months Ended December 30, 2017 December 31, 2016 Cash and cash equivalents, beginning of the period $ 20,289 $ 20,484 Operating activities: Net income 20,065 17,891 Adjustments to reconcile net income to cash generated by operating activities: Depreciation and amortization 2,745 2,987 Share-based compensation expense 1,296 1,256 Deferred income tax expense/(benefit) (33,737 ) 1,452 Other (11 ) (274 ) Changes in operating assets and liabilities: Accounts receivable, net (5,570 ) 1,697 Inventories 434 (580 ) Vendor non-trade receivables (9,660 ) (375 ) Other current and non-current assets (197 ) (1,446 ) Accounts payable 14,588 2,460 Deferred revenue 791 42 Other current and non-current liabilities 37,549 2,124 Cash generated by operating activities 28,293 27,234 Investing activities: Purchases of marketable securities (41,272 ) (54,272 ) Proceeds from maturities of marketable securities 14,048 6,525 Proceeds from sales of marketable securities 16,801 32,166 Payments made in connection with business acquisitions, net (173 ) (17 ) Payments for acquisition of property, plant and equipment (2,810 ) (3,334 ) Payments for acquisition of intangible assets (154 ) (86 ) Payments for strategic investments, net (94 ) — Other 64 (104 ) Cash used in investing activities (13,590 ) (19,122 ) Financing activities: Payments for taxes related to net share settlement of equity awards (1,038 ) (629 ) Payments for dividends and dividend equivalents (3,339 ) (3,130 ) Repurchases of common stock (10,095 ) (10,851 ) Proceeds from issuance of term debt, net 6,969 — Change in commercial paper, net 2 2,385 Cash used in financing activities (7,501 ) (12,225 ) Increase/(Decrease) in cash and cash equivalents 7,202 (4,113 ) Cash and cash equivalents, end of the period $ 27,491 $ 16,371 Supplemental cash flow disclosure: Cash paid for income taxes, net $ 3,551 $ 3,510 Cash paid for interest $ 623 $ 497\n",
        },
        {
            "role": "user",
            "content": 'This section and other parts of this Quarterly Report on Form 10-Q ("Form 10-Q") contain forward-looking statements, within the meaning of the Private Securities Litigation Reform Act of 1995, that involve risks and uncertainties. Forward-looking statements provide current expectations of future events based on certain assumptions and include any statement that does not directly relate to any historical or current fact. Forward-looking statements can also be identified by words such as "future," "anticipates," "believes," "estimates," "expects," "intends," "plans," "predicts," "will," "would," "could," "can," "may," and similar terms. Forward-looking statements are not guarantees of future performance and the Company\'s actual results may differ significantly from the results discussed in the forward-looking statements. Factors that might cause such differences include, but are not limited to, those discussed in Part II, Item 1A of this Form 10-Q under the heading "Risk Factors," which are incorporated herein by reference. The following discussion should be read in conjunction with the Company\'s Annual Report on Form 10-K for the year ended September 30, 2017 (the "2017 Form 10-K") filed with the U.S. Securities and Exchange Commission (the "SEC") and the condensed consolidated financial statements and notes thereto included in Part I, Item 1 of this Form 10-Q. All information presented herein is based on the Company\'s fiscal calendar. Unless otherwise stated, references to particular years, quarters, months or periods refer to the Company\'s fiscal years ended in September and the associated quarters, months and periods of those fiscal years. Each of the terms the "Company" and "Apple" as used herein refers collectively to Apple Inc. and its wholly-owned subsidiaries, unless otherwise stated. The Company assumes no obligation to revise or update any forward-looking statements for any reason, except as required by law.\n',
        },
        {
            "role": "user",
            "content": "On November 13, 2017, the Board adopted this amended and restated Non-Employee Director Stock Plan (formerly known as the 1997 Director Stock Option Plan and the 1997 Director Stock Plan, and, as renamed, the “Plan”), subject to approval by the Company’s shareholders at the Annual Meeting on February 13, 2018. For the terms and conditions of the Plan applicable to an Award, refer to the version of the Plan in effect as of the date such Award was granted.\n",
        },
        {
            "role": "user",
            "content": "1. PURPOSES. The purposes of the Plan are to retain the services of qualified individuals who are not employees of the Company to serve as members of the Board and to secure for the Company the benefits of the incentives inherent in increased Common Stock ownership by such individuals by granting such individuals Awards in respect of Shares.\n",
        },
    ]

    return raw


@pytest.fixture()
def qdrant_cache(messages) -> QdrantSemanticCache:
    cache = QdrantSemanticCache(
        embedding_model=TEST_EMBEDDING_MODEL,
        cosine_similarity_threshold=0.95,
        host="localhost",
        port=6333,
        collection_name="".join(rng.choice(string.ascii_letters) for _ in range(20)),
    )

    cache._sync_qdrant_client.upsert(
        collection_name=cache.collection_name,
        wait=False,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=cache.embed_messages([message]),
                payload={"message": message},
            )
            for message in messages
        ],
    )

    return cache


@pytest.mark.asyncio
async def test_exact(qdrant_cache, messages):
    message = messages[0]
    res = await qdrant_cache.qdrant_client.search(
        collection_name=qdrant_cache.collection_name,
        query_vector=qdrant_cache.embed_messages([message]),
        with_payload=True,
        with_vectors=True,
        limit=1,
    )

    assert res[0].payload["message"] == message
    assert res[0].score >= 1.0


@pytest.mark.asyncio
async def test_close(qdrant_cache, messages):
    message = {
        "role": "user",
        "content": "Apple’s board of directors has declared a cash dividend of $0.53 per share of the Company’s common stock. The dividend is payable on March 12, 2019 to shareholders of record as of the close of business on March 10, 2019.",
    }

    embedded = qdrant_cache.embed_messages([message])

    res = await qdrant_cache.qdrant_client.search(
        collection_name=qdrant_cache.collection_name,
        query_vector=embedded,
        with_payload=True,
        with_vectors=True,
        limit=1,
    )

    res_message = res[0].payload["message"]
    assert res_message == messages[0]

    res_embedded = qdrant_cache.embed_messages([res_message])

    cs = cosine_similarity([embedded], [res_embedded])

    assert numpy.isclose(cs[0][0], res[0].score)


@pytest.mark.asyncio
async def test_far(qdrant_cache, messages):
    message = {
        "role": "user",
        "content": "The tolerance values are positive, typically very small numbers. The relative difference (rtol * abs(b)) and the absolute difference atol are added together to compare against the absolute difference between a and b.",
    }

    embedded = qdrant_cache.embed_messages([message])

    res = await qdrant_cache.qdrant_client.search(
        collection_name=qdrant_cache.collection_name,
        query_vector=embedded,
        with_payload=True,
        with_vectors=True,
        limit=1,
    )

    res_message = res[0].payload["message"]
    res_embedded = qdrant_cache.embed_messages([res_message])

    cs = cosine_similarity([embedded], [res_embedded])

    assert numpy.isclose(cs[0][0], res[0].score)
