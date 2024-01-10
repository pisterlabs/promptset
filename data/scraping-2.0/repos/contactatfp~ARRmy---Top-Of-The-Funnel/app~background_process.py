# import logging
#
# from flask_apscheduler import APScheduler
# import json
# from flask import current_app
# from app.rank_algo import rank_companies
# from main import app, db, cache, celery
# from app.models import Account
# from dotenv import load_dotenv
# import os
#
# scheduler = APScheduler()
# scheduler.init_app(app)
# scheduler.start()
#
# load_dotenv()
#
# openai_api_key = os.getenv("OPENAI_API_KEY")
#
#
# # with open('config.json') as f:
# #     config = json.load(f)
#
#
# @scheduler.task('interval', id='prospecting_task', days=30, start_date='2023-09-13 09:57:11')
# def prospecting():
#     from langchain.chat_models import ChatOpenAI
#     from langchain.prompts.chat import (
#         ChatPromptTemplate,
#         SystemMessagePromptTemplate,
#         HumanMessagePromptTemplate,
#     )
#     from app.prospecting import prospecting_overview, prospecting_products, prospecting_market, \
#         prospecting_recent_partnerships, prospecting_concerns, prospecting_achievements, prospecting_industry_pain, \
#         prospecting_operational_challenges, prospecting_latest_news, prospecting_recent_events, \
#         prospecting_customer_feedback
#
#     with app.app_context():
#         top_accounts = Account.query.order_by(Account.Score.desc()).limit(2).all()
#         for account in top_accounts:
#             company_name = account.Name
#             overview = prospecting_overview(company_name)
#             products = prospecting_products(company_name)
#             market = prospecting_market(company_name)
#             achievements = prospecting_achievements(company_name)
#             industry_pain = prospecting_industry_pain(company_name)
#             concerns = prospecting_concerns(company_name)
#             operational_challenges = prospecting_operational_challenges(company_name)
#             latest_news = prospecting_latest_news(company_name)
#             recent_events = prospecting_recent_events(company_name)
#             customer_feedback = prospecting_customer_feedback(company_name)
#             recent_partnerships = prospecting_recent_partnerships(company_name)
#
#             account.Overview = overview
#             account.Products = products
#             account.Market = market
#             account.Achievements = achievements
#             account.Market = market
#             account.IndustryPain = industry_pain
#             account.Concerns = concerns
#             account.OperationalChallenges = operational_challenges
#             account.LatestNews = latest_news
#             account.RecentEvents = recent_events
#             account.CustomerFeedback = customer_feedback
#             account.RecentPartnerships = recent_partnerships
#
#             db.session.add(account)
#             db.session.commit()
#
#             company_bio = f"{overview} {products} {market} {achievements} {industry_pain} {concerns} {operational_challenges} {latest_news} {recent_events} {customer_feedback} {recent_partnerships}"
#
#             # add all of these to the db for the specific account
#
#             chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)
#             template = (
#                 "You are a helpful assistant that takes in a customer company bio and converts it to a report for a sales "
#                 "rep. The sales rep works for a separate company and is trying to sell into the company with the bio. The bio will"
#                 "have an Overview (including leadership, products, market fit), Challenges, and News. The report should be 3 "
#                 "paragraphs long and one recommendation. At the end the recommendation is for the rep on how to proceed to get a meeting "
#                 "scheduled. The focus should be on solving a challenge for the customer. \n\n"
#                 "Bio: {text}."
#             )
#             system_message_prompt = SystemMessagePromptTemplate.from_template(template)
#             human_template = "{text}"
#             human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
#
#             chat_prompt = ChatPromptTemplate.from_messages(
#                 [system_message_prompt, human_message_prompt]
#             )
#
#             answer = chat(
#                 chat_prompt.format_prompt(
#                     text=company_bio
#                 ).to_messages()
#             )
#             print(answer.content)
#
#             # PROBLEMATIC USING SECTIONS HARDCODED
#             sections = answer.content.split('\n\n')  # Assuming paragraphs are separated by two newlines
#             overview_section = sections[0]
#             challenges_section = sections[1]
#             news_section = sections[2]
#             recommendation_section = sections[3]
#
#             cache_key = f"prospecting_data_{account.Id}"
#
#             data_to_cache = {
#                 'account': account,
#                 'overview_section': overview_section,
#                 'challenges_section': challenges_section,
#                 'news_section': news_section,
#                 'recommendation_section': recommendation_section
#             }
#
#             try:
#                 cache.set(cache_key, data_to_cache, timeout=24 * 60 * 60)  # Cache for 24 hours
#             except Exception as e:
#                 print(f"Error setting cache: {e}")
#
#
# @celery.task
# def my_background_task(arg1, arg2):
#     # some long running task here
#     print(arg1 + arg2)
#
#
# @scheduler.task('interval', id='do_rank_companies', days=1, start_date='2023-08-30 13:07:01')
# def scheduled_rank_companies():
#     try:
#         with app.app_context():
#             ranked_companies = rank_companies()
#             for company_id, (score, rank) in ranked_companies.items():
#                 # search for account by company_id
#                 company = Account.query.get(company_id)
#                 if company:  # Check if company exists
#                     company.Score = score
#                     company.Rank = rank
#                     db.session.add(company)
#             db.session.commit()
#             logging.info("Companies ranked successfully.")
#     except Exception as e:
#         logging.error(f"Error in scheduled_rank_companies: {e}")
#
#
# @app.cli.command("rank-companies")
# def rank_companies_command():
#     scheduled_rank_companies()
#
#
# @app.route('/start_task')
# def start_task():
#     from app.background_process import my_background_task
#
#     task = my_background_task.apply_async(args=[10, 20])
#     return 'Task started'
