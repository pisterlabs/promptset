import requests
from bs4 import BeautifulSoup
from rest_framework.views import APIView
from ..helpers.helpers import return_response, get_soup_from_url, compare_fractions, get_fighters_fighting_style, get_fighters_record_again_each_opponents_fight_style_using_url, find_max, get_fighters_fighting_stance, get_all_fights_in_event, get_basic_fight_stats_from_event, get_fighters_wins_if_in_top_10, get_upcoming_events, get_fighters_record_again_each_opponents_fight_style_using_db
from rest_framework import status
from rest_framework.decorators import api_view
import pandas as pd
import openai
import os
import time
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

load_dotenv()

class EventsList(APIView):
  def get(self, request):
    events = get_upcoming_events()
    # dan dont need below
    # get_fighters_wins_if_in_top_10('conor mcgregor')

    return return_response(events, 'Sucess', status.HTTP_200_OK)
  
  @api_view(['POST'])
  def get_event_by_id(request):
    # url = f'http://ufcstats.com/event-details/3c6976f8182d9527'
    url = request.data.get('event_url')
    fights_data = get_all_fights_in_event(url)
    return return_response(fights_data, 'Success', status.HTTP_200_OK)
  
  @api_view(['POST'])
  def get_basic_fight_stats(request):
    url = request.data.get('fight_url')
    fight_date = request.data.get('fight_date')
    fight_stats = get_basic_fight_stats_from_event(url)
    return return_response(fight_stats, 'Success', status.HTTP_200_OK)
          
  @api_view(['GET'])
  def get_in_depth_stats(request):
    fighter_1 = request.GET.get('fighter_1')

    fighter_2 = request.GET.get('fighter_2')
    fighter_1_style = get_fighters_fighting_style(fighter_1)
    fighter_2_style = get_fighters_fighting_style(fighter_2)
    fighter_1_stance = get_fighters_fighting_stance(fighter_1)
    fighter_1_stance = fighter_1_stance.lower()
    fighter_2_stance = get_fighters_fighting_stance(fighter_2)
    fighter_2_stance = fighter_2_stance.lower()

    
    fighter_1_names_list = fighter_1.split()
    fighter_1_stats_search_url = f'http://ufcstats.com/statistics/fighters/search?query={fighter_1_names_list[0]}'

    fighter_1_search_soup = get_soup_from_url(fighter_1_stats_search_url)
    fighter_1_a_tag = fighter_1_search_soup.find('a', text=re.compile(fighter_1_names_list[1].lower(), re.I))

    fighter_1_stats_url_href =  fighter_1_a_tag.get('href')
    
    fighter_2_names_list = fighter_2.split()
    fighter_2_stats_search_url = f'http://ufcstats.com/statistics/fighters/search?query={fighter_2_names_list[0]}&page=all'
    fighter_2_search_soup = get_soup_from_url(fighter_2_stats_search_url)

    fighter_2_a_tag = fighter_2_search_soup.find('a', text=re.compile(fighter_2_names_list[1].lower(), re.I))

    fighter_2_stats_url_href =  fighter_2_a_tag.get('href')
    
    # fighter_1_data = get_fighters_record_again_each_opponents_fight_style_using_url(fighter_1,fighter_1_stats_url_href ) # Legacy but gets up to date stats, db currently dosnt have fights past 2023-08-12
    
    fighter_1_data = get_fighters_record_again_each_opponents_fight_style_using_db(fighter_1)
    fighter_2_data = get_fighters_record_again_each_opponents_fight_style_using_db(fighter_1)
    
    # fighter_2_data = get_fighters_record_again_each_opponents_fight_style_using_url(fighter_2,fighter_2_stats_url_href ) # Legacy but gets up to date stats, db currently dosnt have fights past 2023-08-12
    
    all_fighter_1_opponents = fighter_1_data['opponents']
    all_fighter_1_opponents= [item.lower() for item in all_fighter_1_opponents]
    all_fighter_2_opponents = fighter_2_data['opponents']
    all_fighter_2_opponents = [item.lower() for item in all_fighter_2_opponents]
    all_fighter_1_results_against_all_opponents = fighter_1_data['result_against_opponents']
    all_fighter_2_results_against_all_opponents = fighter_2_data['result_against_opponents']
    
    fighter_1_record_agains_each_opponent_style = fighter_1_data['record']
    fighter_2_record_agains_each_opponent_style = fighter_2_data['record']

    fighter_1_record_agains_each_opponent_stance = fighter_1_data['fighter_1_record_agains_each_opponent_stance']
    fighter_2_record_agains_each_opponent_stance = fighter_2_data['fighter_1_record_agains_each_opponent_stance']
    
    fighter_1_record_agains_fighter_2_srtle = fighter_1_record_agains_each_opponent_style[fighter_2_style] if fighter_2_style in fighter_1_record_agains_each_opponent_style else []
    fighter_1_record_agains_fighter_2_stance = fighter_1_record_agains_each_opponent_stance[fighter_2_stance] if fighter_2_stance in fighter_1_record_agains_each_opponent_stance else []    

    fighter_2_record_agains_fighter_1_srtle = fighter_2_record_agains_each_opponent_style[fighter_1_style] if fighter_1_style in fighter_2_record_agains_each_opponent_style else []

    fighter_2_record_agains_fighter_1_stance = fighter_2_record_agains_each_opponent_stance[fighter_1_stance] if fighter_1_stance in fighter_2_record_agains_each_opponent_stance else []

    have_fighters_fought_before: bool = fighter_1.lower() in all_fighter_2_opponents or fighter_2.lower() in all_fighter_1_opponents
    
    fighter_with_more_wins_over_other = None
    fighter_with_better_record_against_opponent_style = None
    fighter_1_record_against_fighter_2 = None
    fighter_2_record_against_fighter_1 = None
    amount_of_fighter_1_wins_agains_fighter_2 = None
    amount_of_fighter_1_loss_agains_fighter_2 = None
    amount_of_fighter_1_draw_agains_fighter_2 = None
    amount_of_fighter_1_nc_agains_fighter_2 = None
    amount_of_fighter_2_wins_agains_fighter_1 = None
    amount_of_fighter_2_loss_agains_fighter_1 = None
    amount_of_fighter_2_draw_agains_fighter_1 = None
    amount_of_fighter_2_nc_agains_fighter_1 = None

    if have_fighters_fought_before == True:
      fighter_1_result_previous_results_against_fighter_2 = all_fighter_1_results_against_all_opponents[fighter_2.lower()]
      fighter_2_result_previous_results_against_fighter_1 = all_fighter_2_results_against_all_opponents[fighter_1.lower()]
      amount_of_fighter_1_wins_agains_fighter_2 = fighter_1_result_previous_results_against_fighter_2.count('win')
      amount_of_fighter_1_loss_agains_fighter_2 = fighter_1_result_previous_results_against_fighter_2.count('loss')
      amount_of_fighter_1_draw_agains_fighter_2 = fighter_1_result_previous_results_against_fighter_2.count('draw')
      amount_of_fighter_1_nc_agains_fighter_2 = fighter_1_result_previous_results_against_fighter_2.count('nc')
      fighter_1_record_against_fighter_2 = f"{amount_of_fighter_1_wins_agains_fighter_2}-{amount_of_fighter_1_loss_agains_fighter_2}-{amount_of_fighter_1_draw_agains_fighter_2} -{amount_of_fighter_1_nc_agains_fighter_2}nc"
      amount_of_fighter_2_wins_agains_fighter_1 = fighter_2_result_previous_results_against_fighter_1.count('win')
      amount_of_fighter_2_loss_agains_fighter_1 = fighter_2_result_previous_results_against_fighter_1.count('loss')
      amount_of_fighter_2_draw_agains_fighter_1 = fighter_2_result_previous_results_against_fighter_1.count('draw')
      amount_of_fighter_2_nc_agains_fighter_1 = fighter_2_result_previous_results_against_fighter_1.count('nc')
      fighter_2_record_against_fighter_1 = f"{amount_of_fighter_2_wins_agains_fighter_1}-{amount_of_fighter_2_loss_agains_fighter_1}-{amount_of_fighter_2_draw_agains_fighter_1} -{amount_of_fighter_2_nc_agains_fighter_1}nc"

      if amount_of_fighter_1_wins_agains_fighter_2 > amount_of_fighter_2_wins_agains_fighter_1:
        fighter_with_more_wins_over_other = fighter_1
      elif amount_of_fighter_2_wins_agains_fighter_1 > amount_of_fighter_1_wins_agains_fighter_2:
        fighter_with_more_wins_over_other = fighter_2
    # Do something with return value if have fought before and equal
    else:
      fighter_1_result_previous_results_against_fighter_2 = None
    amount_of_times_fighter_1_has_fought_fighter_2_style = len(fighter_1_record_agains_fighter_2_srtle) if fighter_1_record_agains_fighter_2_srtle != None else None
    amount_of_times_fighter_1_has_fought_fighter_2_stance = len(fighter_1_record_agains_fighter_2_stance) if fighter_1_record_agains_fighter_2_stance != None else None
    amount_of_wins_fighter_1_has_against_fighter_2_style = fighter_1_record_agains_fighter_2_srtle.count('win') if isinstance(fighter_1_record_agains_fighter_2_srtle, list) else None
    amount_of_wins_fighter_1_has_against_fighter_2_stance = fighter_1_record_agains_fighter_2_stance.count('win') if isinstance(fighter_1_record_agains_fighter_2_stance, list) else None

    if amount_of_wins_fighter_1_has_against_fighter_2_style == 0:
      percentage_of_wins_fighter_1_has_against_fighter_2_style = 0
    
    else:
      percentage_of_wins_fighter_1_has_against_fighter_2_style = round(amount_of_wins_fighter_1_has_against_fighter_2_style / amount_of_times_fighter_1_has_fought_fighter_2_style, 2) if amount_of_wins_fighter_1_has_against_fighter_2_style != None else None
    if amount_of_wins_fighter_1_has_against_fighter_2_stance == 0:
      percentage_of_wins_fighter_1_has_against_fighter_2_stance = 0
    
    else:
      percentage_of_wins_fighter_1_has_against_fighter_2_stance = round(amount_of_wins_fighter_1_has_against_fighter_2_stance / amount_of_times_fighter_1_has_fought_fighter_2_stance, 2) if amount_of_wins_fighter_1_has_against_fighter_2_stance != None else None
    
    amount_of_times_fighter_2_has_fought_fighter_1_style = len(fighter_2_record_agains_fighter_1_srtle) if fighter_2_record_agains_fighter_1_srtle != None else None
    amount_of_times_fighter_2_has_fought_fighter_1_stance = len(fighter_2_record_agains_fighter_1_stance) if fighter_2_record_agains_fighter_1_stance != None else None
    amount_of_wins_fighter_2_has_against_fighter_1_style = fighter_2_record_agains_fighter_1_srtle.count('win') if isinstance(fighter_2_record_agains_fighter_1_srtle, list) else None
    amount_of_wins_fighter_2_has_against_fighter_1_stance = fighter_2_record_agains_fighter_1_stance.count('win') if isinstance(fighter_2_record_agains_fighter_1_stance, list) else None

    if amount_of_wins_fighter_2_has_against_fighter_1_style == 0:
      percentage_of_wins_fighter_2_has_against_fighter_1_style = 0
    else:
      percentage_of_wins_fighter_2_has_against_fighter_1_style = round(amount_of_wins_fighter_2_has_against_fighter_1_style / amount_of_times_fighter_2_has_fought_fighter_1_style, 2) if amount_of_wins_fighter_2_has_against_fighter_1_style != None else None
    if amount_of_wins_fighter_2_has_against_fighter_1_stance == 0:
      percentage_of_wins_fighter_2_has_against_fighter_1_stance = 0
    else:
      percentage_of_wins_fighter_2_has_against_fighter_1_stance = round(amount_of_wins_fighter_2_has_against_fighter_1_stance / amount_of_times_fighter_2_has_fought_fighter_1_stance, 2) if amount_of_wins_fighter_2_has_against_fighter_1_stance != None else None

    fighter_with_better_record_against_opponent_style = find_max((fighter_1, percentage_of_wins_fighter_1_has_against_fighter_2_style), (fighter_2, percentage_of_wins_fighter_2_has_against_fighter_1_style)) if percentage_of_wins_fighter_1_has_against_fighter_2_style != None else None  
    fighter_with_better_record_against_opponent_stance = find_max((fighter_1, percentage_of_wins_fighter_1_has_against_fighter_2_stance), (fighter_2, percentage_of_wins_fighter_2_has_against_fighter_1_stance)) if percentage_of_wins_fighter_1_has_against_fighter_2_stance != None else None  

    res = {'fighter_with_better_record_against_opponent_style': fighter_with_better_record_against_opponent_style, 'fighter_with_better_record_against_opponent_stance': fighter_with_better_record_against_opponent_stance, 'fighter_with_more_wins_over_other': fighter_with_more_wins_over_other, 'percentage_of_wins_fighter_1_has_against_fighter_2_style': percentage_of_wins_fighter_1_has_against_fighter_2_style, 'percentage_of_wins_fighter_2_has_against_fighter_1_style': percentage_of_wins_fighter_2_has_against_fighter_1_style, 'fighter_1_result_previous_results_against_fighter_2': fighter_1_result_previous_results_against_fighter_2, 'fighter_1_record_against_fighter_2': fighter_1_record_against_fighter_2, 'fighter_2_record_against_fighter_1': fighter_2_record_against_fighter_1, 'fighter_1_stance': fighter_1_stance, 'fighter_2_stance': fighter_2_stance, 'amount_of_times_fighter_2_has_fought_fighter_1_stance': amount_of_times_fighter_2_has_fought_fighter_1_stance, 'amount_of_wins_fighter_2_has_against_fighter_1_stance': amount_of_wins_fighter_2_has_against_fighter_1_stance,
           'amount_of_times_fighter_1_has_fought_fighter_2_stance': amount_of_times_fighter_1_has_fought_fighter_2_stance, 'amount_of_wins_fighter_1_has_against_fighter_2_stance': amount_of_wins_fighter_1_has_against_fighter_2_stance,
           'percentage_of_wins_fighter_1_has_against_fighter_2_stance': percentage_of_wins_fighter_1_has_against_fighter_2_stance,
           'percentage_of_wins_fighter_2_has_against_fighter_1_stance': percentage_of_wins_fighter_2_has_against_fighter_1_stance,
           'amount_of_times_fighter_1_has_fought_fighter_2_style': amount_of_times_fighter_1_has_fought_fighter_2_style,
           'amount_of_times_fighter_2_has_fought_fighter_1_style': amount_of_times_fighter_2_has_fought_fighter_1_style,
           'amount_of_wins_fighter_1_has_against_fighter_2_style': amount_of_wins_fighter_1_has_against_fighter_2_style,
           'amount_of_wins_fighter_2_has_against_fighter_1_style': amount_of_wins_fighter_2_has_against_fighter_1_style,

           }
    return return_response(res, 'Success', status.HTTP_200_OK)

  @api_view(['GET'])
  def get_next_event_poster(request):
    ufc_event_url = 'https://www.ufc.com/events'

    response = requests.get(ufc_event_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    poster_element = soup.find('div', class_='c-hero__image')

    if poster_element:
        source_element = poster_element.find('source')
        if source_element:
            srcset = source_element['srcset']
            urls = srcset.split(',')
            first_url = urls[0].split(' ')[0]
            return return_response(first_url, 'Success', status.HTTP_200_OK)

    else:
        return return_response({}, 'Error', status.HTTP_200_OK)
    
    return return_response({}, 'Error', status.HTTP_200_OK)
