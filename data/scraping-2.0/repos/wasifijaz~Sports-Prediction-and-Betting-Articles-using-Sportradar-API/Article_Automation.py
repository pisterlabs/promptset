# importing libraries
import requests
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import openai
import time
import schedule
import os
import random
import shutil
import command 
import pytz
import smtplib, ssl
from email.mime.text import MIMEText

load_dotenv('var.env')
# SAVEDIRPATH = os.environ['save_dir']
# IMAGESPATH = os.environ['images_dir']
SAVEDIRPATH = "/home/ubuntu/article-automation"
IMAGESPATH = "/home/ubuntu/article-automation"

class Sitpicks:

    def __init__(self):
        article_date = date.today()
        self.DAY = str(article_date.day)
        self.MONTH = str(article_date.month)
        self.YEAR = str(article_date.year)


    def get_data_request(self, URL):
        """
        This function takes a URL as an argument and returns the data in JSON format.
        If the request is not successful, it returns the status code.
        """
        time.sleep(20)
        # sending get request and saving the response as response object
        response = requests.get(url = URL)
        if response.status_code == 200:
            # extracting data in json format
            data = response.json()
            return data
        else:
            print(f"There's a {response.status_code} error with your request")
            return response.status_code
    

    def generate_response(self, statement):
        """
        Takes string as an input and usea OpenAI API to generate responses.
        """
        time.sleep(30)
        openai.api_key = os.environ['openai_key']
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Rewrite the given statement in a catchy and appealing way.\nStatement:" + statement,
            temperature=0.5,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=1
            )
        return response.choices[0].text
    

    def get_games_info(self, URL):
        data = self.get_data_request(URL)
        return data

    
    def get_article_thumbnail(self, folder_path, homeTeamName):
        folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        for folder in folders:
            if homeTeamName in folder:
                files = [f for f in os.listdir(folder_path+'/'+folder) if os.path.isfile(os.path.join(folder_path+'/'+folder, f))]
                random_file = random.choice(files)
                thumbnail = [str(random_file), str(folder_path + '/' + folder + '/' + random_file)]
                return thumbnail
    

    
class NBA_Article(Sitpicks):

    def __init__(self):
        article_date = date.today()
        self.DAY = str(article_date.day)
        self.MONTH = str(article_date.month)
        self.YEAR = str(article_date.year)
        self.API_KEY = os.environ['nba_api_key']
        self.ODDS_API_KEY = os.environ['odds_api_key']
        

    def get_season_points(self, element):
        return element['average']['points']
    

    def get_season_steals(self, element):
        return element['average']['steals']
    

    def get_season_blocks(self, element):
        return element['average']['blocks']


    def generate_html(self, games_data_article):
        for data in games_data_article:
            html = "<head><meta charset='UTF-8'><meta name='keywords' content='HTML, CSS, JavaScript'></head>"
            html += Sitpicks.generate_response(self, data['Player Stats (T1)'][0]['home_top_season_player_name'] + " and the " + data['Team 1 (T1)'] + " will look to fend off " + data['Player Stats (T2)'][0]['away_top_season_player_name'] + "s " + data['Team 2 (T2)'] + " on " + data['Game Day'] + ". The " + data['favorite_team'] + " are (" + data['spread'] + ")-point favorite. A point total of (" + data['total_over_under'] + ") is set for the game. Find more below on the " + data['Team 1 (T1)'].split()[-1] + " vs. " + data['Team 2 (T2)'].split()[-1] + " betting line, injury report, head-to-head stats, best bets and more.")
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Spread and Betting Line</h2>"
            html += '\n<table class="table"><caption>' + data['Team 1 (T1)'].split()[-1] + ' vs ' + data['Team 2 (T2)'].split()[-1] + ' Betting Information</caption>'
            table1_col = ["Favorite", "Spread", "Favorite Spread Odds", "Underdog Spread Odds", "Total", "Over Total Odds", "Under Total Odds", "Favorite Moneyline", "Underdog Moneyline"]
            table1_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table1_cn.join(table1_col)}</th>\n</tr>\n</thead>'
            table1_row = [data['favorite_team'],data['spread'],data['favorite_spread_odds'],data['underdog_spread_odds'],data['total_over_under'],data['over_total_odds'],data['under_total_odds'],data['favorite_moneyline'],data['underdog_moneyline']]
            table1_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table1_rn.join(table1_row)}</td>\n</tr>\n</tbody>\n</table>'
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Game Info</h2>"
            html += "\n<ul>\n<li><strong>Game Day:</strong> " + str(data['Game Day']) + "</li>\n<li><strong>Game Time:</strong> " + str(data['Game Time']) + "</li>\n<li><strong>Location:</strong> " + data['Location'] + "</li>\n<li><strong>Arena:</strong> " + data['Arena'] + "</li>\n</ul>"
            html += "\n<h2>Computer Predictions for " + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + "</h2>"
            html += '\n<table class="table"><caption>Computer Picks for ' + data['Team 1 (T1)'].split()[-1] + ' vs ' + data['Team 2 (T2)'].split()[-1] + '</caption>'
            table2_col = ["ATS", "Over/Under", "AI Prediction"]
            table2_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table2_cn.join(table2_col)}</th>\n</tr>\n</thead>'
            table2_row = ["+"+data['spread'],data['total_over_under'],data["favorite_team"]+" "+data['spread']+" Over "+data["total_over_under"]]
            table2_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table2_rn.join(table2_row)}</td>\n</tr>\n</tbody>\n</table>'
            # html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Betting Trends</h2>"
            # article_betting_trends = [
            #     Sitpicks.generate_response(self, data['Team 1 (T1)'].split()[0] + "'s record against the spread last year was xx-xx-x."), 
            #     Sitpicks.generate_response(self, "As "+data['spread']+"-point favorites or more, the " + data['Team 1 (T1)'].split()[-1] + " went xx-xx against the spread last season."), 
            #     Sitpicks.generate_response(self, "There were xx " + data['Team 1 (T1)'].split()[0] + " games (out of xx) that went over the total last year.")
            #     ]
            list2_rn = '</li>\n<li>'
            # html += f'\n<ul>\n<li>{list2_rn.join(article_betting_trends)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'] + " Leaders</h2>"
            article_team1_leader = [
                Sitpicks.generate_response(self, data['Player Stats (T1)'][0]['home_top_season_player_name'] + " paced his squad in points (" + str(data['Player Stats (T1)'][0]['home_top_season_points']) + "), rebounds (" + str(data['Player Stats (T1)'][0]['home_top_season_rebounds']) + ") and assists (" + str(data['Player Stats (T1)'][0]['home_top_season_assists']) + ") per contest last season, shooting " + str(round(data['Player Stats (T1)'][0]['home_top_season_field_point'],1)) + "% from the field and " + str(round(data['Player Stats (T1)'][0]['home_top_season_downtownpoint'],1)) + "% from downtown with " + str(data['Player Stats (T1)'][0]['home_top_season_three_points']) + " made 3-pointers per contest. At the other end, he delivered " + str(data['Player Stats (T1)'][0]['home_top_season_steals']) + " steals and " + str(data['Player Stats (T1)'][0]['home_top_season_blocks']) + " blocks."),
                Sitpicks.generate_response(self, data['Player Stats (T1)'][1]['home_top_season_player_name'] + " posted " + str(data['Player Stats (T1)'][1]['home_top_season_points']) + " points, " + str(data['Player Stats (T1)'][1]['home_top_season_assists']) + " assists and " + str(data['Player Stats (T1)'][1]['home_top_season_rebounds']) + " rebounds per contest last year."),
                Sitpicks.generate_response(self, data['Player Stats (T1)'][2]['home_top_season_player_name'] + " averaged " + str(data['Player Stats (T1)'][2]['home_top_season_points']) + " points, " + str(data['Player Stats (T1)'][2]['home_top_season_rebounds']) + " rebounds and " + str(data['Player Stats (T1)'][2]['home_top_season_assists']) + " assists per contest last season. At the other end, he averaged " + str(data['Player Stats (T1)'][2]['home_top_season_steals']) + " steals and " + str(data['Player Stats (T1)'][2]['home_top_season_blocks']) + " blocks."),
                Sitpicks.generate_response(self, data['Player Stats (T1)'][3]['home_top_season_player_name'] + " put up " + str(data['Player Stats (T1)'][3]['home_top_season_points']) + " points, " + str(data['Player Stats (T1)'][3]['home_top_season_rebounds']) + " rebounds and " + str(data['Player Stats (T1)'][3]['home_top_season_assists']) + " assists per contest last year. At the other end, he posted " + str(data['Player Stats (T1)'][3]['home_top_season_steals']) + " steals and " + str(data['Player Stats (T1)'][3]['home_top_season_blocks']) + " blocks."),
                Sitpicks.generate_response(self, data['Player Stats (T1)'][4]['home_top_season_player_name'] + " put up " + str(data['Player Stats (T1)'][4]['home_top_season_points']) + " points, " + str(data['Player Stats (T1)'][4]['home_top_season_rebounds']) + " rebounds and " + str(data['Player Stats (T1)'][4]['home_top_season_assists']) + " assists per game last year, shooting " + str(round(data['Player Stats (T1)'][4]['home_top_season_field_point'],1)) + "% from the floor and " + str(round(data['Player Stats (T1)'][4]['home_top_season_downtownpoint'],1)) + "% from beyond the arc with " + str(data['Player Stats (T1)'][4]['home_top_season_three_points']) + " made 3-pointers per game.")
                ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team1_leader)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'] + " Season Stats</h2>"
            html += '\n<table class="table">'
            table3_col = ["", "Stat"]
            table3_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table3_cn.join(table3_col)}</th>\n</tr>\n</thead>'
            table3_row1 = ["<strong>Field Goal %</strong>",str(data['Field Goal (T1)'])]
            table3_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table3_rn.join(table3_row1)}</td>\n</tr>'
            table3_row2 = ["<strong>Opp. Field Goal %</strong>",str(data['Opp. Field Goal (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row2)}</td>\n</tr>'
            table3_row3 = ["<strong>Rebounds Per Game %</strong>",str(data['Rebounds (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row3)}</td>\n</tr>'
            table3_row4 = ["<strong>Opp. Rebounds Per Game %</strong>",str(data['Opp. Rebounds (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row4)}</td>\n</tr>'
            table3_row5 = ["<strong>Turnovers Per Game %</strong>",str(data['Turnovers (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row5)}</td>\n</tr>'
            table3_row6 = ["<strong>Opp. Turnovers Per Game %</strong>",str(data['Opp. Turnovers (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row6)}</td>\n</tr>\n</tbody>\n</table>'
            # html += "\n<h2>" + data['Team 2 (T2)'].split()[-1] + " vs " + data['Team 1 (T1)'].split()[-1] + " Betting Trends</h2>"
            # article_betting_trends_1 = [
            #     Sitpicks.generate_response(self, "Against the spread, " + data['Team 2 (T2)'].split()[0] + " is xx-xx-x this season."), 
            #     Sitpicks.generate_response(self, "The " + data['Team 2 (T2)'].split()[-1] + " are xx-xx as "+data['spread']+"-point underdogs or more."), 
            #     Sitpicks.generate_response(self, "Out of xx " + data['Team 2 (T2)'].split()[0] + " games so far this season, xx have hit the over.")
            #     ]
            # list3_rn = '</li>\n<li>'
            # html += f'\n<ul>\n<li>{list3_rn.join(article_betting_trends_1)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 2 (T2)'] + " Leaders</h2>"
            article_team2_leader = [
                Sitpicks.generate_response(self, data['Player Stats (T2)'][0]['away_top_season_player_name'] + " averages " + str(data['Player Stats (T2)'][0]['away_top_season_points']) + " points and adds " + str(data['Player Stats (T2)'][0]['away_top_season_assists']) + " assists per game, putting him at the top of the " + data['Team 2 (T2)'].split()[-1] + "’ leaderboards in those statistics."),
                Sitpicks.generate_response(self, data['Player Stats (T2)'][1]['away_top_season_player_name'] + " is at the top of the " + data['Team 2 (T2)'].split()[0] + " " + data['Team 2 (T2)'].split()[1] + " rebounding leaderboard with " + str(data['Player Stats (T2)'][1]['away_top_season_rebounds']) + " rebounds per game. He also notches " + str(data['Player Stats (T2)'][1]['away_top_season_points']) + " points and adds " + str(data['Player Stats (T2)'][1]['away_top_season_assists']) + " assists per game."),
                Sitpicks.generate_response(self, data['Team 2 (T2)'].split()[0] + " " + data['Team 2 (T2)'].split()[1] + " leader in steals is " + data['away_player_best_steals'] + " with " + str(data['away_player_best_steals_avg']) + " per game, and its leader in blocks is " + data['away_player_best_blocks'] + " with " + str(data['away_player_best_blocks_avg']) + " per game.")
                ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team2_leader)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 2 (T2)'] + " Season Stats</h2>"
            html += '\n<table class="table">'
            table4_col = ["", "Stat"]
            table4_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table4_cn.join(table4_col)}</th>\n</tr>\n</thead>'
            table4_row1 = ["<strong>Field Goal %</strong>",str(data['Field Goal (T2)'])]
            table4_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table4_rn.join(table4_row1)}</td>\n</tr>'
            table4_row2 = ["<strong>Opp. Field Goal %</strong>",str(data['Opp. Field Goal (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row2)}</td>\n</tr>'
            table4_row3 = ["<strong>Rebounds Per Game %</strong>",str(data['Rebounds (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row3)}</td>\n</tr>'
            table4_row4 = ["<strong>Opp. Rebounds Per Game %</strong>",str(data['Opp. Rebounds (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row4)}</td>\n</tr>'
            table4_row5 = ["<strong>Turnovers Per Game %</strong>",str(data['Turnovers (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row5)}</td>\n</tr>'
            table4_row6 = ["<strong>Opp. Turnovers Per Game %</strong>",str(data['Opp. Turnovers (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row6)}</td>\n</tr>\n</tbody>\n</table>'
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Injury Report</h2>"
            article_team1_injury = ""
            for injury in data['Injuries (T1)']:
                article_team1_injury += "\n" + injury['name'] + ": " + injury['status'] + " (" + injury['desc'] + ")<br>"
            html += "\n<strong>" + data['Team 1 (T1)'].split()[-1] + ": </strong>" + article_team1_injury
            article_team2_injury = ""
            for injury in data['Injuries (T2)']:
                article_team2_injury += "\n" + injury['name'] + ": " + injury['status'] + " (" + injury['desc'] + ")<br>"
            html += "\n<strong>" + data['Team 2 (T2)'].split()[-1] + ": </strong>" + article_team2_injury
            html += "\n<h2>Betting Tips for " + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + "</h2>"
            html += Sitpicks.generate_response(self, "\nWe have the " + data['Team 1 (T1)'].split()[-1] + " ("+data['spread']+") predicted as our best bet in this game. Our computer model has the scoring going over the total of "+data['total_over_under']+" points, finishing with the final outcome of " + data['Team 1 (T1)'] + " " + data['spread']+ "." )

            match_teams = data['Match']
            match_teams = match_teams.replace(" ","_")
            filename = match_teams + '_Betting_Pick_Against_the_Spread_' + str(data['Game Day'])
            filepath_name = os.path.join(SAVEDIRPATH + "/NBA/", filename+".html")
            try:
                text_file = open(filepath_name , "w")
                text_file.write(html)
                text_file.close()
            except Exception:
                pass
            images_folder_path = IMAGESPATH + "/NBA_Images"
            thumbnail = Sitpicks.get_article_thumbnail(self, images_folder_path, str(data['Team 1 (T1)'].split()[-1]))
            current_path = SAVEDIRPATH + "/NBA/"
            try:
                shutil.copy(thumbnail[1], current_path)
                os.chdir(SAVEDIRPATH + "/NBA")
                os.rename(thumbnail[0], filename+'.jpg')
            except Exception:
                pass
            print("Article Saved!")


    def offline_html(self, games_data_article):
        for data in games_data_article:
            html = "<head><meta charset='UTF-8'><meta name='keywords' content='HTML, CSS, JavaScript'></head>"
            html += data['Player Stats (T1)'][0]['home_top_season_player_name'] + " and the " + data['Team 1 (T1)'] + " will look to fend off " + data['Player Stats (T2)'][0]['away_top_season_player_name'] + "s " + data['Team 2 (T2)'] + " on " + data['Game Day'] + ". The " + data['favorite_team'] + " are (" + data['spread'] + ")-point favorite. A point total of (" + data['total_over_under'] + ") is set for the game. Find more below on the " + data['Team 1 (T1)'].split()[-1] + " vs. " + data['Team 2 (T2)'].split()[-1] + " betting line, injury report, head-to-head stats, best bets and more."
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Spread and Betting Line</h2>"
            html += '\n<table class="table"><caption>' + data['Team 1 (T1)'].split()[-1] + ' vs ' + data['Team 2 (T2)'].split()[-1] + ' Betting Information</caption>'
            table1_col = ["Favorite", "Spread", "Favorite Spread Odds", "Underdog Spread Odds", "Total", "Over Total Odds", "Under Total Odds", "Favorite Moneyline", "Underdog Moneyline"]
            table1_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table1_cn.join(table1_col)}</th>\n</tr>\n</thead>'
            table1_row = [data['favorite_team'],str(data['spread']),str(data['favorite_spread_odds']),str(data['underdog_spread_odds']),str(data['total_over_under']),str(data['over_total_odds']),str(data['under_total_odds']),str(data['favorite_moneyline']),str(data['underdog_moneyline'])]
            table1_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table1_rn.join(table1_row)}</td>\n</tr>\n</tbody>\n</table>'
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Game Info</h2>"
            html += "\n<ul>\n<li><strong>Game Day:</strong> " + str(data['Game Day']) + "</li>\n<li><strong>Game Time:</strong> " + str(data['Game Time']) + "</li>\n<li><strong>Location:</strong> " + data['Location'] + "</li>\n<li><strong>Arena:</strong> " + data['Arena'] + "</li>\n</ul>"
            html += "\n<h2>Computer Predictions for " + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + "</h2>"
            html += '\n<table class="table"><caption>Computer Picks for ' + data['Team 1 (T1)'].split()[-1] + ' vs ' + data['Team 2 (T2)'].split()[-1] + '</caption>'
            table2_col = ["ATS", "Over/Under", "AI Prediction"]
            table2_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table2_cn.join(table2_col)}</th>\n</tr>\n</thead>'
            table2_row = ["+"+data['spread'],data['total_over_under'],data["favorite_team"]+" -"+data['spread']+" Over "+data["total_over_under"]]
            table2_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table2_rn.join(table2_row)}</td>\n</tr>\n</tbody>\n</table>'
            # html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Betting Trends</h2>"
            # article_betting_trends = [
            #     data['Team 1 (T1)'].split()[0] + "'s record against the spread last year was xx-xx-x.", 
            #     "As "+data['spread']+"-point favorites or more, the " + data['Team 1 (T1)'].split()[-1] + " went xx-xx against the spread last season.", 
            #     "There were xx " + data['Team 1 (T1)'].split()[0] + " games (out of xx) that went over the total last year."
            #     ]
            list2_rn = '</li>\n<li>'
            # html += f'\n<ul>\n<li>{list2_rn.join(article_betting_trends)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'] + " Leaders</h2>"
            article_team1_leader = [
                data['Player Stats (T1)'][0]['home_top_season_player_name'] + " paced his squad in points (" + str(data['Player Stats (T1)'][0]['home_top_season_points']) + "), rebounds (" + str(data['Player Stats (T1)'][0]['home_top_season_rebounds']) + ") and assists (" + str(data['Player Stats (T1)'][0]['home_top_season_assists']) + ") per contest last season, shooting " + str(round(data['Player Stats (T1)'][0]['home_top_season_field_point'],1)) + "% from the field and " + str(round(data['Player Stats (T1)'][0]['home_top_season_downtownpoint'],1)) + "% from downtown with " + str(data['Player Stats (T1)'][0]['home_top_season_three_points']) + " made 3-pointers per contest. At the other end, he delivered " + str(data['Player Stats (T1)'][0]['home_top_season_steals']) + " steals and " + str(data['Player Stats (T1)'][0]['home_top_season_blocks']) + " blocks.",
                data['Player Stats (T1)'][1]['home_top_season_player_name'] + " posted " + str(data['Player Stats (T1)'][1]['home_top_season_points']) + " points, " + str(data['Player Stats (T1)'][1]['home_top_season_assists']) + " assists and " + str(data['Player Stats (T1)'][1]['home_top_season_rebounds']) + " rebounds per contest last year.",
                data['Player Stats (T1)'][2]['home_top_season_player_name'] + " averaged " + str(data['Player Stats (T1)'][2]['home_top_season_points']) + " points, " + str(data['Player Stats (T1)'][2]['home_top_season_rebounds']) + " rebounds and " + str(data['Player Stats (T1)'][2]['home_top_season_assists']) + " assists per contest last season. At the other end, he averaged " + str(data['Player Stats (T1)'][2]['home_top_season_steals']) + " steals and " + str(data['Player Stats (T1)'][2]['home_top_season_blocks']) + " blocks.",
                data['Player Stats (T1)'][3]['home_top_season_player_name'] + " put up " + str(data['Player Stats (T1)'][3]['home_top_season_points']) + " points, " + str(data['Player Stats (T1)'][3]['home_top_season_rebounds']) + " rebounds and " + str(data['Player Stats (T1)'][3]['home_top_season_assists']) + " assists per contest last year. At the other end, he posted " + str(data['Player Stats (T1)'][3]['home_top_season_steals']) + " steals and " + str(data['Player Stats (T1)'][3]['home_top_season_blocks']) + " blocks.",
                data['Player Stats (T1)'][4]['home_top_season_player_name'] + " put up " + str(data['Player Stats (T1)'][4]['home_top_season_points']) + " points, " + str(data['Player Stats (T1)'][4]['home_top_season_rebounds']) + " rebounds and " + str(data['Player Stats (T1)'][4]['home_top_season_assists']) + " assists per game last year, shooting " + str(round(data['Player Stats (T1)'][4]['home_top_season_field_point'],1)) + "% from the floor and " + str(round(data['Player Stats (T1)'][4]['home_top_season_downtownpoint'],1)) + "% from beyond the arc with " + str(data['Player Stats (T1)'][4]['home_top_season_three_points']) + " made 3-pointers per game."
                ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team1_leader)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'] + " Season Stats</h2>"
            html += '\n<table class="table">'
            table3_col = ["", "Stat"]
            table3_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table3_cn.join(table3_col)}</th>\n</tr>\n</thead>'
            table3_row1 = ["<strong>Field Goal %</strong>",str(data['Field Goal (T1)'])]
            table3_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table3_rn.join(table3_row1)}</td>\n</tr>'
            table3_row2 = ["<strong>Opp. Field Goal %</strong>",str(data['Opp. Field Goal (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row2)}</td>\n</tr>'
            table3_row3 = ["<strong>Rebounds Per Game %</strong>",str(data['Rebounds (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row3)}</td>\n</tr>'
            table3_row4 = ["<strong>Opp. Rebounds Per Game %</strong>",str(data['Opp. Rebounds (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row4)}</td>\n</tr>'
            table3_row5 = ["<strong>Turnovers Per Game %</strong>",str(data['Turnovers (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row5)}</td>\n</tr>'
            table3_row6 = ["<strong>Opp. Turnovers Per Game %</strong>",str(data['Opp. Turnovers (T1)'])]
            html += f'\n<tr>\n<td>{table3_rn.join(table3_row6)}</td>\n</tr>\n</tbody>\n</table>'
            # html += "\n<h2>" + data['Team 2 (T2)'].split()[-1] + " vs " + data['Team 1 (T1)'].split()[-1] + " Betting Trends</h2>"
            # article_betting_trends_1 = [
            #     "Against the spread, " + data['Team 2 (T2)'].split()[0] + " is xx-xx-x this season.", 
            #     "The " + data['Team 2 (T2)'].split()[-1] + " are xx-xx as "+data['spread']+"-point underdogs or more.", 
            #     "Out of xx " + data['Team 2 (T2)'].split()[0] + " games so far this season, xx have hit the over."
            #     ]
            # list3_rn = '</li>\n<li>'
            # html += f'\n<ul>\n<li>{list3_rn.join(article_betting_trends_1)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 2 (T2)'] + " Leaders</h2>"
            article_team2_leader = [
                data['Player Stats (T2)'][0]['away_top_season_player_name'] + " averages " + str(data['Player Stats (T2)'][0]['away_top_season_points']) + " points and adds " + str(data['Player Stats (T2)'][0]['away_top_season_assists']) + " assists per game, putting him at the top of the " + data['Team 2 (T2)'].split()[-1] + "’ leaderboards in those statistics.",
                data['Player Stats (T2)'][1]['away_top_season_player_name'] + " is at the top of the " + data['Team 2 (T2)'].split()[0] + " " + data['Team 2 (T2)'].split()[1] + " rebounding leaderboard with " + str(data['Player Stats (T2)'][1]['away_top_season_rebounds']) + " rebounds per game. He also notches " + str(data['Player Stats (T2)'][1]['away_top_season_points']) + " points and adds " + str(data['Player Stats (T2)'][1]['away_top_season_assists']) + " assists per game.",
                data['Team 2 (T2)'].split()[0] + " " + data['Team 2 (T2)'].split()[1] + " leader in steals is " + data['away_player_best_steals'] + " with " + str(data['away_player_best_steals_avg']) + " per game, and its leader in blocks is " + data['away_player_best_blocks'] + " with " + str(data['away_player_best_blocks_avg']) + " per game."
                ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team2_leader)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 2 (T2)'] + " Season Stats</h2>"
            html += '\n<table class="table">'
            table4_col = ["", "Stat"]
            table4_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table4_cn.join(table4_col)}</th>\n</tr>\n</thead>'
            table4_row1 = ["<strong>Field Goal %</strong>",str(data['Field Goal (T2)'])]
            table4_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table4_rn.join(table4_row1)}</td>\n</tr>'
            table4_row2 = ["<strong>Opp. Field Goal %</strong>",str(data['Opp. Field Goal (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row2)}</td>\n</tr>'
            table4_row3 = ["<strong>Rebounds Per Game %</strong>",str(data['Rebounds (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row3)}</td>\n</tr>'
            table4_row4 = ["<strong>Opp. Rebounds Per Game %</strong>",str(data['Opp. Rebounds (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row4)}</td>\n</tr>'
            table4_row5 = ["<strong>Turnovers Per Game %</strong>",str(data['Turnovers (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row5)}</td>\n</tr>'
            table4_row6 = ["<strong>Opp. Turnovers Per Game %</strong>",str(data['Opp. Turnovers (T2)'])]
            html += f'\n<tr>\n<td>{table4_rn.join(table4_row6)}</td>\n</tr>\n</tbody>\n</table>'
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Injury Report</h2>"
            article_team1_injury = ""
            for injury in data['Injuries (T1)']:
                article_team1_injury += "\n" + injury['name'] + ": " + injury['status'] + " (" + injury['desc'] + ")<br>"
            html += "\n<strong>" + data['Team 1 (T1)'].split()[-1] + ": </strong>" + article_team1_injury
            article_team2_injury = ""
            for injury in data['Injuries (T2)']:
                article_team2_injury += "\n" + injury['name'] + ": " + injury['status'] + " (" + injury['desc'] + ")<br>"
            html += "\n<strong>" + data['Team 2 (T2)'].split()[-1] + ": </strong>" + article_team2_injury
            html += "\n<h2>Betting Tips for " + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + "</h2>"
            html +=  "\nWe have the " + data['Team 1 (T1)'].split()[-1] + " ("+data['spread']+") predicted as our best bet in this game. Our computer model has the scoring going over the total of "+data['total_over_under']+" points, finishing with the final outcome of " + data['Team 1 (T1)'] + " "+data['spread']+ "."

            match_teams = data['Match']
            match_teams = match_teams.replace(" ","_")
            filename = match_teams + '_Betting_Pick_Against_the_Spread_' + str(data['Game Day'])
            filepath_name = os.path.join(SAVEDIRPATH + "/NBA/", filename+".html")
            # command.run(['sudo', 'chmod', '-R', '777', '/home/ubuntu/article-automation/NBA']) 
            # text_file = open(filepath_name , "w")
            # text_file.write(html)
            # text_file.close()
            try:
                text_file = open(filepath_name , "w")
                text_file.write(html)
                text_file.close()
            except Exception:
                pass
            images_folder_path = IMAGESPATH + "/NBA_Images"
            thumbnail = Sitpicks.get_article_thumbnail(self, images_folder_path, str(data['Team 1 (T1)'].split()[-1]))
            current_path = SAVEDIRPATH + "/NBA/"
            try:
                shutil.copy(thumbnail[1], current_path)
                os.chdir(SAVEDIRPATH+"/NBA")
                os.rename(thumbnail[0], filename+'.jpg')
            except Exception:
                pass
            print("Article Saved!")


    def nba_main(self):
        if os.path.exists('/home/ubuntu/article-automation/NBA'):
            command.run(['sudo', 'chmod', '-R', '777', '/home/ubuntu/article-automation/NBA']) 
            shutil.rmtree(SAVEDIRPATH+"/NBA", ignore_errors=False, onerror=None)
        if os.path.exists(SAVEDIRPATH+"/NBA.zip"):
            os.remove(SAVEDIRPATH+"/NBA.zip")
        print("NBA Folder Removed!")
        # api-endpoint
        URL = "http://api.sportradar.us/nba/trial/v7/en/games/" + self.YEAR + "/" + self.MONTH + "/" + self.DAY + "/schedule.json?api_key=" + self.API_KEY
        # Game Info
        data = Sitpicks.get_games_info(self, URL)
        if data.__contains__('games'):
            games = data['games']
            games_data_article = []
            for game in games:
                if game['status'] == 'unnecessary':
                    smtp_server = 'smtp.gmail.com'
                    smtp_port = 465
                    smtp_username = 'YOUR_EMAIL'
                    smtp_password = 'YOUR_PASSWORD'
                    context = ssl.create_default_context()

                    msg = MIMEText('Dear ,\
                    \n\nI hope this email finds you well. I am writing to inform you that unfortunately, we were unable to generate the NBA article you requested for the match between ' + game['home']['name'] + ' and ' + game['away']['name'] + '. This is due to the fact that the game had an unnecessary status, which means that it did not have a match sr_id associated with it. Without a match sr_id, our automated system is unable to fetch the necessary data from the sportsradar API to generate the article.\
                    \n\nWe understand that you were expecting a detailed article for this match, and we apologize that we could not deliver on this occasion. We apologize once again for any inconvenience caused, and we thank you for your understanding. If you have any questions or concerns, please do not hesitate to reach out to us.\
                    \n\nBest regards,\
                    \n\nAutomated Articles Bot\
                    \nSitpicks')
                    msg['Subject'] = 'Match Article Not Generated Due to Unnecessary Status'
                    msg['From'] = 'YOUR_EMAIL'
                    msg['To'] = 'RECEIVER_EMAIL'
                    sender_email = 'YOUR_EMAIL'
                    receiver_email = 'RECEIVER_EMAIL'
                    try:
                        server = smtplib.SMTP_SSL(smtp_server,smtp_port,context=context)
                        server.ehlo()
                        server.login(smtp_username, smtp_password)
                        server.sendmail(sender_email, receiver_email, msg.as_string())
                    except Exception as e:
                        print(e)
                    finally:
                        server.quit() 

                    continue
                game_id = game['id']
                game_scheduled = game['scheduled']
                game_sr_id = game['sr_id']
                game_date = datetime.strptime(game_scheduled, '%Y-%m-%dT%H:%M:%SZ').astimezone(pytz.timezone('US/Eastern')).date()
                game_time = datetime.strptime(game_scheduled, '%Y-%m-%dT%H:%M:%SZ').astimezone(pytz.timezone('US/Eastern')).time()
                game_time = game_time.strftime("%I:%M %p")
                game_date = game_date.strftime("%m-%d-%Y")
                # game_time = game_time.strftime("%H:%M")
                game_location = game['venue']['city'] + ', ' + game['venue']['state']
                game_arena = game['venue']['name']
                team_home = game['home']['name']
                team_home_id = game['home']['id']
                team_away = game['away']['name']
                team_away_id = game['away']['id']

                """Home Team Profile"""
                URL = "http://api.sportradar.us/nba/trial/v7/en/teams/" + team_home_id + "/profile.json?api_key=" + self.API_KEY
                home_team_profile = Sitpicks.get_data_request(self, URL)
                home_players = home_team_profile['players']
                home_players_names = []
                for players in home_players:
                    home_players_names.append(players['full_name'])

                """Season Stats Home"""
                SEASON_TYPE = "REG"
                YEAR = "2022"
                URL = "http://api.sportradar.us/nba/trial/v7/en/seasons/" + YEAR + "/" + SEASON_TYPE + "/teams/" + team_home_id + "/statistics.json?api_key=" + self.API_KEY
                home_season_stats = Sitpicks.get_data_request(self, URL)
                home_field_goal = home_season_stats['own_record']['average']['field_goals_made']
                home_opp_field_goal = home_season_stats['opponents']['average']['field_goals_made']
                home_rebounds = home_season_stats['own_record']['average']['rebounds']
                home_opp_rebounds = home_season_stats['opponents']['average']['rebounds']
                home_turnovers = home_season_stats['own_record']['average']['turnovers']
                home_opp_turnovers = home_season_stats['opponents']['average']['turnovers']

                home_player_season_stats = home_season_stats['players']
                home_player_season_stats.sort(key=self.get_season_points)
                home_player_season_stats.reverse()
                home_player_season_stats = home_player_season_stats[:5]

                """Top Player Stats Home"""
                top_player_stats_home = []
                for players_stats in home_player_season_stats:
                    home_top = {
                        'home_top_season_player_name': players_stats['full_name'],
                        'home_top_season_points': players_stats['average']['points'],
                        'home_top_season_rebounds': players_stats['average']['rebounds'],
                        'home_top_season_assists': players_stats['average']['assists'],
                        'home_top_season_field_point': players_stats['total']['field_goals_pct']*100,
                        'home_top_season_downtownpoint': players_stats['total']['three_points_pct']*100,
                        'home_top_season_three_points': players_stats['average']['three_points_made'],
                        'home_top_season_steals': players_stats['average']['steals'],
                        'home_top_season_blocks': players_stats['average']['blocks']
                    }
                    top_player_stats_home.append(home_top)

                """Away Team Profile"""
                URL = "http://api.sportradar.us/nba/trial/v7/en/teams/" + team_away_id + "/profile.json?api_key=" + self.API_KEY
                away_team_profile = Sitpicks.get_data_request(self, URL)
                away_players = away_team_profile['players']
                away_players_names = []
                for players in away_players:
                    away_players_names.append(players['full_name'])

                """Season Stats Away"""
                SEASON_TYPE = "REG"
                YEAR = "2022"
                URL = "http://api.sportradar.us/nba/trial/v7/en/seasons/" + YEAR + "/" + SEASON_TYPE + "/teams/" + team_away_id + "/statistics.json?api_key=" + self.API_KEY
                away_season_stats = Sitpicks.get_data_request(self, URL)
                away_field_goal = away_season_stats['own_record']['average']['field_goals_made']
                away_opp_field_goal = away_season_stats['opponents']['average']['field_goals_made']
                away_rebounds = away_season_stats['own_record']['average']['rebounds']
                away_opp_rebounds = away_season_stats['opponents']['average']['rebounds']
                away_turnovers = away_season_stats['own_record']['average']['turnovers']
                away_opp_turnovers = away_season_stats['opponents']['average']['turnovers']

                away_player_season_stats = away_season_stats['players']
                away_player_season_stats.sort(key=self.get_season_steals)
                away_player_season_stats.reverse()
                away_player_best_steals = away_player_season_stats[0]['full_name']
                away_player_best_steals_avg = away_player_season_stats[0]['average']['steals']

                away_player_season_stats.sort(key=self.get_season_blocks)
                away_player_season_stats.reverse()
                away_player_best_blocks = away_player_season_stats[0]['full_name']
                away_player_best_blocks_avg = away_player_season_stats[0]['average']['blocks']

                away_player_season_stats.sort(key=self.get_season_points)
                away_player_season_stats.reverse()
                away_player_season_stats = away_player_season_stats[:3]

                """Top Player Stats Away"""
                top_player_stats_away = []
                for players_stats in away_player_season_stats:
                    away_top = {
                        'away_top_season_player_name': players_stats['full_name'],
                        'away_top_season_points': players_stats['average']['points'],
                        'away_top_season_rebounds': players_stats['average']['rebounds'],
                        'away_top_season_assists': players_stats['average']['assists'],
                        'away_top_season_field_point': players_stats['total']['field_goals_pct']*100,
                        'away_top_season_downtownpoint': players_stats['total']['three_points_pct']*100,
                        'away_top_season_three_points': players_stats['average']['three_points_made'],
                        'away_top_season_steals': players_stats['average']['steals'],
                        'away_top_season_blocks': players_stats['average']['blocks']
                    }
                    top_player_stats_away.append(away_top)

                """Injurys"""
                home_injuries = []
                away_injuries = []
                URL = "http://api.sportradar.us/nba/trial/v7/en/league/injuries.json?api_key=" + self.API_KEY
                season_injury_data = Sitpicks.get_data_request(self, URL)
                season_injury_data = season_injury_data['teams']
                for player_injury in season_injury_data:
                    if player_injury['id'] == team_home_id:
                        home_player_injury = player_injury['players']
                        for injury in home_player_injury:
                            injury_data = {
                                'name': injury['full_name'], 
                                'status': injury['injuries'][0]['status'],
                                'desc': injury['injuries'][0]['desc']
                            }
                            home_injuries.append(injury_data)
                    if player_injury['id'] == team_away_id:
                        away_player_injury = player_injury['players']
                        for injury in away_player_injury:
                            injury_data = {
                                'name': injury['full_name'], 
                                'status': injury['injuries'][0]['status'],
                                'desc': injury['injuries'][0]['desc']
                            }
                            away_injuries.append(injury_data)

                """ODDs Data"""
                URL = "https://api.sportradar.us/oddscomparison-ust1/en/eu/sport_events/" + game_sr_id + "/markets.json?api_key=" + self.ODDS_API_KEY
                odds_data = Sitpicks.get_data_request(self, URL)
                odds_data = odds_data['sport_event']['consensus']['lines']

                spread = "-"
                home_odds = "-"
                away_odds = "-"
                underdog_spread_odds = "-"
                favorite_spread_odds = "-"
                underdog_moneyline = "-"
                favorite_moneyline = "-"
                favorite_team = "-"
                underdog_team = "-"
                total_over_under = "-"
                over_total_odds = "-"
                under_total_odds = "-"
                home_moneyline = "-"
                away_moneyline = "-"

                for line in odds_data:
                    if line['name'] == 'spread_current':
                        spread = line['spread']
                        home_odds = line['outcomes'][0]['odds']
                        away_odds = line['outcomes'][1]['odds']
                        if float(home_odds) > float(away_odds):
                            favorite_team = "0"
                        else:
                            favorite_team = "1"
                    if line['name'] == 'total_current':
                        total_over_under = line['total']
                        over_total_odds = '-'+str((float(line['outcomes'][0]['odds'])-1)*100)
                        under_total_odds = '-'+str((float(line['outcomes'][1]['odds'])-1)*100)
                    if line['name'] == 'moneyline_current':
                        home_moneyline = str(round((float(line['outcomes'][0]['odds'])-1) * 100,1))
                        away_moneyline = str(round((float(line['outcomes'][1]['odds'])-1) * 100,1))

                if float(away_odds) >= 2.00:
                    away_odds = -100/(float(away_odds) - 1)
                else:
                    away_odds = (float(away_odds) - 1)*100

                if float(home_odds) >= 2.00:
                    home_odds = -100/(float(home_odds) - 1)
                else:
                    home_odds = (float(home_odds) - 1)*100
                
                if away_odds > 0:
                    away_odds = '+'+str(away_odds)
                if home_odds > 0:
                    home_odds = '+'+str(home_odds)

                if favorite_team == "0":
                    favorite_team = team_home
                    underdog_team = team_away
                    underdog_spread_odds = away_odds
                    favorite_spread_odds = home_odds
                    underdog_moneyline = '+'+away_moneyline
                    favorite_moneyline = '-'+home_moneyline
                else:
                    favorite_team = team_away
                    underdog_team = team_home
                    underdog_spread_odds = home_odds
                    favorite_spread_odds = away_odds
                    underdog_moneyline = '+'+home_moneyline
                    favorite_moneyline = '-'+away_moneyline
                    
                game_data = {
                    "Match": team_home + ' vs ' + team_away,
                    "Game Day": game_date,
                    "Game Time": game_time,
                    "Location": game_location,
                    "Arena": game_arena,
                    "Team 1 (T1)": team_home,
                    "Player Stats (T1)": top_player_stats_home,
                    "Field Goal (T1)": home_field_goal,
                    "Opp. Field Goal (T1)": home_opp_field_goal,
                    "Rebounds (T1)": home_rebounds,
                    "Opp. Rebounds (T1)": home_opp_rebounds,
                    "Turnovers (T1)": home_turnovers,
                    "Opp. Turnovers (T1)": home_opp_turnovers,
                    "Injuries (T1)": home_injuries,
                    "Team 2 (T2)": team_away,
                    "Player Stats (T2)": top_player_stats_away,
                    "away_player_best_steals": away_player_best_steals,
                    "away_player_best_steals_avg": away_player_best_steals_avg,
                    "away_player_best_blocks": away_player_best_blocks,
                    "away_player_best_blocks_avg": away_player_best_blocks_avg,
                    "Field Goal (T2)": away_field_goal,
                    "Opp. Field Goal (T2)": away_opp_field_goal,
                    "Rebounds (T2)": away_rebounds,
                    "Opp. Rebounds (T2)": away_opp_rebounds,
                    "Turnovers (T2)": home_turnovers,
                    "Opp. Turnovers (T2)": away_opp_turnovers,
                    "Injuries (T2)": away_injuries,
                    "spread": spread,
                    "underdog_spread_odds": underdog_spread_odds,
                    "favorite_spread_odds": favorite_spread_odds,
                    "favorite_team": favorite_team,
                    "total_over_under": total_over_under,
                    "over_total_odds": over_total_odds,
                    "under_total_odds": under_total_odds,
                    "underdog_moneyline": underdog_moneyline,
                    "favorite_moneyline": favorite_moneyline,
                    "underdog_team": underdog_team
                }
                games_data_article.append(game_data)
                print("Article Written")
            
            os.mkdir(SAVEDIRPATH+"/NBA")
            print("NBA Folder Created!")
            try:
                self.generate_html(games_data_article)
                shutil.make_archive(SAVEDIRPATH+"/NBA", 'zip', SAVEDIRPATH+"/NBA")
                print("Success Online!")
            except:
                self.offline_html(games_data_article)            
                shutil.make_archive(SAVEDIRPATH+"/NBA", 'zip', SAVEDIRPATH+"/NBA")
                print("Success Offline!")
        else:
            print("Data not Found")



class NHL_Article(Sitpicks):
    
    def __init__(self):
        article_date = date.today()
        self.DAY = str(article_date.day)
        self.MONTH = str(article_date.month)
        self.YEAR = str(article_date.year)
        self.API_KEY = os.environ['nhl_api_key']
        self.ODDS_API_KEY = os.environ['odds_api_key']
        

    def get_season_goals(self, element):
        return element['statistics']['total']['goals']
    

    def get_season_saves(self, element):
        return element['goaltending']['total']['saves_pct']
    

    def generate_html(self, games_data_article):
        for data in games_data_article:
            html = "<head><meta charset='UTF-8'><meta name='keywords' content='HTML, CSS, JavaScript'></head>"
            if data["Game Number"] == 1: 
                html += Sitpicks.generate_response(self, "The " + data["Team 1 (T1)"] + " are set for Game 1 on " + str(data["Game Day"]) + " against the " + data["Team 2 (T2)"] + ", beginning at " + str(data["Game Time"]) + " ET. The oddsmakers have made the " + data["favorite_team"] + " solid favorites at " + data["favorite_moneyline"] + " on the moneyline, and the " + data["underdog_team"] + " are at " + data["underdog_moneyline"] + ". Find more below on the " + data["Team 1 (T1)"].split()[-1] + " vs. " + data["Team 2 (T2)"].split()[-1] + " betting line, injury report, head-to-head stats, best bets and more.")
            else:
                if data["Advantage Team"] is None:
                    html += Sitpicks.generate_response(self, "On " + str(data["Game Day"]) + " the " + data["Team 1 (T1)"] + " and the " + data["Team 2 (T2)"] + " will face off in Game " + str(data["Game Number"]) + ", beginning at " + str(data["Game Time"]) + " ET. The series is currently tied " + str(data["Home Points"]) + "-" + str(data["Away Points"]) + ". The sportsbooks have made the " + data["Team 1 (T1)"].split()[-1] + " slight favorites at xx on the moneyline, and the " + data["Team 2 (T2)"].split()[-1] + " are at xx. Find more below on the " + data["Match"] + " betting line, injury report, head-to-head stats, best bets and more.")
                elif data["Advantage Team"] == data["Team 1 (T1)"]:
                    html += Sitpicks.generate_response(self, "On " + str(data["Game Day"]) + " the " + data["Team 1 (T1)"] + " and the " + data["Team 2 (T2)"] + " will face off in Game " + str(data["Game Number"]) + ", beginning at " + str(data["Game Time"]) + " ET. The " + data["Team 1 (T1)"].split()[-1] + " have a " + str(data["Home Points"]) + "-" + str(data["Away Points"]) + " edge in the series. The sportsbooks have made the " + data["Team 1 (T1)"].split()[-1] + " slight favorites at xx on the moneyline, and the " + data["Team 2 (T2)"].split()[-1] + " are at xx. Find more below on the " + data["Match"] + " betting line, injury report, head-to-head stats, best bets and more.")
                else:
                    html += Sitpicks.generate_response(self, "On " + str(data["Game Day"]) + " the " + data["Team 1 (T1)"] + " and the " + data["Team 2 (T2)"] + " will face off in Game " + str(data["Game Number"]) + ", beginning at " + str(data["Game Time"]) + " ET. The " + data["Team 2 (T2)"].split()[-1] + " have a " + str(data["Home Points"]) + "-" + str(data["Away Points"]) + " edge in the series. The sportsbooks have made the " + data["Team 2 (T2)"].split()[-1] + " slight favorites at xx on the moneyline, and the " + data["Team 1 (T1)"].split()[-1] + " are at xx. Find more below on the " + data["Match"] + " betting line, injury report, head-to-head stats, best bets and more.")
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Spread and Betting Line</h2>"
            html += '\n<table class="table"><caption>' + data['Team 1 (T1)'].split()[-1] + ' vs ' + data['Team 2 (T2)'].split()[-1] + ' Betting Information</caption>'
            table1_col = ["Favorite", "Moneyline", "Underdog", "Moneyline"]
            table1_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table1_cn.join(table1_col)}</th>\n</tr>\n</thead>'
            table1_row = [data["favorite_team"],data["favorite_moneyline"],data["underdog_team"],data["underdog_moneyline"]]
            table1_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table1_rn.join(table1_row)}</td>\n</tr>\n</tbody>\n</table>'
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Game Info</h2>"
            html += "\n<ul>\n<li><strong>Game Day:</strong> " + str(data['Game Day']) + "</li>\n<li><strong>Game Time:</strong> " + str(data['Game Time']) + "</li>\n<li><strong>Location:</strong> " + data['Location'] + "</li>\n<li><strong>Arena:</strong> " + data['Arena'] + "</li>\n</ul>"
            html += "\n<h2>Computer Predictions for " + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + "</h2>"
            html += '\n<table class="table"><caption>Computer Picks for ' + data['Team 1 (T1)'].split()[-1] + ' vs ' + data['Team 2 (T2)'].split()[-1] + '</caption>'
            table2_col = ["ATS", "Over/Under", "AI Prediction"]
            table2_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table2_cn.join(table2_col)}</th>\n</tr>\n</thead>'
            # table2_row = ["+"+data['spread']+" ("+data["underdog_spread_odds"]+")","+/-"+data["total_over_under"],data["favorite_team"]+" "+data['spread']+" Over "+data["favorite_moneyline"]]
            table2_row = [data['total_over_under'],data["favorite_moneyline"],data["favorite_team"]+" "+data['spread']+" Over "+data["favorite_moneyline"]]
            table2_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table2_rn.join(table2_row)}</td>\n</tr>\n</tbody>\n</table>'
            # html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Betting Trends</h2>"
            # article_betting_trends = [
            #     Sitpicks.generate_response(self, "Through xx games as the moneyline favorite this season, " + data["favorite_team"] + " has won xx times."), 
            #     Sitpicks.generate_response(self, "The " + data["favorite_team"] + " have won xx of the xx games they have played with moneyline odds shorter than xx."), 
            #     Sitpicks.generate_response(self, "There is a " + str(data["win_pct_ml_f"]) + "% chance that " + data["favorite_team"] + " wins this contest, per the moneyline.")
            # ]
            list2_rn = '</li>\n<li>'
            # html += f'\n<ul>\n<li>{list2_rn.join(article_betting_trends)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'] + " Leaders</h2>"
            article_team1_leader = [
                Sitpicks.generate_response(self, data['Player Stats (T1)'][0]['home_top_season_player_name'] + " has been a major player for "+ data['Team 1 (T1)'].split()[-1] + " this season, with " + str(data['Player Stats (T1)'][0]['home_top_season_points']) + " points in " + str(data['Player Stats (T1)'][0]['home_top_season_games_played']) + " games."),
                Sitpicks.generate_response(self, "Through " + str(data['Player Stats (T1)'][1]['home_top_season_games_played']) + " games, " + data['Player Stats (T1)'][1]['home_top_season_player_name'] + " has " + str(data['Player Stats (T1)'][1]['home_top_season_goals']) + " goals and " + str(data['Player Stats (T1)'][1]['home_top_season_assists']) + " assists."),
                Sitpicks.generate_response(self, data['Player Stats (T1)'][2]['home_top_season_player_name'] + "season total of " + str(data['Player Stats (T1)'][2]['home_top_season_points']) + " points has come from " + str(data['Player Stats (T1)'][2]['home_top_season_goals']) + " goals and " + str(data['Player Stats (T1)'][2]['home_top_season_assists']) + " assists."),
                Sitpicks.generate_response(self, data['Player Stats (T1)'][3]['home_top_season_player_name'] + " has a record of " + str(data['Player Stats (T1)'][3]['home_top_season_points']) + " points. He has blocked " + str(data['Player Stats (T1)'][3]['home_top_season_blocks']) + " goals with a " + str(round(data['Player Stats (T1)'][3]['home_top_season_takeaway_pct'],1)) + "% save percentage."),
                Sitpicks.generate_response(self, data["Goal Keeper Stats (T1)"]['Name'] + " has a record of " + data["Goal Keeper Stats (T1)"]['Record'] + ". He has conceded " + str(data["Goal Keeper Stats (T1)"]['Conceded Goals']) + " goals (" + str(data["Goal Keeper Stats (T1)"]['Avg Goals']) + " goals against average) and racked up " + str(data["Goal Keeper Stats (T1)"]['Saved Goals']) + " saves with a " + str(data["Goal Keeper Stats (T1)"]['Save Pct']) + " save percentage (xx-best in the league).")
            ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team1_leader)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'] + " Season Stats</h2>"
            article_team1_stats = [
                Sitpicks.generate_response(self, data['Team 1 (T1)'].split()[-1] + "’s " + str(data["Total Goals (T1)"]) + " goals this season make them the one of the best scoring team in the league."),
                Sitpicks.generate_response(self, data['Team 1 (T1)'].split()[-1] + " are ranked xx in NHL action for the fewest goals against this season, having conceded " + str(data["Opp. Total Goals (T1)"]) + " total goals (" + str(data["Avg goals against (T1)"]) + " per game)."),
                Sitpicks.generate_response(self, "With a " + str(data["Goals Diff (T1)"]) + " goal differential, they’re ranked one of the best in the NHL."),
                Sitpicks.generate_response(self, "The " + str(data["Powerplay Goals (T1)"]) + " power-play goals" + data['Team 1 (T1)'].split()[-1] + " have put up this season (on " + str(data["Powerplay Shots (T1)"]) + " power-play chances) lead the league."),
                Sitpicks.generate_response(self, data['Team 1 (T1)'].split()[-1] + "’s " + str(data["Powerplay Conversation (T1)"]) + "% power-play conversion rate ranks amoung the best in the NHL this season."),
                Sitpicks.generate_response(self, data['Team 1 (T1)'].split()[-1] + "’s offense has scored " + str(data["Shorthanded Goals (T1)"]) + " shorthanded goals this season."),
                Sitpicks.generate_response(self, data['Team 1 (T1)'].split()[-1] + " have the league’s xxth-ranked penalty-kill percentage at " + str(round(data["Penalty Goals Pct (T1)"],1)) + "%."),
                Sitpicks.generate_response(self, data['Team 1 (T1)'].split()[-1] + " win " + str(round(data["Faceoffs Pct (T1)"],1)) + "% of their faceoffs (xxth in the league)."),
                Sitpicks.generate_response(self, data['Team 1 (T1)'].split()[-1] + " have a " + str(round(data["Evenstrength Pct (T1)"],1)) + "% shooting percentage as a squad, ranking xx in the NHL."),
                Sitpicks.generate_response(self, data['Team 1 (T1)'].split()[-1] + "’s players are looking for their shutout win this season. As a team, they are averaging " + str(data["Average Hit (T1)"]) + " hits and " + str(data["Average Blocked (T1)"]) + " blocked shots per game.")
            ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team1_stats)}</li>\n</ul>'
            # html += "\n<h2>" + data['Team 2 (T2)'].split()[-1] + " vs " + data['Team 1 (T1)'].split()[-1] + " Betting Trends</h2>"
            # article_betting_trends_1 = [
            #     Sitpicks.generate_response(self, "The " + data["underdog_team"] + " have claimed an upset victory in xx, or xx%, of the xx games they have played as an underdog this season."), 
            #     Sitpicks.generate_response(self, data["favorite_team"] + " has played with moneyline odds of " + data["spread"] + " or longer once this season and lost that game."), 
            #     Sitpicks.generate_response(self, "The " + data["underdog_team"] + " have a " + str(round(data["win_pct_total_ud"],1)) + "% chance to win this game (implied from the moneyline).")
            # ]
            # list3_rn = '</li>\n<li>'
            # html += f'\n<ul>\n<li>{list3_rn.join(article_betting_trends_1)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 2 (T2)'] + " Leaders</h2>"
            article_team2_leader = [
                Sitpicks.generate_response(self, data['Player Stats (T2)'][0]['away_top_season_player_name'] + " has been a major player for "+ data['Team 2 (T2)'].split()[-1] + " this season, with " + str(data['Player Stats (T2)'][0]['away_top_season_points']) + " points in " + str(data['Player Stats (T2)'][0]['away_top_season_games_played']) + " games."),
                Sitpicks.generate_response(self, "Through " + str(data['Player Stats (T2)'][1]['away_top_season_games_played']) + " games, " + data['Player Stats (T2)'][1]['away_top_season_player_name'] + " has " + str(data['Player Stats (T2)'][1]['away_top_season_goals']) + " goals and " + str(data['Player Stats (T2)'][1]['away_top_season_assists']) + " assists."),
                Sitpicks.generate_response(self, data['Player Stats (T2)'][2]['away_top_season_player_name'] + "season total of " + str(data['Player Stats (T2)'][2]['away_top_season_points']) + " points has come from " + str(data['Player Stats (T2)'][2]['away_top_season_goals']) + " goals and " + str(data['Player Stats (T2)'][2]['away_top_season_assists']) + " assists."),
                Sitpicks.generate_response(self, data['Player Stats (T2)'][3]['away_top_season_player_name'] + " has a record of " + str(data['Player Stats (T2)'][3]['away_top_season_points']) + " points. He has blocked " + str(data['Player Stats (T2)'][3]['away_top_season_blocks']) + " goals with a " + str(round(data['Player Stats (T2)'][3]['away_top_season_takeaway_pct'],1)) + "% save percentage."),
                Sitpicks.generate_response(self, data["Goal Keeper Stats (T2)"]['Name'] + " has a record of " + data["Goal Keeper Stats (T2)"]['Record'] + ". He has conceded " + str(data["Goal Keeper Stats (T2)"]['Conceded Goals']) + " goals (" + str(data["Goal Keeper Stats (T2)"]['Avg Goals']) + " goals against average) and racked up " + str(data["Goal Keeper Stats (T2)"]['Saved Goals']) + " saves with a " + str(data["Goal Keeper Stats (T2)"]['Save Pct']) + " save percentage (xx-best in the league).")
            ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team2_leader)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 2 (T2)'] + " Season Stats</h2>"
            article_team2_stats = [
                Sitpicks.generate_response(self, data['Team 2 (T2)'].split()[-1] + "’s " + str(data["Total Goals (T2)"]) + " goals this season make them the one of the best scoring team in the league."),
                Sitpicks.generate_response(self, data['Team 2 (T2)'].split()[-1] + "are ranked xx in NHL action for the fewest goals against this season, having conceded " + str(data["Opp. Total Goals (T2)"]) + " total goals (" + str(data["Avg goals against (T2)"]) + " per game)."),
                Sitpicks.generate_response(self, "With a " + str(data["Goals Diff (T2)"]) + " goal differential, they’re ranked one of the best in the NHL."),
                Sitpicks.generate_response(self, "The " + str(data["Powerplay Goals (T2)"]) + " power-play goals" + data['Team 2 (T2)'].split()[-1] + " have put up this season (on " + str(data["Powerplay Shots (T2)"]) + " power-play chances) lead the league."),
                Sitpicks.generate_response(self, data['Team 2 (T2)'].split()[-1] + "’s " + str(data["Powerplay Conversation (T2)"]) + "% power-play conversion rate ranks amoung the best in the NHL this season."),
                Sitpicks.generate_response(self, data['Team 2 (T2)'].split()[-1] + "’s offense has scored " + str(data["Shorthanded Goals (T2)"]) + " shorthanded goals this season."),
                Sitpicks.generate_response(self, data['Team 2 (T2)'].split()[-1] + " have the league’s xxth-ranked penalty-kill percentage at " + str(round(data["Penalty Goals Pct (T2)"],1)) + "%."),
                Sitpicks.generate_response(self, data['Team 2 (T2)'].split()[-1] + " win " + str(data["Faceoffs Pct (T2)"]) + "% of their faceoffs (xxth in the league)."),
                Sitpicks.generate_response(self, data['Team 2 (T2)'].split()[-1] + " have a " + str(round(data["Evenstrength Pct (T2)"],1)) + "% shooting percentage as a squad, ranking xx in the NHL."),
                Sitpicks.generate_response(self, data['Team 2 (T2)'].split()[-1] + "’s players are looking for their shutout win this season. As a team, they are averaging " + str(data["Average Hit (T2)"]) + " hits and " + str(data["Average Blocked (T2)"]) + " blocked shots per game.")
            ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team2_stats)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Injury Report</h2>"
            article_team1_injury = ""
            for injury in data['Injuries (T1)']:
                article_team1_injury += "\n" + injury['name'] + ": " + injury['status'] + " (" + injury['desc'] + ")<br>"
            html += "\n<strong>" + data['Team 1 (T1)'].split()[-1] + ": </strong>" + article_team1_injury
            article_team2_injury = ""
            for injury in data['Injuries (T2)']:
                article_team2_injury += "\n" + injury['name'] + ": " + injury['status'] + " (" + injury['desc'] + ")<br>"
            html += "\n<strong>" + data['Team 2 (T2)'].split()[-1] + ": </strong>" + article_team2_injury
            html += "\n<h2>Betting Tips for " + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + "</h2>"
            html += Sitpicks.generate_response(self, "\nWe have the " + data['favorite_team'] + " (" + str(data['spread']) + ") predicted as our best bet in this game. Our computer model has the scoring going over the total of xx points, with the teams finishing with a final score of " + data['Team 1 (T1)'].split()[-1] + " xx, " + data['Team 2 (T2)'].split()[-1] + " xx when it’s sorted out on the court.")

            match_teams = data['Match']
            match_teams = match_teams.replace(" ","_")
            filename = match_teams + '_Betting_Pick_Against_the_Spread_' + str(data['Game Day'])
            filepath_name = os.path.join(SAVEDIRPATH + "/NHL/", filename+".html")
            try:
                text_file = open(filepath_name , "w")
                text_file.write(html)
                text_file.close()
            except Exception:
                pass
            images_folder_path = IMAGESPATH + "/NHL_Images"
            thumbnail = Sitpicks.get_article_thumbnail(self, images_folder_path, str(data['Team 1 (T1)'].split()[-1]))
            current_path = SAVEDIRPATH + "/NHL/"
            try:
                shutil.copy(thumbnail[1], current_path)
                os.chdir(SAVEDIRPATH + "/NHL")
                os.rename(thumbnail[0], filename+'.jpg')
            except Exception:
                pass
            print("Article Saved!")
            

    def offline_html(self, games_data_article):
        for data in games_data_article:
            html = "<head><meta charset='UTF-8'><meta name='keywords' content='HTML, CSS, JavaScript'></head>"
            if data["Game Number"] == 1: 
                html += "The " + data["Team 1 (T1)"] + " are set for Game 1 on " + str(data["Game Day"]) + " against the " + data["Team 2 (T2)"] + ", beginning at " + str(data["Game Time"]) + " ET. The oddsmakers have made the " + data["favorite_team"] + " solid favorites at " + data["favorite_moneyline"] + " on the moneyline, and the " + data["underdog_team"] + " are at " + data["underdog_moneyline"] + ". Find more below on the " + data["Team 1 (T1)"].split()[-1] + " vs. " + data["Team 2 (T2)"].split()[-1] + " betting line, injury report, head-to-head stats, best bets and more."
            else:
                if data["Advantage Team"] is None:
                    html += "On " + str(data["Game Day"]) + " the " + data["Team 1 (T1)"] + " and the " + data["Team 2 (T2)"] + " will face off in Game " + str(data["Game Number"]) + ", beginning at " + str(data["Game Time"]) + " ET. The series is currently tied " + str(data["Home Points"]) + "-" + str(data["Away Points"]) + ". The sportsbooks have made the " + data["Team 1 (T1)"].split()[-1] + " slight favorites at xx on the moneyline, and the " + data["Team 2 (T2)"].split()[-1] + " are at xx. Find more below on the " + data["Match"] + " betting line, injury report, head-to-head stats, best bets and more."
                elif data["Advantage Team"] == data["Team 1 (T1)"]:
                    html += "On " + str(data["Game Day"]) + " the " + data["Team 1 (T1)"] + " and the " + data["Team 2 (T2)"] + " will face off in Game " + str(data["Game Number"]) + ", beginning at " + str(data["Game Time"]) + " ET. The " + data["Team 1 (T1)"].split()[-1] + " have a " + str(data["Home Points"]) + "-" + str(data["Away Points"]) + " edge in the series. The sportsbooks have made the " + data["Team 1 (T1)"].split()[-1] + " slight favorites at xx on the moneyline, and the " + data["Team 2 (T2)"].split()[-1] + " are at xx. Find more below on the " + data["Match"] + " betting line, injury report, head-to-head stats, best bets and more."
                else:
                    html += "On " + str(data["Game Day"]) + " the " + data["Team 1 (T1)"] + " and the " + data["Team 2 (T2)"] + " will face off in Game " + str(data["Game Number"]) + ", beginning at " + str(data["Game Time"]) + " ET. The " + data["Team 2 (T2)"].split()[-1] + " have a " + str(data["Home Points"]) + "-" + str(data["Away Points"]) + " edge in the series. The sportsbooks have made the " + data["Team 2 (T2)"].split()[-1] + " slight favorites at xx on the moneyline, and the " + data["Team 1 (T1)"].split()[-1] + " are at xx. Find more below on the " + data["Match"] + " betting line, injury report, head-to-head stats, best bets and more."
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Spread and Betting Line</h2>"
            html += '\n<table class="table"><caption>' + data['Team 1 (T1)'].split()[-1] + ' vs ' + data['Team 2 (T2)'].split()[-1] + ' Betting Information</caption>'
            table1_col = ["Favorite", "Moneyline", "Underdog", "Moneyline"]
            table1_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table1_cn.join(table1_col)}</th>\n</tr>\n</thead>'
            table1_row = [data["favorite_team"],data["favorite_moneyline"],data["underdog_team"],data["underdog_moneyline"]]
            table1_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table1_rn.join(table1_row)}</td>\n</tr>\n</tbody>\n</table>'
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Game Info</h2>"
            html += "\n<ul>\n<li><strong>Game Day:</strong> " + str(data['Game Day']) + "</li>\n<li><strong>Game Time:</strong> " + str(data['Game Time']) + "</li>\n<li><strong>Location:</strong> " + data['Location'] + "</li>\n<li><strong>Arena:</strong> " + data['Arena'] + "</li>\n</ul>"
            html += "\n<h2>Computer Predictions for " + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + "</h2>"
            html += '\n<table class="table"><caption>Computer Picks for ' + data['Team 1 (T1)'].split()[-1] + ' vs ' + data['Team 2 (T2)'].split()[-1] + '</caption>'
            table2_col = ["ATS", "Over/Under", "AI Prediction"]
            table2_cn = '</th>\n<th scope="col">'
            html += f'\n<thead>\n<tr>\n<th scope="col">{table2_cn.join(table2_col)}</th>\n</tr>\n</thead>'
            # table2_row = ["+"+data['spread']+" ("+data["underdog_spread_odds"]+")","+/-"+data["total_over_under"],data["favorite_team"]+" "+data['spread']+" Over "+data["favorite_moneyline"]]
            table2_row = [data['total_over_under'],data["favorite_moneyline"],data["favorite_team"]+" "+data['spread']+" Over "+data["favorite_moneyline"]]
            table2_rn = '</td>\n<td>'
            html += f'\n<tbody>\n<tr>\n<td>{table2_rn.join(table2_row)}</td>\n</tr>\n</tbody>\n</table>'
            # html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Betting Trends</h2>"
            # article_betting_trends = [
            #     "Through xx games as the moneyline favorite this season, " + data["favorite_team"] + " has won xx times.", 
            #     "The " + data["favorite_team"] + " have won xx of the xx games they have played with moneyline odds shorter than xx.", 
            #     "There is a " + str(data["win_pct_ml_f"]) + "% chance that " + data["favorite_team"] + " wins this contest, per the moneyline."
            # ]
            list2_rn = '</li>\n<li>'
            # html += f'\n<ul>\n<li>{list2_rn.join(article_betting_trends)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'] + " Leaders</h2>"
            article_team1_leader = [
                data['Player Stats (T1)'][0]['home_top_season_player_name'] + " has been a major player for "+ data['Team 1 (T1)'].split()[-1] + " this season, with " + str(data['Player Stats (T1)'][0]['home_top_season_points']) + " points in " + str(data['Player Stats (T1)'][0]['home_top_season_games_played']) + " games.",
                "Through " + str(data['Player Stats (T1)'][1]['home_top_season_games_played']) + " games, " + data['Player Stats (T1)'][1]['home_top_season_player_name'] + " has " + str(data['Player Stats (T1)'][1]['home_top_season_goals']) + " goals and " + str(data['Player Stats (T1)'][1]['home_top_season_assists']) + " assists.",
                data['Player Stats (T1)'][2]['home_top_season_player_name'] + "season total of " + str(data['Player Stats (T1)'][2]['home_top_season_points']) + " points has come from " + str(data['Player Stats (T1)'][2]['home_top_season_goals']) + " goals and " + str(data['Player Stats (T1)'][2]['home_top_season_assists']) + " assists.",
                data['Player Stats (T1)'][3]['home_top_season_player_name'] + " has a record of " + str(data['Player Stats (T1)'][3]['home_top_season_points']) + " points. He has blocked " + str(data['Player Stats (T1)'][3]['home_top_season_blocks']) + " goals with a " + str(round(data['Player Stats (T1)'][3]['home_top_season_takeaway_pct'],1)) + "% save percentage.",
                data["Goal Keeper Stats (T1)"]['Name'] + " has a record of " + data["Goal Keeper Stats (T1)"]['Record'] + ". He has conceded " + str(data["Goal Keeper Stats (T1)"]['Conceded Goals']) + " goals (" + str(data["Goal Keeper Stats (T1)"]['Avg Goals']) + " goals against average) and racked up " + str(data["Goal Keeper Stats (T1)"]['Saved Goals']) + " saves with a " + str(data["Goal Keeper Stats (T1)"]['Save Pct']) + " save percentage (xx-best in the league)."
            ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team1_leader)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'] + " Season Stats</h2>"
            article_team1_stats = [
                data['Team 1 (T1)'].split()[-1] + "’s " + str(data["Total Goals (T1)"]) + " goals this season make them the one of the best scoring team in the league.",
                data['Team 1 (T1)'].split()[-1] + " are ranked xx in NHL action for the fewest goals against this season, having conceded " + str(data["Opp. Total Goals (T1)"]) + " total goals (" + str(data["Avg goals against (T1)"]) + " per game).",
                "With a " + str(data["Goals Diff (T1)"]) + " goal differential, they’re ranked one of the best in the NHL.",
                "The " + str(data["Powerplay Goals (T1)"]) + " power-play goals" + data['Team 1 (T1)'].split()[-1] + " have put up this season (on " + str(data["Powerplay Shots (T1)"]) + " power-play chances) lead the league.",
                data['Team 1 (T1)'].split()[-1] + "’s " + str(data["Powerplay Conversation (T1)"]) + "% power-play conversion rate ranks amoung the best in the NHL this season.",
                data['Team 1 (T1)'].split()[-1] + "’s offense has scored " + str(data["Shorthanded Goals (T1)"]) + " shorthanded goals this season.",
                data['Team 1 (T1)'].split()[-1] + " have the league’s xxth-ranked penalty-kill percentage at " + str(round(data["Penalty Goals Pct (T1)"],1)) + "%.",
                data['Team 1 (T1)'].split()[-1] + " win " + str(round(data["Faceoffs Pct (T1)"],1)) + "% of their faceoffs (xxth in the league).",
                data['Team 1 (T1)'].split()[-1] + " have a " + str(round(data["Evenstrength Pct (T1)"],1)) + "% shooting percentage as a squad, ranking xx in the NHL.",
                data['Team 1 (T1)'].split()[-1] + "’s players are looking for their shutout win this season. As a team, they are averaging " + str(data["Average Hit (T1)"]) + " hits and " + str(data["Average Blocked (T1)"]) + " blocked shots per game."
            ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team1_stats)}</li>\n</ul>'
            # html += "\n<h2>" + data['Team 2 (T2)'].split()[-1] + " vs " + data['Team 1 (T1)'].split()[-1] + " Betting Trends</h2>"
            # article_betting_trends_1 = [
            #     "The " + data["favorite_team"] + " have claimed an upset victory in xx, or " + str(data["win_pct_total_f"]) + "%, of the xx games they have played as an underdog this season.", 
            #     data["underdog_team"] + " has played with moneyline odds of " + str(data["underdog_spread_odds"]) + " or longer once this season and lost that game.", 
            #     "The " + data["favorite_team"] + " have a " + str(data["win_pct_ml_f"]) + "% chance to win this game (implied from the moneyline)."
            # ]
            # list3_rn = '</li>\n<li>'
            # html += f'\n<ul>\n<li>{list3_rn.join(article_betting_trends_1)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 2 (T2)'] + " Leaders</h2>"
            article_team2_leader = [
                data['Player Stats (T2)'][0]['away_top_season_player_name'] + " has been a major player for "+ data['Team 2 (T2)'].split()[-1] + " this season, with " + str(data['Player Stats (T2)'][0]['away_top_season_points']) + " points in " + str(data['Player Stats (T2)'][0]['away_top_season_games_played']) + " games.",
                "Through " + str(data['Player Stats (T2)'][1]['away_top_season_games_played']) + " games, " + data['Player Stats (T2)'][1]['away_top_season_player_name'] + " has " + str(data['Player Stats (T2)'][1]['away_top_season_goals']) + " goals and " + str(data['Player Stats (T2)'][1]['away_top_season_assists']) + " assists.",
                data['Player Stats (T2)'][2]['away_top_season_player_name'] + "season total of " + str(data['Player Stats (T2)'][2]['away_top_season_points']) + " points has come from " + str(data['Player Stats (T2)'][2]['away_top_season_goals']) + " goals and " + str(data['Player Stats (T2)'][2]['away_top_season_assists']) + " assists.",
                data['Player Stats (T2)'][3]['away_top_season_player_name'] + " has a record of " + str(data['Player Stats (T2)'][3]['away_top_season_points']) + " points. He has blocked " + str(data['Player Stats (T2)'][3]['away_top_season_blocks']) + " goals with a " + str(round(data['Player Stats (T2)'][3]['away_top_season_takeaway_pct'],1)) + "% save percentage.",
                data["Goal Keeper Stats (T2)"]['Name'] + " has a record of " + data["Goal Keeper Stats (T2)"]['Record'] + ". He has conceded " + str(data["Goal Keeper Stats (T2)"]['Conceded Goals']) + " goals (" + str(data["Goal Keeper Stats (T2)"]['Avg Goals']) + " goals against average) and racked up " + str(data["Goal Keeper Stats (T2)"]['Saved Goals']) + " saves with a " + str(data["Goal Keeper Stats (T2)"]['Save Pct']) + " save percentage (xx-best in the league)."
            ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team2_leader)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 2 (T2)'] + " Season Stats</h2>"
            article_team2_stats = [
                data['Team 2 (T2)'].split()[-1] + "’s " + str(data["Total Goals (T2)"]) + " goals this season make them the one of the best scoring team in the league.",
                data['Team 2 (T2)'].split()[-1] + "are ranked xx in NHL action for the fewest goals against this season, having conceded " + str(data["Opp. Total Goals (T2)"]) + " total goals (" + str(data["Avg goals against (T2)"]) + " per game).",
                "With a " + str(data["Goals Diff (T2)"]) + " goal differential, they’re ranked one of the best in the NHL.",
                "The " + str(data["Powerplay Goals (T2)"]) + " power-play goals" + data['Team 2 (T2)'].split()[-1] + " have put up this season (on " + str(data["Powerplay Shots (T2)"]) + " power-play chances) lead the league.",
                data['Team 2 (T2)'].split()[-1] + "’s " + str(data["Powerplay Conversation (T2)"]) + "% power-play conversion rate ranks amoung the best in the NHL this season.",
                data['Team 2 (T2)'].split()[-1] + "’s offense has scored " + str(data["Shorthanded Goals (T2)"]) + " shorthanded goals this season.",
                data['Team 2 (T2)'].split()[-1] + " have the league’s xxth-ranked penalty-kill percentage at " + str(round(data["Penalty Goals Pct (T2)"],1)) + "%.",
                data['Team 2 (T2)'].split()[-1] + " win " + str(round(data["Faceoffs Pct (T2)"],1)) + "% of their faceoffs (xxth in the league).",
                data['Team 2 (T2)'].split()[-1] + " have a " + str(round(data["Evenstrength Pct (T2)"],1)) + "% shooting percentage as a squad, ranking xx in the NHL.",
                data['Team 2 (T2)'].split()[-1] + "’s players are looking for their shutout win this season. As a team, they are averaging " + str(data["Average Hit (T2)"]) + " hits and " + str(data["Average Blocked (T2)"]) + " blocked shots per game."
            ]
            html += f'\n<ul>\n<li>{list2_rn.join(article_team2_stats)}</li>\n</ul>'
            html += "\n<h2>" + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + " Injury Report</h2>"
            article_team1_injury = ""
            for injury in data['Injuries (T1)']:
                article_team1_injury += "\n" + injury['name'] + ": " + injury['status'] + " (" + injury['desc'] + ")<br>"
            html += "\n<strong>" + data['Team 1 (T1)'].split()[-1] + ": </strong>" + article_team1_injury
            article_team2_injury = ""
            for injury in data['Injuries (T2)']:
                article_team2_injury += "\n" + injury['name'] + ": " + injury['status'] + " (" + injury['desc'] + ")<br>"
            html += "\n<strong>" + data['Team 2 (T2)'].split()[-1] + ": </strong>" + article_team2_injury
            html += "\n<h2>Betting Tips for " + data['Team 1 (T1)'].split()[-1] + " vs " + data['Team 2 (T2)'].split()[-1] + "</h2>"
            html += "\nWe have the " + data['favorite_team'] + " (" + str(data['spread']) + ") predicted as our best bet in this game. Our computer model has the scoring going over the total of xx points, with the teams finishing with a final score of " + data['Team 1 (T1)'].split()[-1] + " xx, " + data['Team 2 (T2)'].split()[-1] + " xx when it’s sorted out on the court."

            match_teams = data['Match']
            match_teams = match_teams.replace(" ","_")
            filename = match_teams + '_Betting_Pick_Against_the_Spread_' + str(data['Game Day'])
            filepath_name = os.path.join(SAVEDIRPATH + "/NHL/", filename+".html")
            try:
                text_file = open(filepath_name , "w")
                text_file.write(html)
                text_file.close()
            except Exception:
                pass
            images_folder_path = IMAGESPATH + "/NHL_Images"
            thumbnail = Sitpicks.get_article_thumbnail(self, images_folder_path, str(data['Team 1 (T1)'].split()[-1]))
            current_path = SAVEDIRPATH + "/NHL/"
            try:
                shutil.copy(thumbnail[1], current_path)
                os.chdir(SAVEDIRPATH + "/NHL")
                os.rename(thumbnail[0], filename+'.jpg')
            except Exception:
                pass
            print("Article Saved!")


    def nhl_main(self):
        if os.path.exists('/home/ubuntu/article-automation/NHL'):
            command.run(['sudo', 'chmod', '-R', '777', '/home/ubuntu/article-automation/NHL']) 
            shutil.rmtree(SAVEDIRPATH+"/NHL", ignore_errors=False, onerror=None)
        if os.path.exists(SAVEDIRPATH+"/NHL.zip"):
            os.remove(SAVEDIRPATH+"/NHL.zip")
        print("NHL Folder Removed!")
        # api-endpoint
        URL = "http://api.sportradar.us/nhl/trial/v7/en/games/" + self.YEAR + "/" + self.MONTH + "/" + self.DAY + "/schedule.json?api_key=" + self.API_KEY
        # URL = "http://api.sportradar.us/nhl/trial/v7/en/games/2023/06/10/schedule.json?api_key=" + self.API_KEY
        # Game Info
        data = Sitpicks.get_games_info(self, URL)
        if data.__contains__('games'):
            games = data['games']
            games_data_article = []
            for game in games:
                if game['status'] == 'unnecessary':
                    smtp_server = 'smtp.gmail.com'
                    smtp_port = 465
                    smtp_username = 'YOUR_EMAIL'
                    smtp_password = 'YOUR_PASSWORD'
                    context = ssl.create_default_context()

                    msg = MIMEText('Dear ,\
                    \n\nI hope this email finds you well. I am writing to inform you that unfortunately, we were unable to generate the NHL article you requested for the match between ' + game['home']['name'] + ' and ' + game['away']['name'] + '. This is due to the fact that the game had an unnecessary status, which means that it did not have a match sr_id associated with it. Without a match sr_id, our automated system is unable to fetch the necessary data from the sportsradar API to generate the article.\
                    \n\nWe understand that you were expecting a detailed article for this match, and we apologize that we could not deliver on this occasion. We apologize once again for any inconvenience caused, and we thank you for your understanding. If you have any questions or concerns, please do not hesitate to reach out to us.\
                    \n\nBest regards,\
                    \n\nAutomated Articles Bot\
                    \nSitpicks')
                    msg['Subject'] = 'Match Article Not Generated Due to Unnecessary Status'
                    msg['From'] = 'YOUR_EMAIL'
                    msg['To'] = 'RECEIVER_EMAIL'
                    sender_email = 'YOUR_EMAIL'
                    receiver_email = 'RECEIVER_EMAIL'
                    try:
                        server = smtplib.SMTP_SSL(smtp_server,smtp_port,context=context)
                        server.ehlo()
                        server.login(smtp_username, smtp_password)
                        server.sendmail(sender_email, receiver_email, msg.as_string())
                    except Exception as e:
                        print(e)
                    finally:
                        server.quit() 

                    continue
                game_id = game['id']
                game_sr_id = game['sr_id']
                game_scheduled = game['scheduled']
                game_date = datetime.strptime(game_scheduled, '%Y-%m-%dT%H:%M:%SZ').astimezone(pytz.timezone('US/Eastern')).date()
                game_time = datetime.strptime(game_scheduled, '%Y-%m-%dT%H:%M:%SZ').astimezone(pytz.timezone('US/Eastern')).time()
                game_time = game_time.strftime("%I:%M %p")
                game_date = game_date.strftime("%m-%d-%Y")
                # game_time = game_time.strftime("%H:%M")
                game_location = game['venue']['city'] + ', ' + game['venue']['state']
                game_arena = game['venue']['name']
                team_home = game['home']['name']
                team_home_id = game['home']['id']
                team_away = game['away']['name']
                team_away_id = game['away']['id']
                if data.__contains__('home_points') and data.__contains__('away_points'):
                    home_game_points = game['home_points']
                    away_game_points = game['away_points']
                    game_number = home_game_points + away_game_points + 1
                    game_advantage = max(home_game_points, away_game_points)
                    if game_advantage > away_game_points:
                        game_advantage_team = team_home
                    elif game_advantage > home_game_points:
                        game_advantage_team = team_away
                    else:
                        game_advantage_team = None
                else:
                    home_game_points = 0
                    away_game_points = 0
                    game_number = 1
                    game_advantage_team = None
                
                """Home Team Profile"""
                URL = "http://api.sportradar.us/nhl/trial/v7/en/teams/" + team_home_id + "/profile.json?api_key=" + self.API_KEY
                home_team_profile = Sitpicks.get_data_request(self, URL)
                home_players = home_team_profile['players']
                home_players_names = []
                for players in home_players:
                    home_players_names.append(players['full_name'])
                
                """Season Stats Home"""
                SEASON_TYPE = "REG"
                YEAR = "2022"
                URL = "http://api.sportradar.us/nhl/trial/v7/en/seasons/" + YEAR + "/" + SEASON_TYPE + "/teams/" + team_home_id + "/statistics.json?api_key=" + self.API_KEY
                home_season_stats = Sitpicks.get_data_request(self, URL)
                home_total_goals = home_season_stats['own_record']['statistics']['total']['goals']
                home_avg_goals = home_season_stats['own_record']['statistics']['average']['goals']
                home_opp_total_goals = home_season_stats['opponents']['statistics']['total']['goals']
                home_goals_diff = home_total_goals - home_opp_total_goals
                home_pp_goals = home_season_stats['own_record']['statistics']['powerplay']['goals'] 
                home_pp_shots = home_season_stats['own_record']['statistics']['powerplay']['shots'] 
                home_pp_conv_rate = home_season_stats['own_record']['statistics']['powerplay']['percentage'] 
                home_sh_goals = home_season_stats['own_record']['statistics']['shorthanded']['goals'] 
                home_penalty_goals = home_season_stats['own_record']['statistics']['penalty']['goals'] 
                home_penalty_shots = home_season_stats['own_record']['statistics']['penalty']['shots'] 
                if home_penalty_shots > 0:
                    home_penalty_win_percentage = (home_penalty_goals/home_penalty_shots)*100
                else:
                    home_penalty_win_percentage = 0
                home_faceoff_win_percentage = home_season_stats['own_record']['statistics']['evenstrength']['faceoff_win_pct']
                home_es_shots = home_season_stats['own_record']['statistics']['evenstrength']['shots']
                home_es_goals = home_season_stats['own_record']['statistics']['evenstrength']['goals']
                if home_es_shots > 0:
                    home_es_connect_pct = (home_es_goals/home_es_shots)*100
                else:
                    home_es_shots = 0
                home_avg_blocked = home_season_stats['own_record']['statistics']['average']['blocked_shots']
                home_avg_hit = home_season_stats['own_record']['statistics']['average']['hits']
                home_avg_goals_against = home_season_stats['own_record']['goaltending']['average']['goals_against']
                home_player_season_stats = home_season_stats['players']
                
                home_player_season_stats_gk = []
                for goalkeeper in home_player_season_stats:
                    if goalkeeper.__contains__('goaltending'):
                        home_player_season_stats_gk.append(goalkeeper)
                
                home_player_season_stats.sort(key=self.get_season_goals)
                home_player_season_stats.reverse()
                home_player_season_stats = home_player_season_stats[:5]
                
                home_player_season_stats_gk.sort(key=self.get_season_saves)
                home_player_season_stats_gk = home_player_season_stats_gk[-1]
                home_gk_season_stats = {
                    'Record': str(home_player_season_stats_gk['goaltending']['total']['wins']) + "-" + str(home_player_season_stats_gk['goaltending']['total']['losses']) + "-" + str(home_player_season_stats_gk['goaltending']['total']['overtime_losses']),
                    'Name': home_player_season_stats_gk['full_name'],
                    'Save Pct': home_player_season_stats_gk['goaltending']['total']['saves_pct'],
                    'Conceded Goals': home_player_season_stats_gk['goaltending']['total']['goals_against'],
                    'Saved Goals': home_player_season_stats_gk['goaltending']['total']['saves'],
                    'Avg Goals': home_player_season_stats_gk['goaltending']['total']['avg_goals_against']
                }
                
                """Top Player Stats Home"""
                top_player_stats_home = []
                for players_stats in home_player_season_stats:
                    home_top = {
                        'home_top_season_player_name': players_stats['full_name'],
                        'home_top_season_games_played': players_stats['statistics']['total']['games_played'],
                        'home_top_season_goals': players_stats['statistics']['total']['goals'],
                        'home_top_season_points': players_stats['statistics']['total']['points'],
                        'home_top_season_assists': players_stats['statistics']['total']['assists'],
                        'home_top_season_blocks': players_stats['statistics']['total']['blocked_shots'],
                        'home_top_season_takeaway_pct': players_stats['statistics']['average']['takeaways']*100,
                        'home_top_season_giveaway_pct': players_stats['statistics']['average']['giveaways']*100,
                        'home_top_season_giveaway_pct': players_stats['statistics']['average']['goals']
                    }
                    top_player_stats_home.append(home_top)
                
                """Away Team Profile"""
                URL = "http://api.sportradar.us/nhl/trial/v7/en/teams/" + team_away_id + "/profile.json?api_key=" + self.API_KEY
                away_team_profile = Sitpicks.get_data_request(self, URL)
                away_players = away_team_profile['players']
                away_players_names = []
                for players in home_players:
                    away_players_names.append(players['full_name'])
                
                """Season Stats Away"""
                SEASON_TYPE = "REG"
                YEAR = "2022"
                URL = "http://api.sportradar.us/nhl/trial/v7/en/seasons/" + YEAR + "/" + SEASON_TYPE + "/teams/" + team_away_id + "/statistics.json?api_key=" + self.API_KEY
                away_season_stats = Sitpicks.get_data_request(self, URL)
                away_total_goals = away_season_stats['own_record']['statistics']['total']['goals']
                away_avg_goals = away_season_stats['own_record']['statistics']['average']['goals']
                away_opp_total_goals = away_season_stats['opponents']['statistics']['total']['goals']
                away_goals_diff = away_total_goals - away_opp_total_goals
                away_pp_goals = away_season_stats['own_record']['statistics']['powerplay']['goals'] 
                away_pp_shots = away_season_stats['own_record']['statistics']['powerplay']['shots'] 
                away_pp_conv_rate = away_season_stats['own_record']['statistics']['powerplay']['percentage'] 
                away_sh_goals = away_season_stats['own_record']['statistics']['shorthanded']['goals'] 
                away_penalty_goals = away_season_stats['own_record']['statistics']['penalty']['goals'] 
                away_penalty_shots = away_season_stats['own_record']['statistics']['penalty']['shots'] 
                if away_penalty_shots > 0:
                    away_penalty_win_percentage = (away_penalty_goals/away_penalty_shots)*100
                else:
                    away_penalty_win_percentage = 0
                away_faceoff_win_percentage = away_season_stats['own_record']['statistics']['evenstrength']['faceoff_win_pct']
                away_es_shots = away_season_stats['own_record']['statistics']['evenstrength']['shots']
                away_es_goals = away_season_stats['own_record']['statistics']['evenstrength']['goals']
                if away_es_shots > 0:
                    away_es_connect_pct = (away_es_goals/away_es_shots)*100
                else:
                    away_es_shots = 0
                away_avg_blocked = away_season_stats['own_record']['statistics']['average']['blocked_shots']
                away_avg_hit = away_season_stats['own_record']['statistics']['average']['hits']
                away_avg_goals_against = away_season_stats['own_record']['goaltending']['average']['goals_against']
                away_player_season_stats = away_season_stats['players']
                
                away_player_season_stats_gk = []
                for goalkeeper in away_player_season_stats:
                    if goalkeeper.__contains__('goaltending'):
                        away_player_season_stats_gk.append(goalkeeper)
                
                away_player_season_stats.sort(key=self.get_season_goals)
                away_player_season_stats.reverse()
                away_player_season_stats = away_player_season_stats[:5]
                
                away_player_season_stats_gk.sort(key=self.get_season_saves)
                away_player_season_stats_gk = away_player_season_stats_gk[-1]
                away_gk_season_stats = {
                    'Record': str(away_player_season_stats_gk['goaltending']['total']['wins']) + "-" + str(away_player_season_stats_gk['goaltending']['total']['losses']) + "-" + str(away_player_season_stats_gk['goaltending']['total']['overtime_losses']),
                    'Name': away_player_season_stats_gk['full_name'],
                    'Save Pct': away_player_season_stats_gk['goaltending']['total']['saves_pct'],
                    'Conceded Goals': away_player_season_stats_gk['goaltending']['total']['goals_against'],
                    'Saved Goals': away_player_season_stats_gk['goaltending']['total']['saves'],
                    'Avg Goals': away_player_season_stats_gk['goaltending']['total']['avg_goals_against']
                }
                
                """Top Player Stats away"""
                top_player_stats_away = []
                for players_stats in away_player_season_stats:
                    away_top = {
                        'away_top_season_player_name': players_stats['full_name'],
                        'away_top_season_games_played': players_stats['statistics']['total']['games_played'],
                        'away_top_season_goals': players_stats['statistics']['total']['goals'],
                        'away_top_season_points': players_stats['statistics']['total']['points'],
                        'away_top_season_assists': players_stats['statistics']['total']['assists'],
                        'away_top_season_blocks': players_stats['statistics']['total']['blocked_shots'],
                        'away_top_season_takeaway_pct': players_stats['statistics']['average']['takeaways']*100,
                        'away_top_season_giveaway_pct': players_stats['statistics']['average']['giveaways']*100,
                        'away_top_season_giveaway_pct': players_stats['statistics']['average']['goals']
                    }
                    top_player_stats_away.append(away_top)
                
                """Injurys"""
                home_injuries = []
                away_injuries = []
                URL = "http://api.sportradar.us/nhl/trial/v7/en/league/injuries.json?api_key=" + self.API_KEY
                season_injury_data = Sitpicks.get_data_request(self, URL)
                season_injury_data = season_injury_data['teams']
                for player_injury in season_injury_data:
                    if player_injury['id'] == team_home_id:
                        home_player_injury = player_injury['players']
                        for injury in home_player_injury:
                            injury_data = {
                                'name': injury['full_name'], 
                                'status': injury['injuries'][0]['status'],
                                'desc': injury['injuries'][0]['desc']
                            }
                            home_injuries.append(injury_data)
                    if player_injury['id'] == team_away_id:
                        away_player_injury = player_injury['players']
                        for injury in away_player_injury:
                            injury_data = {
                                'name': injury['full_name'], 
                                'status': injury['injuries'][0]['status'],
                                'desc': injury['injuries'][0]['desc']
                            }
                            away_injuries.append(injury_data)
                
                """ODDs Data"""
                URL = "https://api.sportradar.us/oddscomparison-ust1/en/eu/sport_events/" + game_sr_id + "/markets.json?api_key=" + self.ODDS_API_KEY
                odds_data = Sitpicks.get_data_request(self, URL)
                odds_data = odds_data['sport_event']['consensus']
                pct_data = []
                if odds_data.__contains__('bet_percentage_outcomes'):
                    pct_data = odds_data['bet_percentage_outcomes']
                odds_data = odds_data['lines']

                spread = "-"
                home_odds = "-"
                away_odds = "-"
                underdog_spread_odds = "-"
                favorite_spread_odds = "-"
                underdog_moneyline = "-"
                favorite_moneyline = "-"
                favorite_team = "-"
                underdog_team = "-"
                total_over_under = "-"
                over_total_odds = "-"
                under_total_odds = "-"
                home_moneyline = "-"
                away_moneyline = "-"
                win_pct_ml_home = "-"
                win_pct_total_home = "-"
                win_pct_ml_away = "-"
                win_pct_total_away = "-"
                win_pct_ml_f = "-"
                win_pct_total_f = "-"
                win_pct_ml_ud = "-"
                win_pct_total_ud = "-"

                for line in odds_data:
                    if line['name'] == 'spread_current':
                        spread = line['spread']
                        home_odds = line['outcomes'][0]['odds']
                        away_odds = line['outcomes'][1]['odds']
                        if float(home_odds) > float(away_odds):
                            favorite_team = "0"
                        else:
                            favorite_team = "1"
                    if line['name'] == 'total_current':
                        total_over_under = '-'+line['total']
                        over_total_odds = '-'+str((float(line['outcomes'][0]['odds'])-1)*100)
                        under_total_odds = '-'+str((float(line['outcomes'][1]['odds'])-1)*100)
                    if line['name'] == 'moneyline_current':
                        home_moneyline = str(round((float(line['outcomes'][0]['odds'])-1) * 100,1))
                        away_moneyline = str(round((float(line['outcomes'][1]['odds'])-1) * 100,1))

                if not pct_data:
                    pass 
                else:
                    for pct in pct_data:
                        if pct['name'] == 'moneyline':
                            win_pct_ml_home = pct['outcomes'][0]['percentage']
                            win_pct_ml_home = round(float(win_pct_ml_home),1)
                            win_pct_ml_away = pct['outcomes'][1]['percentage']
                            win_pct_ml_away = round(float(win_pct_ml_away),1)
                        if pct['name'] == 'total':
                            win_pct_total_home = pct['outcomes'][0]['percentage']
                            win_pct_total_away = pct['outcomes'][1]['percentage']

                if away_odds != '-':                    
                    if float(away_odds) >= 2.00:
                        away_odds = -100/(float(away_odds) - 1)
                    else:
                        away_odds = (float(away_odds) - 1)*100
                    
                    if away_odds > 0:
                        away_odds = '+'+str(away_odds)
                
                if home_odds != '-':
                    if float(home_odds) >= 2.00:
                        home_odds = -100/(float(home_odds) - 1)
                    else:
                        home_odds = (float(home_odds) - 1)*100
                
                    if home_odds > 0:
                        home_odds = '+'+str(home_odds)

                if favorite_team == "0":
                    favorite_team = team_home
                    underdog_team = team_away
                    underdog_spread_odds = away_odds
                    favorite_spread_odds = home_odds
                    underdog_moneyline = '+'+away_moneyline
                    favorite_moneyline = '-'+home_moneyline
                    win_pct_ml_ud = str(win_pct_ml_away)
                    win_pct_total_ud = win_pct_total_away
                    win_pct_total_f = win_pct_total_home
                    win_pct_ml_f = str(win_pct_ml_home)
                else:
                    favorite_team = team_away
                    underdog_team = team_home
                    underdog_spread_odds = home_odds
                    favorite_spread_odds = away_odds
                    underdog_moneyline = '+'+home_moneyline
                    favorite_moneyline = '-'+str(away_moneyline)
                    win_pct_ml_ud = win_pct_ml_home
                    win_pct_total_ud = win_pct_total_home
                    win_pct_total_f = win_pct_total_away
                    win_pct_ml_f = str(win_pct_ml_home)

                game_data = {
                    "Match": team_home + ' vs ' + team_away,
                    "Game Day": game_date,
                    "Game Time": game_time,
                    "Location": game_location,
                    "Arena": game_arena,
                    "Home Points": home_game_points,
                    "Away Points": away_game_points,
                    "Game Number": game_number,
                    "Advantage Team": game_advantage_team,
                    "Team 1 (T1)": team_home,
                    "Player Stats (T1)": top_player_stats_home,
                    "Goal Keeper Stats (T1)": home_gk_season_stats,
                    "Total Goals (T1)": home_total_goals,
                    "Average Goal (T1)": home_avg_goals,
                    "Opp. Total Goals (T1)": home_opp_total_goals,
                    "Avg goals against (T1)": home_avg_goals_against,
                    "Goals Diff (T1)": home_goals_diff,
                    "Powerplay Goals (T1)": home_pp_goals,
                    "Powerplay Shots (T1)": home_pp_shots,
                    "Powerplay Conversation (T1)": home_pp_conv_rate,
                    "Shorthanded Goals (T1)": home_sh_goals,
                    "Penalty Goals Pct (T1)": home_penalty_win_percentage,
                    "Faceoffs Pct (T1)": home_faceoff_win_percentage,
                    "Evenstrength Pct (T1)": home_es_connect_pct,
                    "Average Blocked (T1)": home_avg_blocked,
                    "Average Hit (T1)": home_avg_hit,
                    "Injuries (T1)": home_injuries,
                    "Team 2 (T2)": team_away,
                    "Player Stats (T2)": top_player_stats_away,
                    "Goal Keeper Stats (T2)": away_gk_season_stats,
                    "Total Goals (T2)": away_total_goals,
                    "Average Goal (T2)": away_avg_goals,
                    "Opp. Total Goals (T2)": away_opp_total_goals,
                    "Avg goals against (T2)": away_avg_goals_against,
                    "Goals Diff (T2)": away_goals_diff,
                    "Powerplay Goals (T2)": away_pp_goals,
                    "Powerplay Shots (T2)": away_pp_shots,
                    "Powerplay Conversation (T2)": away_pp_conv_rate,
                    "Shorthanded Goals (T2)": away_sh_goals,
                    "Penalty Goals Pct (T2)": away_penalty_win_percentage,
                    "Faceoffs Pct (T2)": away_faceoff_win_percentage,
                    "Evenstrength Pct (T2)": away_es_connect_pct,
                    "Average Blocked (T2)": away_avg_blocked,
                    "Average Hit (T2)": away_avg_hit,
                    "Injuries (T2)": away_injuries,
                    "spread": spread,
                    "underdog_spread_odds": underdog_spread_odds,
                    "favorite_spread_odds": favorite_spread_odds,
                    "favorite_team": favorite_team,
                    "total_over_under": total_over_under,
                    "over_total_odds": over_total_odds,
                    "under_total_odds": under_total_odds,
                    "underdog_moneyline": underdog_moneyline,
                    "favorite_moneyline": favorite_moneyline,
                    "underdog_team": underdog_team,
                    "win_pct_ml_ud": win_pct_ml_ud,
                    "win_pct_ml_f": win_pct_ml_f,
                    "win_pct_total_ud": win_pct_total_ud,
                    "win_pct_total_f": win_pct_total_ud
                }
                games_data_article.append(game_data)
                print("Article Written")
            
            os.mkdir(SAVEDIRPATH+"/NHL")
            print("NHL Folder Created!")
            try:
                self.generate_html(games_data_article)
                shutil.make_archive(SAVEDIRPATH + "/NHL", 'zip', SAVEDIRPATH+"/NHL")
                print("Success Online!")
            except:
                self.offline_html(games_data_article)            
                shutil.make_archive(SAVEDIRPATH+"/NHL", 'zip', SAVEDIRPATH+"/NHL")
                print("Success Offline!")
        else:
            print("Data not Found")



if __name__ == '__main__':
    sitpick = Sitpicks()
    nba = NBA_Article()
    # nba.nba_main()
    nhl = NHL_Article()
    # nhl.nhl_main()
    
    try:
        nba.nba_main()
    except Exception:
        print("***** EXCEPTION IN NBA *****")
        pass
    try:
        nhl.nhl_main()
    except Exception:
        print("***** EXCEPTION IN NHL *****")
        pass
    # Run job every day at specific HH:MM
    # schedule.every().day.at("13:00").do(nba.nba_main)
    # schedule.every().day.at("13:00").do(nhl.nhl_main) 
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)
