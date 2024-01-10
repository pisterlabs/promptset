import os
import sqlite3
# # # from langchain import OpenAI, SQLDatabase
# # from langchain.agents import create_sql_agent, AgentType
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import json
from dotenv import load_dotenv
from .webscraper import scrape_fighter_details

load_dotenv()

# dburi = os.getenv("DBURI")
# db = SQLDatabase.from_uri(dburi)

# llm = OpenAI(temperature=0)

# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# sql_agent = create_sql_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# )


current_dir = os.path.dirname(os.path.abspath(__file__))

db_path = os.path.join(current_dir, 'database', 'ufc_data.db')

def retrieve_fighter_previous_fights(fighter_name):

    with sqlite3.connect(database=db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f'''SELECT "Fighter1", "Fighter2", "Result", "Method", "Round", "Strikes_Fighter1", "Strikes_Fighter2", "TD_Fighter1", "TD_Fighter2", "Sub_Fighter1", "Sub_Fighter2" FROM ufc_event_data WHERE "Fighter1" = ? OR "Fighter2" = ?;''', (fighter_name, fighter_name))
        rows = cursor.fetchall()
        # Reset counters and lists to track data
        wins = 0
        losses = 0
        draws = 0
        rounds_won = []
        rounds_lost = []
        total_strikes_jones = 0
        total_strikes_opponents = 0
        total_takedowns_jones = 0
        total_takedowns_opponents = 0
        total_subs_jones = 0
        total_subs_opponents = 0

        # Re-process the data with the updated logic for determining the outcome
        for row in rows:
            fighter1, fighter2, result, method, round, strikes_fighter1, strikes_fighter2, td_fighter1, td_fighter2, sub_fighter1, sub_fighter2 = row
            
            # Convert the data to integers where necessary
            strikes_fighter1 = int(strikes_fighter1) if isinstance(strikes_fighter1, str) and strikes_fighter1.isdigit() else strikes_fighter1
            strikes_fighter2 = int(strikes_fighter2) if isinstance(strikes_fighter2, str) and strikes_fighter2.isdigit() else strikes_fighter2
            td_fighter1 = int(td_fighter1) if isinstance(td_fighter1, str) and td_fighter1.isdigit() else td_fighter1
            td_fighter2 = int(td_fighter2) if isinstance(td_fighter2, str) and td_fighter2.isdigit() else td_fighter2
            sub_fighter1 = int(sub_fighter1) if isinstance(sub_fighter1, str) and sub_fighter1.isdigit() else sub_fighter1
            sub_fighter2 = int(sub_fighter2) if isinstance(sub_fighter2, str) and sub_fighter2.isdigit() else sub_fighter2
            
            # Determine the outcome of the fight for the fighter
            if result == fighter_name:
                wins += 1
                rounds_won.append(round)
            elif result == "Draw":
                draws += 1
            else:
                losses += 1
                rounds_lost.append(round)

            # Calculate strikes, takedowns, and submissions as before
            if fighter1 == fighter_name:
                total_strikes_jones += strikes_fighter1
                total_strikes_opponents += strikes_fighter2
                total_takedowns_jones += td_fighter1
                total_takedowns_opponents += td_fighter2
                total_subs_jones += sub_fighter1
                total_subs_opponents += sub_fighter2
            else:
                total_strikes_jones += strikes_fighter2
                total_strikes_opponents += strikes_fighter1
                total_takedowns_jones += td_fighter2
                total_takedowns_opponents += td_fighter1
                total_subs_jones += sub_fighter2
                total_subs_opponents += sub_fighter1

        # Compute the averages and win percentage
        total_fights = wins + losses + draws
        avg_round_won = sum(rounds_won) / len(rounds_won) if rounds_won else None
        avg_round_lost = sum(rounds_lost) / len(rounds_lost) if rounds_lost else None
        win_percentage = (wins / total_fights) * 100 if total_fights != 0 else 0

        data = {
            "name": fighter_name,
            "strikesLanded": total_strikes_jones - total_strikes_opponents,
            "takedownsLanded": total_takedowns_jones - total_takedowns_opponents,
            "submissionsLanded": total_subs_jones - total_subs_opponents,
            "averageRoundWon": avg_round_won,
            "averageRoundLost": avg_round_lost,
            "fighterStats": retrieve_fighter(fighter_name)
        }

        return data

def retrieve_fighter(fighter_name):
    with sqlite3.connect(database=db_path) as conn:
        try:
            cursor = conn.cursor()
            cursor.execute(f'SELECT * from ufc_fighter_data WHERE "Name" = ?;', (fighter_name,))
            rows = cursor.fetchall()
            row = rows[0]
            print(rows)
            first_name = row[0]
            last_name = row[1]
            nickname = row[2]
            height = row[3]
            weight = row[4]
            reach = row[5]
            stance = row[6]
            wins = row[7]
            losses = row[8]
            draws = row[9]
            image_url = scrape_fighter_details(fighter_name)["image_url"]
            returnObj = {
                "name": f'{first_name} {last_name}',
                "image_url": image_url,
                "nickname": nickname,
                "height": height,
                "weight": weight,
                "reach": reach,
                "stance": stance,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "winpercentage": f"{(wins / (wins + losses + draws)) * 100:.2f}%"
            } 

            return returnObj
        except:
            return None

def get_all_unique_fighter_names():
    with sqlite3.connect(database=db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT "Fighter1" FROM ufc_event_data UNION SELECT DISTINCT "Fighter2" FROM ufc_event_data')
        rows = cursor.fetchall()
        fighter_names = [{"value": row[0], "label": row[0]} for row in rows]

        with open("fighter_names.json", "w") as f:
            json.dump(fighter_names, f)

get_all_unique_fighter_names()

# # print(retrieve_fighter("Conor McGregor"))
# print(retrieve_fighter("Khabib Nurmagomedov"))
# print(retrieve_fighter_previous_fights("Jon Jones"))
# print(retrieve_fighter_previous_fights("Conor McGregor"))
# sql_agent.run('Take in account number of strikes landed, number of takedowns landed, and number of submission attempts from all previous fights including the fighter Jon Jones and provide an analysis on him. For example if the fighter landed 100 strikes and his opponent landed 50 mark him down as +50. Provide the data in a structured json object of the form: {"name":"___","strikeslanded":"___" ...} where there are underlines insert the data calculated')