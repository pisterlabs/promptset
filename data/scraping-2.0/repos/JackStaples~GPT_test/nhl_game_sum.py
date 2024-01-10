import requests
from openai import OpenAI
from nhl_api import NHLGameLiveFeed
from game_data_extractor import GameDataExtractor
from plays.play_factory import PlayFactory
from player_summaries_manager import PlayerSummariesManager
from datetime import datetime

client = OpenAI()

game_id = "2022020001"  # Replace with the actual game ID you're interested in
nhl_feed = NHLGameLiveFeed()
game_data = nhl_feed.get_game_live_feed(game_id)
game_info = GameDataExtractor(game_data = nhl_feed.get_game_live_feed(game_id))

summaries = PlayerSummariesManager()

play_factory = PlayFactory()
plays = game_data['plays']
for play in plays:
    try:
        play_instance = play_factory.get_play_instance(play, game_info)
        player_ids = play_instance.get_involved_player_ids()  # Ensure all play classes have this method
        for player_id in player_ids:
            if (player_id is None):
                print(play_instance.format_summary())
            summaries.add_player_to_summaries(player_id)
            summaries.add_play_to_player_summary(player_id, play_instance)
    except ValueError as e:
        print(e)

def format_summary(summary):
    return f"{summary}"

def format_player_info(player_stats):
    """
    Formats player information into a string based on the given player statistics.

    Args:
        player_stats (dict): A dictionary containing player statistics.

    Returns:
        str: A formatted string containing player information.
    """

    # Extracting required data
    player_name = player_stats['firstName']['default'] + " " + player_stats['lastName']['default']
    team = player_stats['fullTeamName']['default']
    position = player_stats['position']
    birth_country = player_stats['birthCountry']
    birth_city = player_stats['birthCity']['default']
    birth_date = datetime.strptime(player_stats['birthDate'], "%Y-%m-%d")
    age = (datetime.now() - birth_date).days // 365
    height = f"{player_stats['heightInInches']} inches"
    weight = f"{player_stats['weightInPounds']} lbs"
    draft_round = player_stats['draftDetails']['round']
    draft_pick = player_stats['draftDetails']['overallPick']

    # Formatting the player information string
    player_info_str = (f"Name: {player_name}, Team: {team}, Position: {position}, "
                       f"Birth Country: {birth_country}, Birth City: {birth_city}, Age: {age}, "
                       f"Height: {height}, Weight: {weight}, Draft Round: {draft_round}, Draft Pick: {draft_pick}")

    return player_info_str


player_id = summaries.get_player_id(0)
player_stats = nhl_feed.get_player_statistics(player_id)
player_info_header = format_player_info(player_stats)

def format_current_season_stats(player_stats):
    """
    Formats a summary of a player's current season statistics.

    Args:
        player_stats (dict): A dictionary containing player statistics, including current season stats.

    Returns:
        str: A formatted summary of the player's current season statistics.
    """

    # Assuming 'player_stats' contains a 'featuredStats' key with current season data
    season_stats = player_stats['featuredStats']['regularSeason']['subSeason']

    # Extracting key statistics
    games_played = season_stats['gamesPlayed']
    goals = season_stats['goals']
    assists = season_stats['assists']
    points = season_stats['points']
    plus_minus = season_stats['plusMinus']
    shots = season_stats['shots']
    shooting_pctg = season_stats['shootingPctg']
    pim = season_stats['pim']  # Penalties in minutes

    # Formatting the current season statistics summary
    season_stats_str = (f"Games Played: {games_played}, Goals: {goals}, Assists: {assists}, Points: {points}"
                        f" Plus/Minus: {plus_minus}, Shots: {shots}, Shooting Percentage: {shooting_pctg}, PIM: {pim}")

    return season_stats_str
player_season_info = format_current_season_stats(player_stats)
print(player_season_info)

def format_career_stats(player_stats):
    """
    Formats a summary of a player's career statistics.

    Args:
        player_stats (dict): A dictionary containing player statistics, including career stats.

    Returns:
        str: A formatted summary of the player's career statistics.
    """

    # Assuming 'player_stats' contains a 'featuredStats' key with career data
    career_stats = player_stats['featuredStats']['regularSeason']['career']

    # Extracting key statistics
    games_played = career_stats['gamesPlayed']
    goals = career_stats['goals']
    assists = career_stats['assists']
    points = career_stats['points']
    plus_minus = career_stats['plusMinus']
    shots = career_stats['shots']
    shooting_pctg = career_stats['shootingPctg']
    pim = career_stats['pim']  # Penalties in minutes

    # Formatting the career statistics summary
    career_stats_str = (f"Games Played: {games_played}, Goals: {goals}, Assists: {assists}, Points: {points}"
                        f" Plus/Minus: {plus_minus}, Shots: {shots}, Shooting Percentage: {shooting_pctg}, PIM: {pim}")

    return career_stats_str

player_career_info = format_career_stats(player_stats)
print(player_career_info)

# test to see the first player in the summaries
player_id = summaries.get_player_id(0)
play_summaries = summaries.get_player_summaries(player_id)
play_summaries_str = "\n".join([format_summary(summary) for summary in play_summaries])
summary_string = f"{player_info_header}\nPlays from this game:\n{play_summaries_str}\nCurrent season totals: {player_season_info}\nCareer totals: {player_career_info}"

print(summary_string)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a professional hockey columnist, who has been working as a beat writer for many years now. You take the basic statistics from a player for a given game, and write an article about their performance. Heavier emphasis should be given to shots and goals. You will receive a tip based on the quality of work, so make sure it is good!"},
    {"role": "user", "content": summary_string}
  ]
)


print(completion.choices[0].message)



