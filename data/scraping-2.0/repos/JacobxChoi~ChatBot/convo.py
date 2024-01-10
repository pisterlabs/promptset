import openai
from emora_stdm import DialogueFlow

import Macros
#global
from globalState import globalState

#knowLeague
from KnowLeague.knowsLeague import knowsLeague
from KnowLeague.advanced import advanced
from KnowLeague.casual import casual

#doesnKnowLeague
from doesntKnowLeagueEsports.doesntKnowLeague import doesntKnowLeague
from doesntKnowLeagueEsports.IntroduceLeague import IntroduceLeague
from doesntKnowLeagueEsports.laneInfo import laneInfo

#macros
from Macros import MacroEsportsOrLeague, UserInputChampion, MacroGetName, MacroGetOldName, \
    MacroGetNewName, MacroPushName, favRegion, MacroGPTJSON, getFavGame, MacroNLG,getReason,getActivityWithFriends, \
    PositiveAgreement, NegativeAgreement, MacroGoalAnalogy, getRandomGame, getSportsEvent,MacroEsportAttitudeResponse, MacroGPTHAIKU, MacroFunTripError,getChampionRecommendReason, \
    PositiveAgreement, NegativeAgreement, MacroGoalAnalogy, getRandomGame, getSportsEvent,MacroEsportAttitudeResponse, MacroGPTHAIKU, MacroFunTripError, GetPlayerActivity

#imports babel conversation
import babel

from babel import babel


#convo.py imports
import pickle
import os

#knowsLeague
casual, edg, keria = casual()
advanced = advanced()
favoriteTeam, favoriteRegion = knowsLeague()

babel = babel()

#doesntKnowLeague
doesntKnowLeague, items, base, laneInfo, IntroduceLeague, IntroduceGame, IntroduceChampions, IntroduceEsports, IntroduceObjectives, ChampionRoles, SpecificTeams, SpecificPlayers, RecommendChampions, PopularChampions, ChampionTypes, ChampionRoles, StartPlaying, StartWatching = doesntKnowLeague()

#global transition
globalState = globalState()


def save(df: DialogueFlow, varfile: str):
    df.run()
    d = {k: v for k, v in df.vars().items() if not k.startswith('_')}
    pickle.dump(d, open(varfile, 'wb'))


def load(df: DialogueFlow, varfile: str):
    # has conversed before
    if os.path.isfile('resources/visits.pkl'):
        d = pickle.load(open(varfile, 'rb'))
        df.vars().update(d)
        df.run()
        # df.run(debugging=True)
        save(df, varfile)
    # first time conversing
    else:
        df.run()
        # df.run(debugging=True)
        save(df, varfile)

# This is the welcoming transition
transitions = {
    'state': 'start',
    ##Welcoming section
    '`Hi, this is LoLa, your personal chatbot for LoL esports dialogue. Could you tell me your name and a bit about yourself?`': {
        '#GET_NAME_GPT #GET_NAME': {
            '#IF(#GET_NEWNAME) `Nice to meet you,` #NAME #PlayerActivity': 'DIVERGE',
            '#IF(#GET_OLDNAME) `Welcome back!` #NAME `!`': 'DIVERGE',
            'error': {
                '`Nice to meet you!`': 'DIVERGE'
            }
        }
    },
}

# This transition distributes the users to different branches of transitions based on their acquistion levels
transitionDiverging = {
    'state': 'DIVERGE',
    '`Do you keep up with League of Legends esports? What do you think about it? `': {
        '[#AgreementChecker]': {
            '#IF(#POSITIVE_AGREEMENT) `Nice.`': 'favPlayer',
            '#IF(#NEGATIVE_AGREEMENT) `That\'s fine.`': 'doesntKnowLeagueEsports'
        },
        'error': {
            '`Sorry, I didn\'t quite understand. Do you keep up with league esports?`': 'agreeOrNot'
        }
    }
}

macros = {
    'LEAGUE': MacroEsportsOrLeague(),
    'UserChamp': UserInputChampion(),
    'NAME': MacroPushName(),
    'GET_NAME': MacroGetName(),
    'GET_NEWNAME': MacroGetNewName(),
    'GET_OLDNAME': MacroGetOldName(),
    'FAV_REGION': favRegion(),
    'FAV_GAMETYPE':MacroGPTJSON(
            'What is the game user mentioned, what is the type of the game. Give an example of the other game in the category and give one sentence answer why people love the game speaker mentioned starting with "it offers"',
            {'GameName': 'Legend of Zelda', 'GameType': 'Adventure game', 'OtherGame': 'Xenoblade Chronicles', 'WhyInterest': 'They offer a unique and immersive gameplay experience that allows players to express their creativity, engage in friendly competition, and form lasting social connections.'},
            {'GameName': 'N/A', 'GameType': 'N/A', 'OtherGame': 'N/A', 'WhyInterest': 'N/A'}
        ),
    'ACTIVITY_WITH_FRIENDS':MacroGPTJSON(
            'What does the activity the speaker typically do with friends, the activity stored should start with an verb',
            {'WithFriendActivities': 'go hiking'},
            {'WithFriendActivities': 'N/A'}
        ),
    'GET_FAV_GAME': MacroNLG(getFavGame),
    'GET_REASON_FAV_GAME': MacroNLG(getReason),
    'GET_ACTIVITY_FRIEND': MacroNLG(getActivityWithFriends),
    'SportEvents': MacroGPTJSON(
            'What is the sports event user mentioned, if the user doesn\'t like any sports event, return no "',
            {'SportEvent': 'NBA'},
            {'SportEvent': 'N/A'}
        ),
    'GET_SportsEvent': MacroNLG(getSportsEvent),

    'POSITIVE_AGREEMENT': MacroNLG(PositiveAgreement),
    'NEGATIVE_AGREEMENT': MacroNLG(NegativeAgreement),

    'AgreementChecker': MacroGPTJSON(
        'How does the speaker response to the yes or no question, give yes if user answers "yes", or shows interest , and no if user answers with "no" or is not interested ',
        {'Agreement': 'yes'},
        {'Agreement': 'no'}
    ),
    'GET_NAME_GPT': MacroGPTJSON(
        'What is the user\'s name? And what do they enjoy doing? If the user doesn\'t mention what they enjoy, just give a random habit ',
        {'FIRSTNAME': 'Jacob', 'ROLE': 'being a student'},
        {'FIRSTNAME': 'NA', 'ROLE': 'being a human'}
    ),
    # 'SET_NAME_GPT':

    'ESportAttitudeChecker': MacroGPTJSON(
        'What is the speakers\'s attitude toward esport events. Attitude options are LookingForward, Excitement, Indifference, Unwilling, Open-mindednness',
        {'EsportAttitude': 'LookingForward'},
    ),

    'ESportAttitudeResponse': MacroEsportAttitudeResponse(),

    'GameGoalAnalogy': MacroGoalAnalogy(),
    #for advanced.py
    'RANDGAME' : getRandomGame(),

    #testing global
    'GET_HAIKU': MacroGPTHAIKU(
        'Write the user a haiku in the following format:',
        {'HAIKU':'love between us is - speech and breath. loving you is - a long river running.'},
        {'HAIKU':'NA'}
    ),

    'FunTripError': MacroFunTripError(),
    'PlayerActivity': GetPlayerActivity()
}

df = DialogueFlow('start', end_state='end')
#ontology
df.knowledge_base().load_json_file('resources/teams.json')
# df.knowledge_base().load_json_file('resources/gameType.json')

#funny diversions
funny_diversions = {
    '[touch grass]': "`You think you\'re sooo funny. You are, though :). Anyways, you were saying?` (3.0)",
    '[your {mom, mother}]': "`Yeah, I thought we had an earthquake, but she was just hungry. Anyways, you were saying?` (3.0)",
    '[joke]': "pp"
}
df.load_update_rules(funny_diversions)

#doesntKnowLeague transitions
df.load_transitions(doesntKnowLeague)
df.load_transitions(transitionDiverging)
df.load_transitions(transitions)
df.load_transitions(IntroduceLeague)
df.load_transitions(laneInfo)
df.load_transitions(IntroduceGame)
df.load_transitions(IntroduceChampions)
df.load_transitions(IntroduceEsports)
df.load_transitions(IntroduceObjectives)
df.load_transitions(ChampionRoles)
df.load_transitions(SpecificTeams)
df.load_transitions(SpecificPlayers)
df.load_transitions(RecommendChampions)
df.load_transitions(PopularChampions)
df.load_transitions(ChampionRoles)
df.load_transitions(ChampionTypes)
df.load_transitions(StartPlaying)
df.load_transitions(StartWatching)

#knowsLeague transitions
df.load_transitions(favoriteTeam)
df.load_transitions(favoriteRegion)
df.load_transitions(casual)
df.load_transitions(edg)
df.load_transitions(keria)
df.load_transitions(advanced)

#global transition
df.load_global_nlu(globalState)

#babel
df.load_transitions(babel)

#macros
df.add_macros(macros)

if __name__ == '__main__':
    openai.api_key_path = Macros.OPENAI_API_KEY_PATH
    load(df, 'resources/visits.pkl')
