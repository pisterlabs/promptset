import json
import random
import re
from typing import Dict, Any, List, Callable, Pattern
from json import JSONDecodeError

import spacy
from emora_stdm import Macro, Ngrams
import openai
import regexutils

OPENAI_API_KEY_PATH = 'resources/openai_api.txt'
CHATGPT_MODEL = 'gpt-3.5-turbo'

class MacroGetName(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        #TODO: CHANGE THIS TO CHATGPT BASED
        # r = re.compile(
        #     r"(?:(?:hi)?(?:\,)?(?:\s)*my(?:\s)*(?:name|nickname)(?:\s)*(?:is)?(?:\s)*|(?:hi)?(?:\,)?(?:\s)*i(?:\s)*am(?:\s)*|(?:please|you(?:\s)*can|everyone)?(?:\s)*(?:call|calls)(?:\s)*me(?:\s)*?|(?:hi)?(?:\,)?(?:\s)*i(?:\')?(?:m)?(?:ts)?(?:t\'s)?(?:\s)*(?:go by)?)?(?:\s)*(mr|mrs|ms|dr|dr\.)?(?:^|\s)*([a-z']+)(?:\s([a-z']+))?(?:(?:\,)?(?:\s)*.*)?")
        #
        # title, firstname, lastname = None, None, None
        #
        # for m in re.finditer(r, ngrams.text()):
        #     if m.group(1) is not None:
        #         title = m.group(1)
        #     if m.group(2) is not None:
        #         firstname = m.group(2)
        #     if m.group(3) is not None:
        #         lastname = m.group(3)
        #
        # if title is None and firstname is None and lastname is None:
        #     return False
        #
        # vars['TITLE'] = title
        # vars['LASTNAME'] = lastname
        vn_FN = 'FIRSTNAME'
        vn_PI = 'PLAYERINFO'

        if vn_FN not in vars:
            pass
        else:
            firstname = vars[vn_FN]
            vn_firstname = firstname.capitalize()

        #if 'FIRSTNAME' var isn't in vars
        # if vn_FN not in vars:
        #     vars[vn_FN] = firstname
        #     vars[vn_firstname] = False

        if vn_PI not in vars:
            vars[vn_PI] = {}
            vars[vn_PI][vn_firstname] = {}
            vars[vn_firstname] = False
            return True

        #if vn_firstname (their actual name) isn't in vars['FIRSTNAME']
        # if vn_firstname not in vars['FIRSTNAME']:
        if vn_firstname not in vars[vn_PI]:
            vars['FIRSTNAME'] = vn_firstname
            vars[vn_PI][vn_firstname] = {}
            vars[vn_firstname] = False
        else:
            vars[vn_firstname] = True

        # add dictionary to gather info about user

        return True

class MacroGetOldName(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        vn = vars['FIRSTNAME']
        return vars[vn]

class MacroGetNewName(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        vn = vars['FIRSTNAME']
        return not vars[vn]

class MacroPushName(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        return vars['FIRSTNAME'] + '.'

class GetPlayerActivity(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        role = 'ROLE'
        if 'NA' in vars[role] or 'enjoyment not mentioned' in vars[role] or 'unknown' in vars[role] or 'unknown (no mention of interests)' in vars[role]:
            return
        else:
            return 'That\'s really cool!'

class MacroEsportsOrLeague(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        r = re.compile(r"(dont.*play(?:ing)*)|([lL]eague(?:\s)*(?:[oO]f [lL]egend(?:s)?)?)?")
        # m = r.search(ngrams.text())
        hasLeague = False
        for m in re.finditer(r, ngrams.text()):
            if m.group(1) or m.group(2) is not None:
                hasLeague = True
        return hasLeague

class UserInfo(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        # variables used in conversation so far.
        # TODO We don't need to intialize the variables here, but it might be good to keep track of which variables we use
        visits = 'VISITS'
        player = 'FAV_PLAYER'
        playerRec = 'PLAYER_RECOMMEND'
        champ = 'FAV_CHAMP'
        vars[champ] = ''
        vars[player] = ''
        vars[playerRec] = ''
        vars[visits] = 1

class favRegion(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        userResponse = ngrams.text()

        # opens json
        f = open('resources/tourneys.json', )
        data = json.load(f)

        # gets user input
        mystr = ngrams.text().split()

        # different ways a user can reference the regions
        regions = {'na': 'NORTH AMERICA',
                   'north': 'NORTH AMERICA',
                   'america': 'NORTH AMERICA',
                   'north american': 'NORTH AMERICA',
                   'lcs': 'NORTH AMERICA',
                   'nacl': 'NORTH AMERICA',
                   'kr': 'KOREA',
                   'korea': 'KOREA',
                   'lck': 'KOREA',
                   'emea': 'EMEA',
                   'lec': 'EMEA',
                   'tcl': 'EMEA',
                   'latam': 'LATIN AMERICA',
                   'lla': 'LATIN AMERICA',
                   'hong kong': 'HONG KONG, MACAU, TAIWAN',
                   'macau': 'HONG KONG, MACAU, TAIWAN',
                   'taiwan': 'HONG KONG, MACAU, TAIWAN',
                   'pcs': 'HONG KONG, MACAU, TAIWAN',
                   'cis': 'COMMONWEALTH OF INDEPENDENT STATES',
                   'lcl': 'COMMONWEALTH OF INDEPENDENT STATES',
                   'tr': 'TURKEY',
                   'turkey': 'TURKEY',
                   'vt': 'VIETNAM',
                   'vcs': 'VIETNAM',
                   'vietnam': 'VIETNAM',
                   'oc': 'OCEANIA',
                   'oceania': 'OCEANIA',
                   'lco': 'OCEANIA',
                   'international': 'INTERNATIONAL',
                   'br': 'BRAZIL',
                   'brazil': 'BRAZIL',
                   'cblol': 'BRAZIL',
                   'cn': 'CHINA',
                   'china': 'CHINA',
                   'lpl': 'CHINA',
                   'jp': 'JAPAN',
                   'japan': 'JAPAN',
                   'japan': 'JAPAN',
                   'ljl': 'JAPAN'
                }

        # labeled T because they're temporary variables that are not meant to be stored.

        t_tourney = 'T_TOURNEY'
        t_typeOfMatch = 'T_MATCH'
        team1 = 'T_TEAM1'
        team2 = 'T_TEAM2'
        winner = 'T_WINNER'
        loser = 'T_LOSER'
        t_date = 'T_DATE'
        winner_code = 'WINNER_CODE'
        loser_code = 'LOSER_CODE'

        # t_month = 'T_MONTH'
        # t_day = 'T_DAY'

        vars[t_tourney] = ''
        vars[t_typeOfMatch] = ''
        vars[team1] = ''
        vars[team2] = ''
        vars[winner] = ''
        vars[winner_code] = ''
        vars[loser_code] = ''
        # vars[t_day] = ''
        # vars[t_month] = ''

        # region local variable
        region = ''

        # sees if nouns match region dictionary and retrieves region
        for word in mystr:
            if word.lower() in regions:
                region = regions[word.lower()]

        # no region found. Return false
        if region == '':
            return False
        else: #ADDS FAVORITE REGION TO PLAYERINFO
            vn_PI = 'PLAYERINFO'
            vn_FN = 'FIRSTNAME'
            vn_FR = 'FAV_REGION'
            vars[vn_PI][vars[vn_FN]][vn_FR] = region

        # some regions don't have any games from this year so far. If this is the case, return false
        if (len(data['ontology'][region]) >= 1):
            #picks random tourney from region
            tourney = data['ontology'][region]
            tourney = tourney[random.randrange(len(tourney))]
            hasTourney = True
            noTourney = 0
            #if a regional tourney is empty, select another tourney
            while(hasTourney):
                noTourney +=1
                if tourney not in data['ontology']:
                    tourney = data['ontology'][region]
                    tourney = tourney[random.randrange(len(tourney))]
                else:
                    hasTourney = False
                if noTourney >= 17:
                    print("NO GAMES IN TOURNEY")
                    return False

            #stores tourney into vars
            vars[t_tourney] = tourney.replace('_', ' ')
        else:
            return False

        # pulling game info from ontology. Last index -1 means most recent game. LOLA should remember which game was suggested
        game = data['ontology'][tourney]
        game = data['ontology'][tourney][random.randrange(len(game))]

        #storing suggested game to personal info
        vn_GS = 'GAME_SUGGESTED'
        #if user already has games suggested to them
        if vn_GS in vars[vn_PI][vars[vn_FN]]:
            vars[vn_PI][vars[vn_FN]][vn_GS].append(game)
        #user has not yet had games suggested to them
        else:
            vars[vn_PI][vars[vn_FN]][vn_GS] = []
            vars[vn_PI][vars[vn_FN]][vn_GS].append(game)

        #update variables to get random game
        typeOfMatch = game['week']
        vars[team1] = game['teams'][0]
        vars[team2] = game['teams'][1]
        vars[winner] = game['winner']
        date = game['time'][0:10]
        month = date[5:7]
        day = date[-2:]
        year = date[0:4]
        # adds date to vars
        vars[t_date] = month + '/' + day + '/' + year

        # print(vars)

        #gets winners and loser
        if vars[winner] == game['teams'][1]:
            vars[loser] = game['teams'][0]
            vars[loser_code] = game['teamCodes'][0]
            vars[winner_code] = game['teamCodes'][1]
        else:
            vars[loser] = game['teams'][1]
            vars[loser_code] = game['teamCodes'][1]
            vars[winner_code] = game['teamCodes'][0]

        # playoffs
        if typeOfMatch[0:8] == 'Playoffs':
            vars[t_typeOfMatch] = typeOfMatch[-7:].lower() + " " + typeOfMatch[0:8].lower()
        # knockout or weekly games
        else:
            vars[t_typeOfMatch] = typeOfMatch.lower()

        # change numerical month to month name
        # if month == '01':
        #     vars[t_month] = 'January'
        # elif month == '02':
        #     vars[t_month] = 'February'
        # elif month == '03':
        #     vars[t_month] = 'March'
        # elif month == '04':
        #     vars[t_month] = 'April'
        #
        # # rd, st, th for days
        # if day[-1:] == '2' or day[-1:] == '3':
        #     vars[t_day] = day + "rd"
        # elif day[-1:] == 1:
        #     vars[t_day] = day + "st"
        # else:
        #     vars[t_day] = day + "th"

        return True

class getRandomGame(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):

        print(ngrams)

        if ngrams == None or 'yes' in ngrams or 'yeah' in ngrams:
            f = open('resources/tourneys.json', )
            data = json.load(f)

            #vars for game
            t_tourney = 'T_TOURNEY'
            t_typeOfMatch = 'T_MATCH'
            team1 = 'T_TEAM1'
            team2 = 'T_TEAM2'
            winner = 'T_WINNER'
            loser = 'T_LOSER'
            t_month = 'T_MONTH'
            t_day = 'T_DAY'

            #vars for playerinfo
            vn_PI = 'PLAYERINFO'
            vn_FN = 'FIRSTNAME'
            vn_FR = 'FAV_REGION'
            region = vars[vn_PI][vars[vn_FN]][vn_FR]

            if (len(data['ontology'][region]) >= 1):
                # picks random tourney from region
                tourney = data['ontology'][region]
                tourney = tourney[random.randrange(len(tourney))]
                hasTourney = True
                noTourney = 0
                # if a regional tourney is empty, select another tourney
                while (hasTourney):
                    noTourney += 1
                    if tourney not in data['ontology']:
                        tourney = data['ontology'][region]
                        tourney = tourney[random.randrange(len(tourney))]
                    else:
                        hasTourney = False
                    # TODO: handle case where there are tourneys, but no games
                    if noTourney >= 17:
                        print("NO GAMES IN TOURNEY")
                        return False

                # stores tourney into vars
                vars[t_tourney] = tourney.replace('_', ' ')
            else:
                print("REGION HAS NO GAMES")
                return False

            # pulling game info from ontology. Last index -1 means most recent game. LOLA should remember which game was suggested
            game = data['ontology'][tourney]
            game = data['ontology'][tourney][random.randrange(len(game))]

            # storing suggested game to personal info
            vn_GS = 'GAME_SUGGESTED'
            # if user already has games suggested to them
            if vn_GS in vars[vn_PI][vars[vn_FN]]:
                vars[vn_PI][vars[vn_FN]][vn_GS].append(game)
            # user has not yet had games suggested to them
            else:
                vars[vn_PI][vars[vn_FN]][vn_GS] = []
                vars[vn_PI][vars[vn_FN]][vn_GS].append(game)

            # update variables to get random game
            typeOfMatch = game['week']
            vars[team1] = game['teams'][0]
            vars[team2] = game['teams'][1]
            vars[winner] = game['winner']
            date = game['time'][0:10]
            month = date[5:7]
            day = date[-2:]

            # gets winners and loser
            if vars[winner] == game['teams'][1]:
                vars[loser] = game['teams'][0]
            else:
                vars[loser] = game['teams'][1]

            # playoffs
            if typeOfMatch[0:8] == 'Playoffs':
                vars[t_typeOfMatch] = typeOfMatch[-7:].lower() + " " + typeOfMatch[0:8].lower()
            # knockout or weekly games
            else:
                vars[t_typeOfMatch] = typeOfMatch.lower()

            # change numerical month to month name
            if month == '01':
                vars[t_month] = 'January'
            elif month == '02':
                vars[t_month] = 'February'
            elif month == '03':
                vars[t_month] = 'March'
            elif month == '04':
                vars[t_month] = 'April'

            # rd, st, th for days
            if day[-1:] == '2' or day[-1:] == '3':
                vars[t_day] = day + "rd"
            elif day[-1:] == 1:
                vars[t_day] = day + "st"
            else:
                vars[t_day] = day + "th"
            return True
        else:
            return False

class UserInputChampion(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):

        # variables
        playerRec = 'PLAYER_RECOMMEND'
        fav_champ = 'FAV_CHAMP'
        vn_PI = 'PLAYERINFO'
        vn_FN = 'FIRSTNAME'

        # opening jason
        f = open('resources/champs.json', )
        data = json.load(f)
        mystr = ngrams.text()
        # takes user input as string
        champs = {
            'k\'sante':'ksante',
            'cho gath':'chogath',
            'cho\'gath':'chogath',
            'lee sin': 'leesin',
            'jarvan':'jarvaniv',
            'jarvan iv':'jarvaniv',
            'dr mundo':'drmundo',
            'dr. mundo': 'drmundo',
            'tahm kench':'tahmkench',
            'xin zhao':'xinzhao',
            'bel\'veth':'belveth',
            'bel veth':'belveth',
            'kha zix':'khazix',
            'kha\'zix': 'khazix',
            'master yi':'masteryi',
            'rek\'sai':'reksai',
            'rek sai': 'reksai',
            'le\'blanc':'leblanc',
            'le blanc': 'leblanc',
            'aurelion sol':'aurelionsol',
            'vel\'koz':'velkoz',
            'vel koz': 'velkoz',
            'twisted fate':'twistedfate',
            'kog maw':'kogmaw',
            'kog\'maw': 'kogmaw',
            'miss fortune':'missfortune'
        }

        for key in champs:
            if key in mystr.lower():
                #user's favorite champion stored as temp variable
                vars[fav_champ] = champs[key].capitalize()
                #if user already has said their favorite champion
                if fav_champ in vars[vn_PI][vars[vn_FN]]:
                    pass
                #user has not yet said their favorite champion
                else:
                    vars[vn_PI][vars[vn_FN]][fav_champ] = champs[key].capitalize()
                # grabs player that plays this champion
                player = data['ontology'][champs[key]][random.randrange(len(data['ontology'][champs[key]]))]

                #playerRec has alreayd been made
                if playerRec in vars[vn_PI][vars[vn_FN]]:
                    # if player has already been suggested previously
                    if player in vars[vn_PI][vars[vn_FN]][playerRec]:
                        diffPlayer = data['ontology'][champs[key]][random.randrange(len(data['ontology'][champs[key]]))]
                        vars[vn_PI][vars[vn_FN]][playerRec].append(diffPlayer)
                        vars[playerRec] = diffPlayer
                        return True
                    #player has not yet been suggested
                    else:
                        vars[vn_PI][vars[vn_FN]][playerRec].append(player)
                        vars[playerRec] = player
                        return True
                #player has not yet been suggested
                else:
                    vars[vn_PI][vars[vn_FN]][playerRec] = []
                    newPlayer = data['ontology'][champs[key]][random.randrange(len(data['ontology'][champs[key]]))]
                    vars[vn_PI][vars[vn_FN]][playerRec].append(newPlayer)
                    vars[playerRec] = newPlayer
                    return True

        mystr = ngrams.text().split()
        #iterates through player text
        for word in mystr:
            #if champion in ontology
            if word.lower() in data['ontology']:
                # user's favorite champion stored as temp variable
                vars[fav_champ] = word.capitalize()
                # if user already has said their favorite champion
                if fav_champ in vars[vn_PI][vars[vn_FN]]:
                    pass
                # user has not yet said their favorite champion
                else:
                    vars[vn_PI][vars[vn_FN]][fav_champ] = word.capitalize()
                # grabs player that plays this champion
                player = data['ontology'][word.lower()][random.randrange(len(data['ontology'][word.lower()]))]

                # playerRec has alreayd been made
                if playerRec in vars[vn_PI][vars[vn_FN]]:
                    # if player has already been suggested previously
                    if player in vars[vn_PI][vars[vn_FN]][playerRec]:
                        diffPlayer = data['ontology'][word.lower()][random.randrange(len(data['ontology'][word.lower()]))]
                        vars[vn_PI][vars[vn_FN]][playerRec].append(diffPlayer)
                        vars[playerRec] = diffPlayer
                        return True
                    # player has not yet been suggested
                    else:
                        vars[vn_PI][vars[vn_FN]][playerRec].append(player)
                        vars[playerRec] = player
                        return True
                # player has not yet been suggested
                else:
                    vars[vn_PI][vars[vn_FN]][playerRec] = []
                    newPlayer = data['ontology'][word.lower()][random.randrange(len(data['ontology'][word.lower()]))]
                    vars[vn_PI][vars[vn_FN]][playerRec].append(newPlayer)
                    vars[playerRec] = newPlayer
                    return True

        return False

class MacroGPTJSON(Macro):
    def __init__(self, request: str, full_ex: Dict[str, Any], empty_ex: Dict[str, Any] = None,
                 set_variables: Callable[[Dict[str, Any], Dict[str, Any]], None] = None) -> object:
        """
        :rtype: object
        :param request: the task to be requested regarding the user input (e.g., How does the speaker want to be called?).
        :param full_ex: the example output where all values are filled (e.g., {"call_names": ["Mike", "Michael"]}).
        :param empty_ex: the example output where all collections are empty (e.g., {"call_names": []}).
        :param set_variables: it is a function that takes the STDM variable dictionary and the JSON output dictionary and sets necessary variables.
        """
        self.request = request
        self.full_ex = json.dumps(full_ex)
        self.empty_ex = '' if empty_ex is None else json.dumps(empty_ex)
        self.check = re.compile(regexutils.generate(full_ex))
        self.set_variables = set_variables

    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        examples = f'{self.full_ex} or {self.empty_ex} if unavailable' if self.empty_ex else self.full_ex
        prompt = f'{self.request} Respond in the JSON schema such as {examples}: {ngrams.raw_text().strip()}'
        output = gpt_completion(prompt)
        # print(output)
        if not output: return False

        try:
            d = json.loads(output)
        except JSONDecodeError:
            return False

        if self.set_variables:
            self.set_variables(vars, d)
        else:
            vars.update(d)
        return True


class MacroNLG(Macro):
    def __init__(self, generate: Callable[[Dict[str, Any]], str]):
        self.generate = generate

    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        return self.generate(vars)


def gpt_completion(input: str, regex: Pattern = None) -> str:
    response = openai.ChatCompletion.create(
        model=CHATGPT_MODEL,
        messages=[{'role': 'user', 'content': input}]
    )
    output = response['choices'][0]['message']['content'].strip()

    if regex is not None:
        m = regex.search(output)
        output = m.group().strip() if m else None

    return output

#Section: casual communication: What's your favorite game?

def getFavGame(vars: Dict[str, Any]):
    return vars['GameType']


def getReason(vars: Dict[str, Any]):
    return vars['WhyInterest']
def getActivityWithFriends(vars: Dict[str, Any]):
    return vars['WithFriendActivities']

def getSportsEvent(vars: Dict[str, Any]):
    return vars['SportEvent']

def PositiveAgreement(vars: Dict[str, Any]):
    if vars['Agreement'] == 'yes':
        return True
    else:
        return False

def NegativeAgreement(vars: Dict[str, Any]):
    if vars['Agreement'] == 'no':
        return True
    else:
        return False

class MacroGPTHAIKU(Macro):
    def __init__(self, request: str, full_ex: Dict[str, Any], empty_ex: Dict[str, Any] = None,
                 set_variables: Callable[[Dict[str, Any], Dict[str, Any]], None] = None) -> object:
        """
        :rtype: object
        :param request: the task to be requested regarding the user input (e.g., How does the speaker want to be called?).
        :param full_ex: the example output where all values are filled (e.g., {"call_names": ["Mike", "Michael"]}).
        :param empty_ex: the example output where all collections are empty (e.g., {"call_names": []}).
        :param set_variables: it is a function that takes the STDM variable dictionary and the JSON output dictionary and sets necessary variables.
        """
        self.request = request
        self.full_ex = json.dumps(full_ex)
        self.empty_ex = '' if empty_ex is None else json.dumps(empty_ex)
        self.check = re.compile(regexutils.generate(full_ex))
        self.set_variables = set_variables

    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        examples = f'{self.full_ex} or {self.empty_ex} if unavailable' if self.empty_ex else self.full_ex
        prompt = f'{self.request} Respond in the JSON schema such as {examples}'
        output = gpt_completion(prompt)
        # print(output)
        if not output: return False

        try:
            d = json.loads(output)
        except JSONDecodeError:
            return False

        if self.set_variables:
            self.set_variables(vars, d)
        else:
            vars.update(d)
        return True

# This macro use analogy to explain the game goal according to the favorite game user select
class MacroGoalAnalogy(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        vn = 'GameType'
        GameTypeResponseDic = {
            'RPG': {
                'GameGoalSim': 'You can conceive it as a 30 min Mini role play game with real person friends and foes, \n where each hero has different sets of power to achieve the ultimate goal: destroy the enemy nexus with your teammates. \n Along the process, side missions are seizing the available resources, ambushing the enemy champions, and getting buffs \n from neutral monsters, upgrading the weapons and defense... '
            },

            'shooter': {
                'GameGoalSim': 'It\'s really similar to the shooter games like CS or Overwatch, as teammates cooperate \n in a competitive environment to achieve the ultimate goal. In here, it\'s destroying the turrets and finally the enemy nexus  '
            },

            'towerDefense': {
                'GameGoalSim': 'You can regard it as the tower defense game where you shall protect your turrets and bases from enemy attack. (ゝ∀･).'
            },

            'other': {
                'GameGoalSim': 'The main goal of league of legends is to destroy the other team\'s base. And, of course, there are many obstacles on the way to final goal and side missions.'
            }
        }

        if vn not in vars:
            vars[vn] = 'Role play Game'



        if vars[vn] == 'First-person shooter ':
            return GameTypeResponseDic['shooter']['GameGoalSim']

        if vars[vn] == 'Tower defense':
            return GameTypeResponseDic['towerDefense']['GameGoalSim']

        if vars[vn] == 'Role play Game':
            return GameTypeResponseDic['RPG']['GameGoalSim']
        else:
            return GameTypeResponseDic['RPG']['GameGoalSim']

class MacroEsportAttitudeResponse(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        vn = 'EsportAttitude'

        EsportAttitudeResponseDic = {
            'LookingForward': {
                'EsportAttitudeSim': 'It\'s definitely worth a try if you want to meet more friends with same interest !'
            },

            'Excitement': {
                'EsportAttitudeSim': 'ヽ(゜▽゜ )－C<(/;◇;)/~,'
            },

            'Unwilling': {
                'EsportAttitudeSim': '(｡í _ ì｡), '
            },

            'Open-mindedness': {
                'EsportAttitudeSim': 'I agree with you. Embracing new things is definitely a joy in the life ! '
            },

            'Indifference':{
                'EsportAttitudeSim': 'That\'s fine. After all, one fun thing to be a person is each of us has different interest. '
            },


            'other': {
                'EsportAttitudeSim': '(´～`),'
            }

        }

        if vars[vn] == 'LookingForward':
            return EsportAttitudeResponseDic['LookingForward']['EsportAttitudeSim']

        if vars[vn] == 'Excitement':
            return EsportAttitudeResponseDic['Excitement']['EsportAttitudeSim']

        if vars[vn] == 'Unwilling':
            return EsportAttitudeResponseDic['Unwilling']['EsportAttitudeSim']

        if vars[vn] == 'Open-mindedness':
            return EsportAttitudeResponseDic['Open-mindedness']['EsportAttitudeSim']

        else:
            return EsportAttitudeResponseDic['other']['EsportAttitudeSim']


class MacroFunTripError(Macro):
    def run(self, ngrams: Ngrams, vars: Dict[str, Any], args: List[Any]):
        n = random.randint(0, 4)

        ErrorDic = {
            1: {"error": 'I\'m sorry, we have to stay quiet around the fierce creatures, what do you want to know, the origin of the creature or power it beholds, \nor you want to continue the journey to behold more mysterious creatures'
                },
            2: {"error": 'Don\'t move further, it notice us. Let us first moves further from those irrational monsters. By the way, do you have other questions \n or you just want to move away from the burning place'
                },
            3: {"error": 'My apologies, my attentions are completely drawn away from those adorable monsters. Do you see their scales glittering under the sunshine. \n That\'s definitely breathetaking. By the way, do you have questions over the origin of those creatures. \n If you want to move to another place, just let me know'
                },
            4: {"error": 'The wind is too loud there at rift, I didn\t hear you much just then. Could you repeat your questions or you want to continue our trip to visit other monsters'
                },
            5: {"error": 'We seemed to rushed into its territory. Let\'s first get rid of there... What questions did you have or you just want to leave'
                },
        }

        match n:
            case 0:
                return ErrorDic[1]['error']
            case 1:
                return ErrorDic[2]['error']
            case 2:
                return ErrorDic[3]['error']
            case 3:
                return ErrorDic[4]['error']
            case 4:
                return ErrorDic[5]['error']

        return True

def getChampionRecommendReason(vars: Dict[str, Any]):
    print (vars['RecommendedChampion'])
    return vars['ChampionRecommendReason']



