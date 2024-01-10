from discord import Embed
from prompts import *
import datetime

# create error embed in case of mistakes
def create_error_embed(error_message:str) -> Embed:
    embed = Embed(
        title='Error',
        description=error_message,
        color=0xff0000,
    )

    return embed

# function to handle all user commands (!)
def get_response(message:str) -> Embed:
    l_message = message.lower()

    # help
    if l_message == 'help':
        embed = Embed(
            title='Figure it out yourself huh',
            description='Here\'s a list of commands you can use:',
            color= 0x8834d8,
        )

        embed.add_field(name='!help', value='Displays this message.', inline=False)
        embed.add_field(name='!trending d', value='Displays top google searches for the day.', inline=False)
        embed.add_field(name='!trending wX', value='Weekly top searches. Must specify [1 - 4] weeks as X.', inline=False)
        embed.add_field(name='!trending mX', value='Monthly top searches. Must specify [1, 3] months as X.', inline=False)
        embed.add_field(name='!trending yXXXX', value='Yearly top searches. Must specify a year before the current year as XXXX.', inline=False)
        embed.add_field(name='!enhance', value='Enhance your AI images with a descriptive prompt from OpenAI\'s chatbot.', inline=False)

        return embed
    
    # trending
    # add another argument after the date for country later on
    # !!! weekly/monthly support dropped by google FIX !!!
        # !trending y2021 ___
        # !trending m3 ___
        # !trending w1 ___
        # !trending d ___
    if l_message.startswith('trending'):
        l_message = l_message[8:]

        # gather arguments
        args = l_message.split(' ')
        number_of_args = len(args)

        # check presence of arguments
        if number_of_args < 1:
            return create_error_embed('You must specify a time period.')
        elif number_of_args > 6:
            return create_error_embed('You must specify no more than 5 keywords.')
        
        # handle date
        full_date = args[0]
        date_type = full_date[0]

        # check if date_type is valid
        if date_type in ['y', 'm', 'w', 'd']:
            # get time period as integer
            if date_type != 'd':
                try:
                    time_period = int(full_date[1:])
                except:
                    return create_error_embed('Time period unable to be obtained')
        # if date_type not valid cancel
        else:
            return create_error_embed('You must specify a valid time period. (y,m,w,d)')

        # optional args (gather keywords)
        keywords = []
        if number_of_args > 1 and number_of_args < 6:
            for i in range(1, number_of_args):
                keywords.append(args[i])

        if len(keywords) > 0:
            create_payload(keywords)

        # call appropriate functions with error checking params
        embed_title = ""
        embed_description = ""
        # get daily trends
        if date_type == 'd':
            embed_title = "Daily Trends"
            embed_description = str(datetime.datetime.now().date())
            df = get_daily_trends()
        
        # get weekly trends
        elif date_type == 'w':
            if time_period <= 4:
                df = get_weekly_trends(weeks=time_period)
            else:
                return create_error_embed('You must specify 4 or less weeks.')
            
        # get monthly trends
        elif date_type == 'm':
            if time_period in [1, 3]:
                df = get_monthly_trends(months=time_period)
            else:
                return create_error_embed('You must specify 1 or 3 months.')
            
        # get yearly trends
        elif date_type == 'y':
            curYear = datetime.datetime.now().year
            if time_period < curYear:
                embed_title = "Yearly Trends"
                embed_description = str(time_period)
                df = get_yearly_trends(year=time_period)
            else:
                return create_error_embed('You must specify a year before the current year.')
        
        else:
            return create_error_embed('date type error')


        # create embed
        embed = Embed(
            title=embed_title,
            description=embed_description,
            color=0x00ff00,
        )

        # move data from df into embed fields
        # iterate through resulting df
        for index, row in df:
            embed.add_field(name=f'{index}. ', value=row[0], inline=True)

        return embed

    # prompt enhancer
    # if l_message starts with enhance
    if l_message.startswith('enhance'):
        l_message = l_message[8:]
        # give user options for version, testp (lifelike), ar, quality, chaos, and creative
        # then make a call to chatgpt openai
        # then return the response as embed

        
    
    return 'I didn\'t understand what you said there'