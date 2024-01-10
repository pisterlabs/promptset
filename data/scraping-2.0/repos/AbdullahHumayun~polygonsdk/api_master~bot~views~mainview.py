import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




import disnake
from menus.selectmenus.mainselect import FTDShortSelect
import requests
from sdks.polygon_sdk.async_polygon_sdk import AsyncPolygonSDK
from sdks.webull_sdk.webull_sdk import AsyncWebullSDK, thresholds
from sdks.helpers.helpers import get_checkmark, get_date_string

from sdks.polygon_sdk.async_options_sdk import PolygonOptionsSDK
from sdks.webull_sdk.forecast import ForecastEvaluator

from typing import List
from _discord import emojis
from bs4 import BeautifulSoup
from datetime import datetime
import asyncio
import pandas as pd
from sdks.fudstop_sdk.fudstop_sdk import fudstopSDK
from tabulate import tabulate
import openai
from cfg import YOUR_OPENAI_KEY, today_str
today = datetime.today()
from cfg import YOUR_API_KEY
polygon = AsyncPolygonSDK(YOUR_API_KEY)
opts = PolygonOptionsSDK(YOUR_API_KEY)
fudstop = fudstopSDK()
webull = AsyncWebullSDK()
class MainView(disnake.ui.View):
    def __init__(self, bot, ticker):
        self.ticker = ticker
        self.bot = bot
        super().__init__(timeout=None)
 
        self.add_item(FTDShortSelect(bot, ticker, self))



        self.conversation_history = {}


    def __iter__(self):
        return iter(self.children)

    def get_item(self, _id: str) -> disnake.ui.Item:
        for child in self:
            if child.custom_id == _id:
                return child
            
    async def engage_gpt4_conversation(self, interaction: disnake.ApplicationCommandInteraction):
        openai.api_key = YOUR_OPENAI_KEY
        conversation_history = {}
        conversation_id = str(interaction.user.id)
        prompt = "Start a conversation with GPT-4"
        # Retrieve the conversation history from the dictionary
        history = conversation_history.get(conversation_id, [])
        await interaction.send(f'> {emojis.redline} **GPT-4 Initializing...** {emojis.redline}')
        while True:
            # Add the new prompt to the conversation history
            history.append({"role": "user", "content": prompt})

            # Create the messages list including system message and conversation history
            messages = [
                {"role": "system", "content": f"Today is {today_str}. You are in the year 2023. Your knowledge cut-off is {today_str}."}
            ]
            messages.extend(history)

            # Generate a response based on the full conversation history
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )

            message_content = completion.choices[0].message.content

            # Store the updated conversation history in the dictionary
            conversation_history[conversation_id] = history

            embed = disnake.Embed(title=f"{emojis.redline} GPT4 {emojis.redline}", description=f"```py\n{message_content}```")
            embed.add_field(name=f"YOUR PROMPT:", value=f"```py\nYou asked: {prompt}```")

            # Send the response to the user
            await interaction.channel.send(embed=embed)
            print(message_content)

            # Check if the user wants to stop the conversation
            if prompt.lower() == "stop":
                await interaction.channel.send("Conversation ended.")
                break

            # Wait for the user's next message
            def check(m):
                return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

            try:
                user_message = await self.bot.wait_for("message", check=check, timeout=60)
            except asyncio.TimeoutError:
                await interaction.channel.send("Conversation timed out. Please start a new conversation.")
                break

            prompt = user_message.content


    @disnake.ui.button(style=disnake.ButtonStyle.blurple, custom_id="fwewe", label="GPT4")
    async def getplays(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        # Disable the button and update the message
        button.disabled = True
        await interaction.response.edit_message(view=self)

        await self.engage_gpt4_conversation(interaction)

        # Enable the button again and update the message
        button.disabled = False
        await interaction.response.edit_message(view=self)

    @disnake.ui.button(style=disnake.ButtonStyle.green, emoji=f"{emojis.redline}", label="Core Calls", row=4, custom_id="corecalls")
    async def coreputs(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        await interaction.response.defer()
        url="https://www.alphaquery.com/service/run-screen?a=59151c73bd4421857d85cb49d1f3e836a248e6cf930c2a36693d15232a4dd38c&screen=[{%22columnName%22:%22sector%22,%22operator%22:%22is%20not%22,%22value%22:%22Healthcare%22,%22valueType%22:%22%22,%22unit%22:%22%22},{%22columnName%22:%22days_since_report_date_qr0%22,%22operator%22:%22is%20less%20than%22,%22value%22:%229%22,%22valueType%22:%22number%22,%22unit%22:%22%22},{%22columnName%22:%22close_price%22,%22operator%22:%22is%20greater%20than%22,%22value%22:%225%22,%22valueType%22:%22number%22,%22unit%22:%22%22},{%22columnName%22:%22rsi_14%22,%22operator%22:%22is%20less%20than%22,%22value%22:%2230%22,%22valueType%22:%22number%22,%22unit%22:%22%22}]"
        response = requests.get(url).json()
        results = response['resultsHtml']

        soup = BeautifulSoup(results, 'html.parser')
        table_rows = soup.find_all('tr')

        data = []
        for row in table_rows[1:]:
            columns = row.find_all('td')
            ticker = columns[0].text
            sector = columns[2].text
            rsi = columns[5].text
            days_since_last_earnings = columns[3].text

            data.append([ticker, sector, rsi, days_since_last_earnings])

        df = pd.DataFrame(data, columns=['Symb', 'Sector', 'RSI', 'Days Since ER'])
        table = tabulate(df, headers='', tablefmt='fancy')
        view = MainView(self.bot,self.ticker)
        embed = disnake.Embed(title=f"{emojis.redcheck} Core Calls {emojis.redcheck}", description=f"\n**{emojis.redcheck} Symbol       {emojis.redcheck} Sector       {emojis.redcheck} RSI          {emojis.redline} Days Since ER**\n```{table}```", color=disnake.Colour.dark_green())
        await interaction.edit_original_message(embed=embed)
    @disnake.ui.button(style=disnake.ButtonStyle.red, emoji=f"{emojis.greencheck}", label=f"Core Puts", row=4, custom_id="coreputs")
    async def corecalls(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        """Returns core logic results for calls. Make sure to check for criteria"""
        await interaction.response.defer()
        url = "https://www.alphaquery.com/service/run-screen?a=59151c73bd4421857d85cb49d1f3e836a248e6cf930c2a36693d15232a4dd38c&screen=[{%22columnName%22:%22sector%22,%22operator%22:%22is%20not%22,%22value%22:%22Healthcare%22,%22valueType%22:%22%22,%22unit%22:%22%22},{%22columnName%22:%22rsi_14%22,%22operator%22:%22is%20greater%20than%22,%22value%22:%2270%22,%22valueType%22:%22number%22,%22unit%22:%22%22},{%22columnName%22:%22days_since_report_date_qr0%22,%22operator%22:%22is%20less%20than%22,%22value%22:%229%22,%22valueType%22:%22number%22,%22unit%22:%22%22}]"
        response = requests.get(url).json()
        results = response['resultsHtml']
        self.add_item(self.coreputs)
        soup = BeautifulSoup(results, 'html.parser')
        table_rows = soup.find_all('tr')

        data = []
        for row in table_rows[1:]:
            columns = row.find_all('td')
            ticker = columns[0].text
            sector = columns[2].text
            rsi = columns[3].text
            days_since_last_earnings = columns[4].text

            data.append([ticker, sector, rsi, days_since_last_earnings])
        df = pd.DataFrame(data, columns=['Symb', 'Sector', 'RSI', 'Days Since ER'])

        table = tabulate(df, headers='', tablefmt='fancy')
        view = MainView(self.bot,self.ticker)
        embed = disnake.Embed(title=f"{emojis.greencheck} Core Puts {emojis.greencheck}", description=f"\n**{emojis.greencheck} Symbol {emojis.greencheck} Sector{emojis.greencheck} RSI {emojis.greencheck} Days Since ER**\n```{table}```", color=disnake.Colour.dark_red())
        await interaction.edit_original_message(embed=embed, view=view)




    @disnake.ui.button(style=disnake.ButtonStyle.red, custom_id="indices", emoji=f"{emojis.redline}", label="All Indices", row=3)
    async def indices(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        """All indices data"""
        indices = await polygon.get_all_indices()
        file = 'files/indices/all_indices.csv'
        indices.to_csv(file)
        await interaction.send(file=disnake.File(file))


    @disnake.ui.button(style=disnake.ButtonStyle.red, custom_id="tickers", emoji=f"{emojis.redline}", label="All Stocks", row=3)
    async def stocks(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        """All indices data"""
        await interaction.response.defer(with_message=True)
        await interaction.followup.send(f'> {emojis.redline} **Gathering all stock snapshots...** {emojis.redline}')
        indices = await polygon.get()
        df = pd.DataFrame(indices)
        file = 'files\stocks\snapshots.csv'
        
        await interaction.edit_original_message(file=disnake.File(file))

    @disnake.ui.button(style=disnake.ButtonStyle.red, custom_id="alloptions", emoji=f"{emojis.redline}", row=3, label="All Options")
    async def alloptions(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        await interaction.response.defer(with_message=True)
        await interaction.followup.send(f'> {emojis.redline} **Fetching all options expiring today...** {emojis.redline}')
        options_contracts = await opts.fetch_all_option_contracts(expiration_date_gte=today_str, expiration_date_lte="2023-09-01")
        x = await opts.get_snapshots(options_contracts,output_file="all_options.csv")
        df = pd.DataFrame(x)
        
        # for i, row in df.iterrows():

        await interaction.edit_original_message(file=disnake.File('all_options.csv'))


    # @disnake.ui.button(style=disnake.ButtonStyle.blurple, custom_id="earningcal", emoji=f"{calendar}", label="Earnings Screener", row=0)
    # async def screener(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
    #     """The Earnings Screener"""
    #     embed=disnake.Embed(title=f"Earnings Screener", description=f"Select Your Earnings Info", color=disnake.Colour.random())

    #     await interaction.response.edit_message(embed=embed,view=EarningsScreenerView(self.bot, self.ticker))




    @disnake.ui.button(style=disnake.ButtonStyle.green,row=0, custom_id="financescore", emoji=f"{emojis.earnings}", label="Financial Score")
    async def financialscore(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        """The Earnings Screener"""
        await interaction.response.defer()
        await interaction.send(f'> {emojis.greencheck} Gathering financial score..** {emojis.greencheck}', delete_after=4)
        self.tik, symbol = await webull.fetch_ticker_id(self.ticker)

        analysts = await webull.get_analysis_data(self.ticker)

        strong_buy=analysts.strongbuy

        analysts.underperform


        result = await webull.financial_score(self.ticker)
        
        output = f"```py\nScore and Metrics for {self.ticker}:\n\nView ratio definitions by clicking the dropdown.```"
        for k, v in result.items():
            check = get_checkmark(k, v, thresholds)

            description = {
                'current_ratio': 'Measures company\'s short-term liquidity. A ratio of 2+ is good.',
                'quick_ratio': 'Measures company\'s immediate short-term liquidity. A ratio of 1+ is good.',
                'debt_to_equity_ratio': 'Measures company\'s debt relative to equity. Lower ratio is better.',
                'return_on_assets': 'Measures company\'s profit relative to assets. A positive ROA is good.',
                'return_on_equity': 'Measures company\'s profit relative to equity. A positive ROE is good.',
                'gross_profit_margin': 'Measures company\'s profit after cost of goods sold. A higher margin is good.',
                'operating_margin': 'Measures company\'s profit after operating expenses. A higher margin is good.',
                'net_profit_margin': 'Measures company\'s profit after all expenses. A higher margin is good.',
                'dividend_payout_ratio': 'Measures percentage of net income paid as dividends. 30-60% is good.',
                'revenue_growth': 'Measures company\'s revenue growth rate. A higher rate is good.',
                'total_debt_to_ebitda': 'Measures company\'s ability to pay off debt. Lower ratio is better.',
                'interest_coverage': 'Measures company\'s ability to pay interest on debt. Higher ratio is better.',
                'price_to_earnings_ratio': 'Measures company\'s share price relative to earnings. Lower ratio is better.',
                'price_to_sales_ratio': 'Measures company\'s share price relative to revenue. Lower ratio is better.',
                'price_to_book_value': 'Measures company\'s share price relative to book value. Lower ratio is better.',
                'cash_conversion_cycle': 'Measures company\'s ability to convert inventory to cash. Lower cycle is better.',
                'inventory_turnover': 'Measures how quickly inventory is sold. Higher turnover is better.',
                'fixed_asset_turnover': 'Measures company\'s ability to generate revenue from fixed assets. Higher turnover is better.',
                'asset_turnover': 'Measures company\'s ability to generate revenue from all assets. Higher turnover is better.',
                'debt_ratio': 'Measures company\'s total debt relative to assets. Lower ratio is better.',
                'total_asset_turnover': 'Measures company\'s ability to generate revenue from all assets. Higher turnover is better.',
                'days_sales_outstanding': 'Measures how long it takes to collect payment from customers. Lower days is better.',
                'eps_growth': 'Measures company\'s earnings per share growth rate. Positive growth is good.',
                'free_cash_flow_margin': 'Measures company\'s free cash flow relative to revenue. Higher margin is good.',
                'score': 'Total score assigned to company\'s financial health and performance. A higher score is better.'
            }
            
            if isinstance(v, float):
                output += f"{check} **{k}: {float(v):,.2f}**\n"
            elif v is None:
                output += f"{check} **{k}: N/A**\n\n"
            else:
                output += f"{check} **{k}: {float(v):,.2f}**\n"
            if result['score'] >= 16:
                color = disnake.Colour.dark_green()
            elif 10 <= result['score'] <= 16:
                color = disnake.Colour.yellow()
            elif result['score'] < 10:
                color = disnake.Colour.dark_red()

        
        embed = disnake.Embed(title=f"Financial Score", description=f"{output.format(ticker=self.ticker)}", color=color)
        embed2 = disnake.Embed(title=f"{emojis.eye} {emojis.eye}", description=f"Viewing Additional Data for {self.ticker}", color=disnake.Colour.random())
        embed2.set_thumbnail(await polygon.get_polygon_logo(symbol))
        embed2.set_footer(text=f"Click the arrow to return to the main page.")
        embed.set_thumbnail(await polygon.get_polygon_logo(symbol))
        embed.set_footer(text=f"16+ = Green | 10-15 = Yellow | 0-10 = Dogshit")

        self.ticker, symbol = await webull.fetch_ticker_id(self.ticker)
        evaluator = ForecastEvaluator(self.ticker, symbol)
        await evaluator.evaluate()
        # Send the generated images to the Discord channel



        view = disnake.ui.View(timeout=None)
        view2 = disnake.ui.View(timeout=None)
        eps_quarterly = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"EPS quarterly", emoji=f"{emojis.movingchart}", row=4)
        eps_quarterly.callback = lambda interaction: interaction.send(embed=embed, file=disnake.File('files/financials/EPS_quarterly.png'), ephemeral=True)
        view.add_item(eps_quarterly)

        revenue_quarterly = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"Revenue Quarterly", emoji=f"{emojis.movingchart}", row=4)
        revenue_quarterly.callback = lambda interaction: interaction.send(embed=embed, file=disnake.File('files/financials/Revenue_quarterly.png'), ephemeral=True)
        view.add_item(revenue_quarterly)

        roa_quarterly = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"ROA - Quarterly", emoji=f"{emojis.movingchart}", row=4)
        roa_quarterly.callback = lambda interaction: interaction.send(embed=embed, file=disnake.File('files/financials/ROA_quarterly.png'), ephemeral=True)
        view.add_item(roa_quarterly)


        roe_quarterly = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"Return on Investment - Quarterly", emoji=f"{emojis.movingchart}", row=4)
        roe_quarterly.callback = lambda interaction: interaction.send(embed=embed, file=disnake.File('files/financials/ROE_quarterly.png'), ephemeral=True)
        view.add_item(roe_quarterly)


        await interaction.edit_original_message(embed=embed)






    # @disnake.ui.button(style=disnake.ButtonStyle.grey, label=f"Company Info", emoji=f"{question}", custom_id="Cmpinfo")    
    # async def companyinfo(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
    #     info = await polygon.company_information(self.ticker)
    #     desc = info.description
    #     name = info.name
    #     list_date = info.list_date
    #     website = info.homepage_url
    #     embed = disnake.Embed(title=f"{green_magic} Data Ready {sparkly}", description=f"```py\n**{name}**:\n```py\n{desc}```", color=disnake.Colour.old_blurple())
    #     embed.add_field(name=f"List Date:", value=f"> **{list_date}**\n\n> Website: **{info.homepage_url}**")
    #     embed.add_field(name=f"Address:", value=f"> **{info.street}** **{info.city}**, **{info.state}**\n> **{info.phone_number}**")
    #     embed.add_field(name=f"Primary Exchange:", value=f"> **{info.primary_exchange}**")
    #     embed.add_field(name=f"Market:", value=f"> **{info.market}**")
        
    #     embed.add_field(name=f"Employees & Website:", value=f"> **{info.total_employees}**")
    #     embed.add_field(name=f"Codes:", value=f"> CIK#: **{info.cik}**")
    #     embed.add_field(name=f"Industry Code:", value=f"> **{info.sic_code}**\n\n> **{info.sic_description}**")
    #     embed.add_field(name=f"Market Cap & Shares:", value=f"> **${float(info.market_cap):,}**\n\n> Outstanding: **{float(info.share_class_shares_outstanding):,}**\n> Weighted: **{float(info.weighted_shares_outstanding):,}**", inline=True)
    #     await interaction.response.edit_message(embed=embed)

    @disnake.ui.button(style=disnake.ButtonStyle.blurple, label="Change Ticker", emoji=f"{emojis.inof}", row=4, custom_id="changetick")
    async def change_ticker(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        logo = await polygon.get_polygon_logo(self.ticker)
        # Set up the message and listener to ask for a new ticker
        embed = disnake.Embed(title="Change Ticker", description="Please enter the new ticker symbol in chat:",
                            color=disnake.Colour.random())
        message = await interaction.response.edit_message(embed=embed)
        message = await interaction.send(f"{emojis.inof} **Type A Ticker** {emojis.inof}", delete_after=5)
        messages = await interaction.channel.history(limit=1).flatten()
        origin_message = interaction.message
    
        def check(m):
            # Check if the message is from the user that triggered the interaction
            return m.author == interaction.user and m.channel == interaction.channel

        try:
            # Wait for the user to enter a new ticker
            message = await self.bot.wait_for('message', timeout=60.0, check=check)


        except asyncio.TimeoutError:
            pass  # You may handle the case when the user does not enter anything here
        else:
            # Update the view with the new ticker, re-run the original data command
            self.ticker = message.content.upper()

            # Delete the user's message
            await message.delete()

            # Fetch the new ticker's logo
            logo = await polygon.get_polygon_logo(self.ticker)

            # Update the OptionSelect and FTDShortSelect dropdown menus
            option_select = self.get_item('optionselect')  # Using the 'get_item' method
            option_select.ticker = self.ticker


            ftd_short_select = self.get_item('ftdshortselect')  # Using the 'get_item' method
            ftd_short_select.ticker = self.ticker
            await FTDShortSelect(self.bot, self.ticker, self).refresh_options(interaction)

            # Fetch the new ticker's logo
            logo = await polygon.get_polygon_logo(self.ticker)
            totals = await fudstop.option_market_totals()

            f2high=float(totals.fiftytwohigh)
            f2low=float(totals.fiftytwolow)
            futures=float(totals.futures_vol)
            mdaily=float(totals.monthlydailyavg)
            optvol=float(totals.optionsVol)
            ydaily=float(totals.yearlydailyavg)


            ftd_short_select = self.get_item('ftdshortselect')
            ftd_short_select.ticker = self.ticker
            await FTDShortSelect(self.bot,self.ticker,self).refresh_options(interaction)

            await origin_message.edit(embed=embed, view=self)

            embed = disnake.Embed(
                title=f"{emojis.redline} Data Ready {emojis.redline}",
                description=f"```py\nWelcome. Before you look at {self.ticker}'s data - here are the option stats on the day:```",
                color=disnake.Colour.old_blurple())
            embed.add_field(name=f"Futures Volume:", value=f"> **{futures:,}**")
            embed.add_field(name=f"Options Volume:", value=f"> {emojis.uparrow} High: **{f2high:,}**\n\n> {emojis.redline} Today: **{optvol:,}**\n\n> {emojis.redline} Low: **{f2low:,}**")
            embed.add_field(name=f"Averages:", value=f"> Monthly-Daily: **{mdaily:,}**\n> Yearly-Dayly: **{ydaily:,}**", inline=True)
            embed.set_thumbnail(logo)

            embed.set_footer(text=f"Viewing data for {self.ticker} | Implemented by FUDSTOP -")

            await origin_message.edit(embed=embed, view=self)
            await interaction.send(f"```py\n Ticker data has been updated to {self.ticker}```", delete_after=3)
            ftd_short_select.refresh_state(interaction)
            option_select.refresh_state(interaction)
            




def send_embeds(symbols):
    max_embed_description_length = 4096

    current_embed_description = ""
    embeds = []

    for entry in symbols:
        formatted_entry = f"{entry} - Report Time: postmarket\n"

        # Check if the current description plus the new entry would exceed the limit
        if len(current_embed_description) + len(formatted_entry) > max_embed_description_length:
            embed = disnake.Embed(title="Earnings Tickers", description=f"```py\n{current_embed_description}\n```", color=disnake.Color.random())
            embeds.append(embed)
            current_embed_description = ""

        current_embed_description += formatted_entry

    # Create the last embed for remaining data
    if current_embed_description:
        embed = disnake.Embed(title="Earnings Tickers", description=f"```py\n{current_embed_description}\n```", color=disnake.Color.random())
        embeds.append(embed)

    return embeds




def generate_sectors_url(sectors: List[str]) -> str:
    sectors_url = ""
    for sector in sectors:
        sectors_url += f"&sectors[]={sector}"
    return sectors_url






