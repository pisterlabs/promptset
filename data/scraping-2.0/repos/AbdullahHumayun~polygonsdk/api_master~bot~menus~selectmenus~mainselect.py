import disnake
import pandas as pd
from sdks.webull_sdk.forecast import ForecastEvaluator
import openai
from sdks.polygon_sdk.get_all_options import fetch_all_option_contracts
from tabulate import tabulate
from cfg import today_str, YOUR_OPENAI_KEY, YOUR_API_KEY
from datetime import datetime
from sdks.webull_sdk.webull_sdk import thresholds
from sdks.polygon_sdk.async_polygon_sdk import AsyncPolygonSDK
from sdks.helpers.helpers import get_checkmark
from bs4 import BeautifulSoup
from menus.embedmenus import AlertMenus
import pandas as pd
from sdks.polygon_sdk.async_options_sdk import PolygonOptionsSDK
import disnake
from sdks.webull_sdk.webull_sdk import AsyncWebullSDK
webull = AsyncWebullSDK()
import requests
import disnake
import requests
from sdks.webull_sdk.webull_sdk import AsyncWebullSDK
from cfg import today_str
from sdks.polygon_sdk.async_polygon_sdk import AsyncPolygonSDK
from datetime import datetime
import asyncio

today = datetime.today()
from cfg import YOUR_API_KEY
from sdks.fudstop_sdk.fudstop_sdk import fudstopSDK
fudstop = fudstopSDK()
polygon = AsyncPolygonSDK(YOUR_API_KEY)
from sdks.polygon_sdk.async_options_sdk import PolygonOptionsSDK
polyoptions = PolygonOptionsSDK(YOUR_API_KEY)
webull = AsyncWebullSDK()
import disnake
import asyncio


def find_gaps(o, h, l, c, t):
    gap_ups = []
    gap_downs = []

    for i in range(1, len(o)):
        if o[i] > c[i-1]:  # Check if the opening price is greater than the previous high price
            gap_ups.append(i)
        elif o[i] < c[i-1]:  # Check if the opening price is less than the previous low price
            gap_downs.append(i)

    gap_ups_with_timestamps = [(t[i], i) for i in gap_ups]
    gap_downs_with_timestamps = [(t[i], i) for i in gap_downs]

    return gap_ups_with_timestamps, gap_downs_with_timestamps


def find_gap_price_range(o, h, l, c, t, candle, filled=None):
    if o[candle] > h[candle - 1]:
        direction = "up"
        gap_low = l[candle - 1]
        gap_high = h[candle]
    elif o[candle] < l[candle - 1]:
        direction = "down"
        gap_low = l[candle]
        gap_high = h[candle - 1]
    else:
        direction = "unknown"
        gap_low = None
        gap_high = None

    if filled is None:
        return gap_low, gap_high
    else:
        fill_index = None
        for i in range(candle + 1, len(c)):
            if direction == "up":
                if l[i] <= gap_high and h[i] >= gap_low:
                    fill_index = i
                    break
            elif direction == "down":
                if h[i] >= gap_high and l[i] <= gap_low:
                    fill_index = i
                    break

        if filled is True:
            if fill_index is not None:
                return t[fill_index]
            else:
                return None
        else:
            if fill_index is not None:
                return t[fill_index]
            else:
                return None


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
        await interaction.send(f'> **GPT-4 Initializing...**')
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

            embed = disnake.Embed(title=f" GPT4 ", description=f"```py\n{message_content}```")
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


    @disnake.ui.button(style=disnake.ButtonStyle.blurple, custom_id="gpt4", label="GPT4")
    async def getplays(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        # Disable the button and update the message
        button.disabled = True
        await interaction.response.edit_message(view=self)

        await self.engage_gpt4_conversation(interaction)

        # Enable the button again and update the message
        button.disabled = False
        await interaction.response.edit_message(view=self)

    @disnake.ui.button(style=disnake.ButtonStyle.green,label="Core Calls", row=4, custom_id="corecalls")
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
        embed = disnake.Embed(title=f"Core Calls", description=f"\n** Symbol        Sector        RSI           Days Since ER**\n```{table}```", color=disnake.Colour.dark_green())
        await interaction.edit_original_message(embed=embed)
    @disnake.ui.button(style=disnake.ButtonStyle.red, label=f"Core Puts", row=4, custom_id="coreputs")
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
        embed = disnake.Embed(title=f" Core Puts ", description=f"\n** Symbol        Sector        RSI           Days Since ER**\n```{table}```", color=disnake.Colour.dark_red())
        await interaction.edit_original_message(embed=embed, view=view)




    @disnake.ui.button(style=disnake.ButtonStyle.red, custom_id="indices",label="All Indices", row=3)
    async def indices(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        """All indices data"""
        indices = await polygon.get_all_indices()
        file = 'files/indices/all_indices.csv'
        indices.to_csv(file)
        await interaction.send(file=disnake.File(file))


    @disnake.ui.button(style=disnake.ButtonStyle.red, custom_id="tickers",label="All Stocks", row=3)
    async def stocks(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        """All indices data"""
        await interaction.response.defer(with_message=True)
        await interaction.followup.send(f'> **Gathering all stock snapshots...**')
        indices = await polygon.get_all_snapshots()
        df = pd.DataFrame(indices)
        file = 'files\stocks\snapshots.csv'
        
        await interaction.edit_original_message(file=disnake.File(file))

    @disnake.ui.button(style=disnake.ButtonStyle.red, custom_id="alloptions", emoji=f"👑", row=3, label="All Options")
    async def alloptions(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        await interaction.response.defer(with_message=True)
        await interaction.followup.send(f'> **Fetching all options expiring today...**')
        options_contracts = await fetch_all_option_contracts(expiration_date_gte=today_str, expiration_date_lte=today_str)
        x = await polyoptions.get_snapshots(options_contracts,output_file="all_options.csv")
        df = pd.DataFrame(x)
        
        # for i, row in df.iterrows():

        await interaction.edit_original_message(file=disnake.File('all_options.csv'))





    @disnake.ui.button(style=disnake.ButtonStyle.green,row=0, custom_id="financescore", label="Financial Score")
    async def financialscore(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        """The Earnings Screener"""
        await interaction.response.defer()
        await interaction.send(f'> Gathering financial score..**', delete_after=4)
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
        embed2 = disnake.Embed(title=f"Additional Data:", description=f"Viewing Additional Data for {self.ticker}", color=disnake.Colour.random())
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
        eps_quarterly = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"EPS quarterly", row=4)
        eps_quarterly.callback = lambda interaction: interaction.send(embed=embed, file=disnake.File('files/financials/EPS_quarterly.png'), ephemeral=False)
        view.add_item(eps_quarterly)

        revenue_quarterly = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"Revenue Quarterly", row=4)
        revenue_quarterly.callback = lambda interaction: interaction.send(embed=embed, file=disnake.File('files/financials/Revenue_quarterly.png'), ephemeral=False)
        view.add_item(revenue_quarterly)

        roa_quarterly = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"ROA - Quarterly", row=4)
        roa_quarterly.callback = lambda interaction: interaction.send(embed=embed, file=disnake.File('files/financials/ROA_quarterly.png'), ephemeral=False)
        view.add_item(roa_quarterly)


        roe_quarterly = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"Return on Investment - Quarterly", row=4)
        roe_quarterly.callback = lambda interaction: interaction.send(embed=embed, file=disnake.File('files/financials/ROE_quarterly.png'), ephemeral=False)
        view.add_item(roe_quarterly)


        await interaction.edit_original_message(embed=embed)






    @disnake.ui.button(style=disnake.ButtonStyle.grey,label=f"Econ Calendar", row=3, custom_id="eoncal1337")
    async def economic_events(self, button: disnake.ui.Button, interaction: disnake.AppCommandInteraction):
        df = await webull.economic_events()
        filename="files/economy/economic_calendar.csv"
        df.to_csv(filename)
        embeds=[]
        for i, row in df.iterrows():

            actual = row['Actual']
            comment =row['Comment']
            country =row['Country']
            currency =row['Currency']
            date =row['Date']
            time =row['Time']
            forecast =row['Forecast']
            importance =row['Importance']
            indicator =row['Indicator']
            link =row['Link']
            period =row['Period']
            previous =row['Previous']
            scale =row['Scale']
            source =row['Source']
            title =row['Title']
            unit =row['Unit']

            embed = disnake.Embed(title=f"IMPORTANT Economic Calendar of Events", description=f"> **{comment}**", color=disnake.Colour.dark_orange())
            
            embed.add_field(name=f"Name:", value=f">  **{title}** ", inline=False)
            embed.add_field(name=f"Time:", value=f"> **{time}**")
            embed.add_field(name=f"Country:", value=f"> **{country}**\n> **{currency}** @ **{unit}**")
            embed.add_field(name=f"Forecast v Actual:", value=f">  **{forecast}** v. **{actual}**")
            embed.add_field(name=f"Period:", value=f"> **{period}**")
            embed.add_field(name=f"Previous:", value=f">  **{previous}**")
            embed.add_field(name=f"Source:", value=f"> {source}")
            embed.set_footer(text=f"Implemented by FUDSTOP")
            embeds.append(embed)
        view = AlertMenus(embeds)
        view.add_item(FTDShortSelect(self.bot, self.ticker, self))
        view.add_item(MainView(self.bot,self.ticker).change_ticker)


    



        button = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"Download Data")
        button.callback = lambda interaction: interaction.response.send_message(file=disnake.File(filename))
        view.add_item(button)
        await interaction.response.edit_message(embed=embeds[0], view=view)



    @disnake.ui.button(style=disnake.ButtonStyle.blurple, label="Change Ticker", emoji=f"🔀", row=4, custom_id="changetick")
    async def change_ticker(self, button: disnake.ui.Button, interaction: disnake.ApplicationCommandInteraction):
        logo = await polygon.get_polygon_logo(self.ticker)
        # Set up the message and listener to ask for a new ticker
        embed = disnake.Embed(title="Change Ticker", description="Please enter the new ticker symbol in chat:",
                            color=disnake.Colour.random())
        message = await interaction.response.edit_message(embed=embed)
        message = await interaction.send(f"**Type the next ticker.**", delete_after=5)
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

            # Update the OptionSelect and FTDShortSelect dropdown menus
            option_select = self.get_item('optionselect')
            option_select.ticker = self.ticker

            ftd_short_select = self.get_item('ftdshortselect')
            ftd_short_select.ticker = self.ticker
            await FTDShortSelect(self.bot,self.ticker,self).refresh_options(interaction)

            await origin_message.edit(embed=embed, view=self)

            embed = disnake.Embed(
                title=f"Data Ready",
                description=f"```py\nWelcome. Before you look at {self.ticker}'s data - here are the option stats on the day:```",
                color=disnake.Colour.old_blurple())
            embed.add_field(name=f"Futures Volume:", value=f"> **{futures:,}**")
            embed.add_field(name=f"Options Volume:", value=f"> High: **{f2high:,}**\n\n> Today: **{optvol:,}**\n\n> Low: **{f2low:,}**")
            embed.add_field(name=f"Averages:", value=f"> Monthly-Daily: **{mdaily:,}**\n> Yearly-Dayly: **{ydaily:,}**", inline=True)
            embed.set_thumbnail(logo)

            embed.set_footer(text=f"Viewing data for {self.ticker} | Implemented by FUDSTOP -")

            await origin_message.edit(embed=embed, view=self)
            await interaction.send(f"```py\n Ticker data has been updated to {self.ticker}```", delete_after=3)
            ftd_short_select.refresh_state(interaction)
            option_select.refresh_state(interaction)
            






class FTDShortSelect(disnake.ui.Select):
    def __init__(self, bot, ticker, main_view: MainView):
        self.bot=bot
        self.main_view = main_view
        self.polygon = AsyncPolygonSDK(YOUR_API_KEY)
        self.ticker=ticker
        super().__init__(
            placeholder=f"🇸 🇹 🇴 🇨 🇰  🇩 🇦 🇹 🇦  ➡️",
            min_values=1,
            max_values=1,
            custom_id=f"ftdshortselect",
            options= [ 
                disnake.SelectOption(label=f"View Cost Distribution", value="5", description=f"View the shares profit proportion and cost data."),
                disnake.SelectOption(label=f"View ETFs with {self.ticker}", value=f"4", description=f"View ETFs exposed to {self.ticker}"),
            ]
        )


    async def callback(self, interaction: disnake.MessageCommandInteraction):
        """Each selected option should initialize a command."""
        print("Callback function started") 
        self.current_ticker = self.main_view.ticker

        if self.values[0] == "5":
            cost = await webull.cost_distribution(self.current_ticker)
            avgcost=[i.avgCost for i in cost]
            inprotif70end=[i.chip70End for i in cost]
            inprofit70ratio=[i.chip70Ratio for i in cost]
            inprofit70start=[i.chip70Start for i in cost]
            inprofit90end=[i.chip90End for i in cost]
            inprofit90ratio=[i.chip90Ratio for i in cost]
            inprofit90start=[i.chip90Start for i in cost]
            closeprice=[i.close for i in cost]
            closeprofitratio=[i.closeProfitRatio for i in cost]
            distributions=[i.distributions for i in cost]
            totalshares=[i.totalShares for i in cost]

            data = { 
                'Close Price': closeprice,
                'Percent Profiting': closeprofitratio,
                'Average Cost': avgcost,
                'In Profit 70% Start Price': inprofit70start,
                'In Profit 70% End Price': inprotif70end,
                'In Profit 70% Ratio': inprofit70ratio,
                'In Profit 90% Start Price': inprofit90start,
                'In Profit 90% End Price': inprofit90end,
                'In Profit 90% Ratio': inprofit90ratio,
                'Total Shares': totalshares, 
            }


            df = pd.DataFrame(data)
            filename = f'files/stocks/cost/{self.current_ticker}_cost.csv'
            df.to_csv(filename)
            embeds = []
            for i,row in df.iterrows():
                close_price = row['Close Price']
                pctprofit = row['Percent Profiting']
                avg_cost = row['Average Cost']
                inprofitstart70 = row['In Profit 70% Start Price']
                inprofitend70 = row['In Profit 70% End Price']
                inprofitratio70 = row['In Profit 70% Ratio']
                inprofitstart90 = row['In Profit 90% Start Price']
                inprofitend90 = row['In Profit 90% End Price']
                inprofitratio90 = row['In Profit 90% Ratio']
                total_shares = row['Total Shares']
                embed = disnake.Embed(title=f"Cost Distribution Analysis for {self.current_ticker}", description=f"```py\nCurrently viewing the % shares proportioned in profit as well as average cost.```")
                embed.add_field(name=f"Close Price:", value=f"> **${close_price}**")
                embed.add_field(name=f"70% Profit Proportion:", value=f">  Start Price: **${inprofitstart70}**\n>  End Price: **${inprofitend70}**\n> Ratio: **{round(float(inprofitratio70)*100,2)}%**")
                embed.add_field(name=f"90% Profit Proportion:", value=f">  Start Price: **${inprofit90start}**\n>  End Price: **${inprofitend90}**\n> Ratio: **{round(float(inprofitratio90)*100,2)}%**")
                embed.add_field(name=f"Average Price & Percent Profiting:", value=f"> **${avg_cost}**\n> **${round(float(pctprofit)*100,2)}%**")
                embed.add_field(name=f"Total Shares:", value=f"> **{float(total_shares):,}**")
                embeds.append(embed)
            view = AlertMenus(embeds).add_item(FTDShortSelect(self.bot, self.current_ticker, self.main_view))
            df.to_csv(filename)
            button = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"Download ETF Holdings")
            button.callback = lambda interaction: interaction.response.send_message(file=disnake.File(filename))
            view.add_item(button)
            view.add_item(MainView(self.bot,self.ticker).change_ticker)                
        if self.values[0] == "4":
            etf_holdings = await webull.get_etfs_for_ticker(self.ticker)
            changeratio = [i.changeRatio for i in etf_holdings]
            etfname=[i.etfname for i in etf_holdings]
            stockname=[i.name for i in etf_holdings]
            ratio=[i.ratio for i in etf_holdings]
            sharenum=[i.shareNumber for i in etf_holdings]
            tickerid=[i.tickerId for i in etf_holdings]
            symbol=[i.symbol for i in etf_holdings]
            data = { 
                'ETF Name': etfname,
                'Name': stockname,
                'Shares of Stock': sharenum,
                'Ratio %': ratio,
                'Change Ratio': changeratio,
                'tick_id': tickerid

            }
            df = pd.DataFrame(data)
            filename = f'files/stocks/etfs/{self.current_ticker}_etfs.csv'
            
            embeds = []
            df_sorted = df.sort_values('Shares of Stock', ascending=False)
            df_sorted.to_csv(filename)
            for i, row in df_sorted.iterrows():
                etf = row['ETF Name']
                Name = row['Name']
                Shares = row['Shares of Stock']
                Ratio = row['Ratio %']
                ChangeRatio = row['Change Ratio']
                TickerID = row['tick_id']
                if Ratio is not None:
                    ratio_value = round(float(Ratio) * 100, 2)
                else:
                    ratio_value = "N/A"
                            
                embed = disnake.Embed(title=f"ETFs for {self.current_ticker}", description=f"```py\nViewing ETFs exposed to {self.current_ticker}```", color=disnake.Colour.dark_gold())
                embed.add_field(name=f"ETF:", value=f"> **{etf}** | **{Name}**")
                embed.add_field(name=f"Number of Shares:", value=f"> **{float(Shares):,}**")
                embed.add_field(name="Ratio", value=f"> **{ratio_value}**")
                embed.add_field(name=f"Change Ratio:", value=f"> **{round(float(ChangeRatio)*100,2)}**")
                embeds.append(embed)

            view = AlertMenus(embeds).add_item(FTDShortSelect(self.bot, self.ticker, self.main_view))
            df.to_csv(filename)
            button = disnake.ui.Button(style=disnake.ButtonStyle.blurple, label=f"Download ETF Holdings")
            button.callback = lambda interaction: interaction.response.send_message(file=disnake.File(filename))
            view.add_item(button)
            view.add_item(MainView(self.bot,self.ticker).change_ticker)

            await interaction.response.edit_message(view=view, embed=embeds[0])


    async def refresh_options(self, interaction: disnake.MessageCommandInteraction):
        # Update the options according to the new ticker
        for option in self.options:
            option.label = option.label.replace(self.ticker, self.main_view.ticker)
            option.description = option.description.replace(self.ticker, self.main_view.ticker)

        # Update the ticker attribute
        self.ticker = self.main_view.ticker
        self.refresh_state(interaction)


