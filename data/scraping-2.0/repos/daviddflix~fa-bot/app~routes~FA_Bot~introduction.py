import os
import gspread
import requests
from gspread.cell import Cell
from dotenv import load_dotenv
from app.services.openAI import ask

# Load environment variables from the .env file
load_dotenv()


BASE_URL = 'https://pro-api.coingecko.com/api/v3'
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

gc = gspread.service_account(filename='app\\services\\service_account.json')


def introduction(coin, sh_url): 
    try:
        intro_prompt = f"""Write a short paragraph (maximum 400 characters) explaining the problem addressed by {coin} cryptocurrency project and evaluate its effectiveness in solving it, using a professional tone. Analyze its positioning in the market without using adjectives that exaggerate its attributes, such as 'bold', 'unique' or 'groundbreaking'"""

        result = ask(intro_prompt)

        if result:
            try:
                sh = gc.open_by_url(url=sh_url)
                worksheet = sh.get_worksheet(1)

                cell_list = [
                        Cell(3, 2, result),  
                    ]
        
                worksheet.update_cells(cell_list)
                return f'Introduction data updated successfully for {coin}'
            
            except Exception as e:
                return f'Error writing {coin} introduction data to the spreadsheet: {str(e)}'

        return f'Empty result received from OpenAI for {coin}'
    
    except Exception as e:
        return f'An error occurred in the introduction function: {str(e)}'
    

# Gets the market cap and 24h volume
def fetch_and_write_introduction_data(coin, sh_url):

    formatted_coin = str(coin).casefold()

    try:

        headers = {
            "Content-Type": "application/json",
            "x-cg-pro-api-key": COINGECKO_API_KEY,
        }

        response = requests.get(f'{BASE_URL}/simple/price?ids={formatted_coin}&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_last_updated_at=true', headers=headers)
        response.raise_for_status()
     
        if response.status_code == 200:
            response_json = response.json()
            coin_data = response_json.get(formatted_coin, {})

            usd_market_cap = coin_data.get('usd_market_cap', False)
            usd_24h_vol = coin_data.get('usd_24h_vol', False)

            # Format numbers with commas and round to 2 decimal places
            formatted_market_cap = '{:,.2f}'.format(usd_market_cap)
            formatted_24h_vol = '{:,.2f}'.format(usd_24h_vol)

            # Starts writing data to the spreadhseet
            try:
                sh = gc.open_by_url(url=sh_url)
                worksheet = sh.get_worksheet(1)

                cell_list = [
                        Cell(4, 2, formatted_market_cap),  
                        Cell(5, 2, formatted_24h_vol),  
                    ]
        
                worksheet.update_cells(cell_list)
                return f'Market cap and 24h volume updated successfully for {formatted_coin}'
            
            except Exception as e:
                return f'Error writing {formatted_coin} Market cap and 24h volume to the spreadsheet: {str(e)}'
             
    except Exception as e:
        return f'Error fetching Market cap and 24h volume for {formatted_coin}: {str(e)}'


