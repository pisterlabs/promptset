```python
# data_fetcher.py

import requests
from config import OPENAI_API_KEY, TRADING_FREQUENCY, ASSET

class DataFetcher:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.trading_frequency = TRADING_FREQUENCY
        self.asset = ASSET
        self.base_url = 'https://api.openai.com/v1/'

    def get_data(self):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        response = requests.get(f'{self.base_url}assets/{self.asset}/trades?frequency={self.trading_frequency}', headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f'Failed to fetch data: {response.content}')

if __name__ == "__main__":
    fetcher = DataFetcher()
    data = fetcher.get_data()
    print(data)
```
