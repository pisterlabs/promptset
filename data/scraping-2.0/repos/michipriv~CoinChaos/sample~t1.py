import openai
import matplotlib.pyplot as plt
from binance.client import Client
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os

def check_and_create_database(db_path):
    # Überprüfen, ob die Datenbankdatei bereits existiert
    db_exists = os.path.exists(db_path)

    if not db_exists:
        # Erstellen der Datenbank und der Tabelle, wenn sie nicht existiert
        create_database()
        print(f"Datenbank {db_path} erstellt und Tabelle btcusd_data angelegt.")
    else:
        print(f"Datenbank {db_path} existiert bereits.")

def create_database():
    conn = sqlite3.connect('crypto_data.db')
    c = conn.cursor()

    # Löschen der alten Tabelle (Dies entfernt alle bestehenden Daten!)
    c.execute('DROP TABLE IF EXISTS btcusd_data')

    # Erstellen der neuen Tabelle mit der Spalte 'currency_pair'
    c.execute('''CREATE TABLE btcusd_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  date TEXT, 
                  open REAL, 
                  high REAL, 
                  low REAL, 
                  close REAL, 
                  currency_pair TEXT)''')
    conn.commit()
    conn.close()



def save_to_database(bars, currency_pair):
    conn = sqlite3.connect('crypto_data.db')
    c = conn.cursor()
    for bar in bars:
        # Hinzufügen des Währungspaars zu jedem Datensatz
        c.execute('REPLACE INTO btcusd_data VALUES (?, ?, ?, ?, ?, ?)', bar + (currency_pair,))
    conn.commit()
    conn.close()




def get_historical_data_from_db():
    conn = sqlite3.connect('crypto_data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM btcusd_data ORDER BY date')
    bars = c.fetchall()
    conn.close()
    return bars

def get_historical_data(client):
    # Überprüfen Sie zuerst, ob die Daten bereits in der Datenbank vorhanden sind
    bars = get_historical_data_from_db()
    if not bars:
        # Daten sind nicht vorhanden, also laden Sie sie von der API
        bars = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "30 days ago UTC")
        save_to_database([(datetime.fromtimestamp(bar[0]/1000).strftime('%Y-%m-%d'), float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4])) for bar in bars])
        return bars
    return bars

def get_data(client, db_path, time_frame, currency_pair, start_datum, ende_datum):
    # Verbindung zur Datenbank herstellen
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Daten aus der Datenbank abrufen, die dem Zeitrahmen entsprechen
    c.execute('SELECT * FROM btcusd_data WHERE currency_pair = ? AND date BETWEEN ? AND ?', (currency_pair, start_datum, ende_datum))
    data = c.fetchall()

    if data:
        print("Daten wurden aus der Datenbank für", currency_pair, "im Zeitraum", start_datum, "bis", ende_datum, "abgefragt.")
    else:
        print("Daten für", currency_pair, "im Zeitraum", start_datum, "bis", ende_datum, "werden von der API abgerufen.")

        # Konvertierung des Zeitintervalls in ein Format, das von der Binance API akzeptiert wird
        kline_interval = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
        # Fügen Sie hier weitere Konvertierungen hinzu
    }.get(time_frame, "1h")  # Standardwert ist 1 Tag

        # API-Aufruf, um die Daten zu bekommen
        bars = client.get_historical_klines(currency_pair, kline_interval, start_datum, ende_datum)
        data_to_save = [(datetime.fromtimestamp(bar[0]/1000).strftime('%Y-%m-%d'), float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]), currency_pair) for bar in bars]

        # Test welcher Zeitruam wird abgerufen
        #for data in data_to_save:
        #  print(data)

        # Speichern der neuen Daten in der Datenbank
        c.executemany('INSERT INTO btcusd_data (date, open, high, low, close, currency_pair) VALUES (?, ?, ?, ?, ?, ?)', data_to_save)
        conn.commit()

    conn.close()

    return data_to_save



def get_api_keys(file_name):
    keys = {}
    with open(file_name, 'r') as file:
        for line in file:
            if "OPENAI_API_KEY" in line:
                keys['openai'] = line.split('=')[1].strip()
            elif "BINANCE_API_KEY" in line:
                keys['binance_key'] = line.split('=')[1].strip()
            elif "BINANCE_API_SECRET" in line:
                keys['binance_secret'] = line.split('=')[1].strip()
            elif "DB_PATH" in line:
                keys['db_path'] = line.split('=')[1].strip()
    return keys




def plot_candlestick_chart(bars, start_date_str, end_date_str):
    # Überprüfen, ob Daten vorhanden sind
    if not bars:
        print("Keine Daten zum Plotten verfügbar.")
        return

    # Umwandeln der Daten in ein für mplfinance passendes Format
    ohlc_data = []
    for bar in bars:
        # Umwandeln des Datumsstring in ein datetime-Objekt
        date = datetime.strptime(bar[0], '%Y-%m-%d')
        open_price = float(bar[1])
        high = float(bar[2])
        low = float(bar[3])
        close = float(bar[4])
        ohlc_data.append((date, open_price, high, low, close))

    # Erstellen eines DataFrame und Setzen des 'Date' als Index
    df = pd.DataFrame(ohlc_data, columns=['Date', 'Open', 'High', 'Low', 'Close'])
    df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
    df.drop(columns=['Date'], inplace=True)  # Entfernen der nun redundanten 'Date'-Spalte

    # Überprüfen, ob der DataFrame Daten enthält
    if df.empty:
        print(f"Keine Daten für den Zeitraum {start_date_str} bis {end_date_str} gefunden.")
        return

    # Erstellen des Candlestick-Charts
    mpf.plot(df, type='candle',
             title=f"BTCUSD Candlestick Chart from {start_date_str} to {end_date_str}",
             volume=False,
             figratio=(20,10),
             tight_layout=True,
             show_nontrading=True,
             style='charles')





def main():
    # Konfigurationsdaten auslesen
    api_keys = get_api_keys('config.txt')
    openai.api_key = api_keys['openai']
    binance_client = Client(api_keys['binance_key'], api_keys['binance_secret'])
    db_path = api_keys['db_path']

    # Überprüfen und Erstellen der Datenbank
    check_and_create_database(db_path)

    # Parameter für den Zeitraum und die Währung definieren
    time_frame = "1h"  # Beispiele: "1d" für 1 Tag, "4h" für 4 Stunden
    currency_pair = "BTCUSDT"  # Beispiele: "BTCUSDT", "ETHUSDT"
    start_datum = "2023-12-07"  # Beispiel: Startdatum
    ende_datum = "2023-12-08"  # Beispiel: Enddatum

    # Daten abrufen oder aus DB auslesen
    data = get_data(binance_client, db_path, time_frame, currency_pair, start_datum, ende_datum)
 
    #candlestick anzeigen
    plot_candlestick_chart(data, start_datum, ende_datum)



# Verwendung der Funktion
db_path = 'crypto_data.db'
check_and_create_database(db_path)

if __name__ == "__main__":
    main()
