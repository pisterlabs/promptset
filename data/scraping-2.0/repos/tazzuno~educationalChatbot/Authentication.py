import streamlit as st
import streamlit_authenticator as stauth
import secrets
import bcrypt
import re
import mysql.connector
import openai
from mysql.connector import Error

# Credenziali
credentials = {'usernames': {'user1': 'pass123'}}

# Genera chiave random
key = secrets.token_urlsafe(16)

# Inizializza login manager
login_manager = stauth.Authenticate(credentials,
                                    cookie_name='auth',
                                    key=key)

# Variabile globale password validata
validated_password = ""


def connetti_database():
    try:
        # Recupera le informazioni di connessione dal file secrets
        return mysql.connector.connect(**st.secrets["mysql"])
    except Exception as e:
        st.error(f"Errore di connessione al database: {e}")
        return None


def chiudi_connessione_database(connection):
    if connection and connection.is_connected():
        connection.close()


def validate_password(password):
    global validated_password

    if len(password) > 0:

        # Controllo lunghezza
        if len(password) < 8:
            st.error("Password troppo corta")
            return

        # Controllo maiuscolo
        if not any(char.isupper() for char in password):
            st.error("Inserisci almeno 1 maiuscola")
            return

            # Controllo carattere speciale
        if not re.search(r'[!@#$]', password):
            st.error("Inserisci almeno 1 carattere speciale")
            return

        validated_password = password
        return validated_password


def is_api_key_valid(key):
    try:
        openai.api_key = key
        response = openai.Completion.create(
            engine="davinci",  # https://platform.openai.com/docs/models
            prompt="This is a test.",
            max_tokens=5
        )
    except Exception as ex:
        return str(ex)
        return False
    else:
        return True


def aggiungi_utente_al_database(username, password, email, api_key, connection):
    if connection:
        try:
            cursor = connection.cursor()

            # Aggiungi l'utente al database
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
            cursor = connection.cursor()
            query = '''INSERT INTO Utenti (Username, Password, Email, API_key) VALUES (%s, %s, %s, %s)'''
            args = (username, password, email, api_key)
            cursor.execute(query, args)
            connection.commit()

        except Error as e:
            print(f"Errore durante l'aggiunta dell'utente al database: {e}")
        finally:
            chiudi_connessione_database(connection)


def verifica_credenziali(username, password, connection):
    if connection:
        try:
            cursor = connection.cursor()

            query = "SELECT * FROM utenti WHERE username = %s AND password = %s"
            values = (username, password)

            cursor.execute(query, values)

            # Estrai i risultati
            result = cursor.fetchall()

            # Mostra il risultato
            if result:
                return 1
            else:
                return 0

        except Error as e:
            print(f"Errore durante l'aggiunta dell'utente al database: {e}")
        finally:
            chiudi_connessione_database(connection)



