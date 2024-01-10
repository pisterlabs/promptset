import os
import sys


if __name__ == '__main__':
    if sys.prefix == sys.base_prefix:
        print('\U0001F44E Ten plik należy uruchomić w wirtualnym środowisku Pythona. Punkt 1 instrukcji w pliku README.md')
        sys.exit(1)
    else:
        print('\U0001F44D Skrypt uruchomiony w wirtualnym środowisku Pythona')

    try:
        import openai
        print('\U0001F44D openai zainstalowane')
    except ImportError:
        print('\U0001F44E openai nie zainstalowane. Zainstaluj pakiety z pliku requirements.txt. Punkt 5 instrukcji w pliku README.md')
        sys.exit(1)

    try:
        from langchain.llms import OpenAI
        print('\U0001F44D langchain zainstalowane')
    except ImportError:
        print('\U0001F44E langchain nie zainstalowane. Zainstaluj pakiety z pliku requirements.txt. Punkt 5 instrukcji w pliku README.md')
        sys.exit(1)

    try:
        import flask
        print('\U0001F44D flask zainstalowane')
    except ImportError:
        print('\U0001F44E flask nie zainstalowane. Zainstaluj pakiety z pliku requirements.txt. Punkt 5 instrukcji w pliku README.md')
        sys.exit(1)

    try:
        import flask_cors
        print('\U0001F44D FlaskCORS zainstalowane')
    except ImportError:
        print('\U0001F44E FlaskCORS nie zainstalowane. Zainstaluj pakiety z pliku requirements.txt. Punkt 5 instrukcji w pliku README.md')
        sys.exit(1)

    print("\U0001F44D \U0001F44D \U0001F44D DZIAŁA!")
    sys.exit(0)