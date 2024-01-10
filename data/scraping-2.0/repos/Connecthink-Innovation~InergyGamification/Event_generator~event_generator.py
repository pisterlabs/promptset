# -*- coding: utf-8 -*-

import datetime
import openai
import random
import time
from dotenv import load_dotenv
load_dotenv()

import ast
from typing import List
import csv
import os 

class EventGenerator():
    def __init__(self):
        """
        Constructor for the class.

        Attributes:
            current_day_str (str): The current day as a string.
            list_of_contexts (list): A list to store contexts.
            deployment_name (str): The deployment name.
            descriptions_events_today (list): A list to store the descriptions of events for today.
        """

        self.current_day_str = None
        self.list_of_contexts = []
        self.deployment_name = None

        #Initialize class attributes
        self.initialize()

        # Create an empty list to store the descriptions of the events for today:
        self.descriptions_events_today = []


    def initialize(self,):
        """
        Initialize settings and data for the class.

        Parameters:
            None
  
        Returns:
            None
        """
        self.save_current_day()
        self.save_gpt_settings()
        self.save_prompts()


    def save_current_day(self,):
        """
        Save the current day in the format "day de month_name".

        Parameters:
            None

        Returns:
            None
        """

        # Get the current day
        current_day = datetime.date.today()

        # Extract the month and day components
        month = current_day.month
        day = current_day.day

        # Create a month list
        month_list = ["Gener", "Febrer", "Març", "Abril", "Maig", "Juny", "Juliol", "Agost", "Setembre", "Octubre", "Novembre", "Desembre"]

        # Update the attribute concatenating the day and the corresponding month name from "month_list"
        self.current_day_str = str(day) + " de " + month_list[month-1]


    def save_gpt_settings(self,):
        """
        Save the GPT settings for connecting to Azure OpenAI model.

        Parameters:
            None
        
        Returns:
            None
        """

        #AZURE OPENAI
        #Conect to Azure OpenAI model:
        api_key = os.environ.get('MY_API_KEY')
        openai.api_key = api_key
        openai.api_base = "https://pruebaabel1231234152414.openai.azure.com/" # your endpoint
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15' # this may change in the future
        self.deployment_name='gpt35turbo' #This will correspond to the custom name you chose for your deployment when you deployed a model.


    def save_prompts(self,):
        """
        Generate a list of context dictionaries for different event scenarios.

        Parameters:
            None

        Returns:
            None
        """

        # List with different types of events in a certain place
        events_casal = ["un concert", "una actuació", "una obra de teatre", "una xerrada", "una festa de música moderna", "un sopar a l'aire lliure de festa major", "una festa popular"]

        # Get a random event of the previous list
        event_casal = random.choice(events_casal)

        # Generate a context based on the event obtained
        context_generation_night_events =  [
            {

                "role": "user",

                "content": f"""

                    Crea un event nocturn fictici com {event_casal} que es celebri a la plaça del casal de Canyelles

                    el dia {self.current_day_str}. L'event serà publicat a les xarxes socials. Es important que la descripció del event sigui molt extensa i sigui el més creible possible, que inclogui la data i el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM)

                    i que també inclogui la localització del event, que en aquest cas és la Plaça del casal. Respon únicament amb la descripcó de l'event.

                """

            },
        ]

        # Same for other certain places
        # ...

        events_pavello = ["un torneig de bàsquet", "un torneig de voleibol", "un torneig de futbol sala amb partits amistosos entre equips locals de veïns", "un partit de l'equip femení de hoquei sobre patins de la lliga catalana contra un altre equip regional"

                            "una exibició o competició de patinatge artísitc", "un torneig de tennis taula", "la celebració escolar de l'institut de Canyelles", "les jornades de la ciéncia per els més petits", "un partit de fútbol de l'equip masculí de futbol sala de la lliga catalana"]

        event_pavello = random.choice(events_pavello)




        context_generation_sports_events_pavello =  [

            {

                "role": "user",

                "content": f"""

                    Crea un event esportiu fictici com {event_pavello} que es pugui celebrar al pavelló municipal del municipi de Canyelles

                    el dia {self.current_day_str}. L'event serà publicat a les xarxes socials. Es important que la descripció del event sigui molt extensa i que inclogui la data i el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM)

                    i la localització del event, que en aquest cas és el pavelló municipal. Respon únicament amb la descripcó de l'event.

                """

            },

        ]

        context_generation_sports_events_futbol_field1 =  [

            {

                "role": "user",

                "content": f"""

                    Crea un event esportiu fictici que es pugui celebrar al camp de futbol municipal del municipi de Canyelles

                    el dia {self.current_day_str}. Es important que la descripció del event sigui molt extensa i que inclogui la data i el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM)

                    i la localització del event, que en aquest cas és el camp de futbol municipal. Respon únicament amb la descripcó de l'event i no diguis que és fictici.

                """

            },

        ]

        events_futbol_field2 = ["un torneig de bàsquet", "un torneig de voleibol adaptant el camp de fútbol amb xarxes", "un torneig de tennis taula a l'aire lliure", "un torneig de futbol amistós amb equips de l'institut de Canyelles", "un partit de fútbol de l'equip masculí de futbol sala de la lliga catalana"

                                "unes carreres de cotxes teledirigits", "unes jornades de les ciéncies naturals"]

        event_futbol_field2 = random.choice(events_futbol_field2)

        context_generation_sports_events_futbol_field2 =  [

            {

                "role": "user",

                "content": f"""

                    Creea la descripció d'un event que es pugui celebrar a un camp de fútbol relacionat amb {event_futbol_field2} el dia {self.current_day_str}.

                    Es important que la descripció del event sigui molt extensa i que inclogui la data i el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM)

                    i la localització del event, que en aquest cas és el camp de fútbol de la urbanització Vora Sitges.

                """

            },

        ]

        context_generation_events_school =  [

            {

                "role": "user",

                "content": f"""

                    Crea un event que es pugui celebrar a la Escola Pública Sant Nicolau de Canyelles

                    el dia {self.current_day_str}. Es important que la descripció del event sigui molt extensa i que inclogui la data i el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM)

                    i la localització del event, que en aquest cas és la Escola Pública Sant Nicolau de Canyelles. Respon únicament amb la descripcó de l'event.

                """

            },

        ]

        events_hotelito = ["una reunió de l'asociacció de veïns", "una conferencia sobre botànica", "una sessió de cinema a la fresca", "un concert de música en directe"

                                "Una sessió de ioga als jardins", "un sopar pouplar", "un sopar popular estil barbacoa", "una jornada de lectura per els més petits"]

        event_hotelito = random.choice(events_hotelito)

        context_generation_events_hotelito =  [

            {

                "role": "user",

                "content": f"""

                    Crea un event que es pugui celebrar als jardins de l'hotelito que està situat a la urbanització California

                    el dia {self.current_day_str}. l'event ha d'estar relacionat amb {event_hotelito}.

                    Es important que la descripció del event sigui molt extensa i que inclogui la data i el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM)

                    i la localització del event, que en aquest cas és els jardins el Hotelito.

                """

            },

        ]

        events_hotelito_night = ["una sessió de jazz", "un concert acústic d'un artista local", "un concert de rock", "concerts de grups musicals formats per joves del municipi"

                                    "una obra de teatre organitzada per l'associació de teatre de Canyelles"]

        event_hotelito_night = random.choice(events_hotelito_night)

        context_generation_events_hotelito_night = [

            {

                "role": "user",

                "content": f"""

                    Crea un event que es puguin celebrar als jardins de l'hotelito que estan situats a la urbanització California

                    el dia {self.current_day_str}. L'event ha d'anar relacionat amb l'oci nocturn com {event_hotelito_night}.

                    Es important que la descripció del event sigui molt extensa (com a mínim 100 paraules) i que inclogui la data concreta i el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM)

                    i la localització del event, que en aquest cas és els jardins el Hotelito.

                """

            },

        ]

        events_CANYA_club = ["una sessió de Jazz", "una festa per turistes", "una nit de mojitos", "una nit de cocktails"]

        event_CANYA_club = random.choice(events_CANYA_club)

        context_generation_events_CANYA_hclub = [
            {
                "role": "user",
                "content": f"""
                    Crea un event simulat que es celebri a la vila vacacional CANYA hlclub
                    el dia {self.current_day_str}. L'event ha d'anar relacionat amb l'oci nocturn com {event_CANYA_club}.
                    Es molt important que en la descripció aparegui el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM) i l'adreça del CANYA hclub que és Carrer l'Hospitalet, 17, 08811 Califòrnia, Barcelona.
                    Respon únicament amb la descripcó de l'event.

                """

            },

        ]

        events_club_patinatge = ["una sessió de portes obertes per veure entrenar al equip", "una sessió de portes obertes per principiants que vulguin venir a probar per primera vegada el patinatge", "una exhibició de l'equip local"

                                "una xerrada de la famosa patinadora Carla Sanchis sobre les seves experiencies i éxits", "una exhibició pre competició de les nostres millors patinadores"]

        event_club_patinatge = random.choice(events_club_patinatge)

        context_generation_events_patinatge = [

            {

                "role": "user",

                "content": f"""

                    Crea un event simulat com {event_club_patinatge} que es celebri al club de patinatge arístic de Canyelles el dia {self.current_day_str}.

                    Es important que la descripció del event inclogui la data concreta i el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM)

                    i la localització del event, que en aquest cas és el club de patinatge arístic. Respon únicament amb la descripció de l'event.

                """

            }

        ]

        events_TARA = ["una festa jove", "una nit jove", "una nit blanca amb una banda de Rock", "una festa hawaiana", "una festa organitzada per el propietaris de la vila"

                    "una festa de entrada lliure", "una nit de cocktails amb els millors baristes"]

        event_TARA = random.choice(events_TARA)

        context_generation_events_TARA_club = [

            {

                "role": "user",

                "content": f"""

                    Crea un event simulat que es celebri a la vila vacacional TARA club

                    el dia {self.current_day_str}. L'event ha d'anar relacionat amb l'oci nocturn com {event_TARA}.

                    Es molt important que en la descripció aparegui el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM) i la adreça del TARA club que és Carrer Nicaragua, 21, 08811 Canyelles, Barcelona.

                    Respon únicament amb la descripcó de l'event.

                """

            },

        ]


        events_castell = ["una nit de portes obertes al castell", "un dia de portes obertes al castell", "una visita guiada", "una visita a les masmorres del castell", "una visita a la torre del castell per veure la posta de sol"

                        "una conferencia sobre la historia de Canyelles", "una nit de histories de terror pels més petits"]

        event_castell = random.choice(events_castell)

        context_generation_events_castell = [

            {

                "role": "user",

                "content": f"""

                    Crea un event simulat que es celebri al castell de Canyelles el dia {self.current_day_str} com {event_castell}.

                    Es important que la descripció del event sigui molt extensa i que inclogui la data i el rang d'hores a la que comença y termina l'event en format (HH:MM-HH:MM)

                    i la localització del event, que en aquest cas és el castell de Canyelles.

                """

            }

        ]


        self.list_of_contexts = [

            context_generation_night_events,

            context_generation_sports_events_pavello,

            context_generation_sports_events_futbol_field1,

            context_generation_sports_events_futbol_field2,

            context_generation_events_school,

            context_generation_events_hotelito,

            context_generation_events_hotelito_night,

            context_generation_events_CANYA_hclub,

            context_generation_events_patinatge,

            context_generation_events_TARA_club,

            context_generation_events_castell,

        ]


    def generate_events(self,):
        """
        Generate events using the GPT model.

        Parameters:
            None

        Returns:
            None
        """

        # Add the minus 4 so we do not get 11 events for some days
        number_of_events = random.randint(1, len(self.list_of_contexts))

        # Get a random list len=number_of_elements from the list of contexts:
        today_contexts_events = random.sample(self.list_of_contexts, number_of_events)

        print("\nGenerating events..\n")
        
        #Iter. through each event context and try to generate an event description
   
        for event_context in today_contexts_events:

            try:
                completion = openai.ChatCompletion.create(
                    deployment_id=self.deployment_name,
                    messages=event_context
                )
                
                reply_content = completion.choices[0].message.content
                self.descriptions_events_today.append(reply_content)

            except Exception as e:
                print("OPENAI API ERROR. Rerun the code")        

class EventExtractor:

    def __init__(self, descriptions_events_today: List, project_root: str):
        """
        Class constructor to initialize the EventExtractor.

        Attributes:
            descriptions_events_today (List): A list of descriptions of events for the current day.
            project_root (Str): Project root
        """

        self.project_root = project_root

        # Store the list of descriptions of events for the current day
        self.descriptions_events_today = descriptions_events_today

        # Define the geolocation relationship dictionary
        self.coord_dict = {
            "plaça del casal": (41.285159447102856, 1.7217341064298708),

            "castell": (41.28713224852074, 1.722669984574967),

            "urbanització california": (41.26815636260956, 1.7241116939992642),

            "camp de futbol municipal": (41.281734200477494, 1.7186586581957097),
            "pavelló municipal": (41.28209636259826, 1.7193940157374974),

            "camp de futbol organització Vora sitges": (41.277844, 1.738181), 

            "escola pública sant nicolau": (41.282733770529845, 1.7185643436835267),

            "club de patinatge": (41.28290637166816, 1.7195591123248108),

            "canya hlclub": (41.26732522298323, 1.7244491828494213),
            "canya hclub": (41.26732522298323, 1.7244491828494213),
            "carrer l\hospitalet, 17": (41.26732522298323, 1.7244491828494213),

            "tara club": (41.28431829911173, 1.7481244797587359),
            "carrer nicaragua, 21": (41.28431829911173, 1.7481244797587359)
        }

        # Prompts with examples to extract the date, location and exact time of the event
        self.extractor_context = []

        # List where we are going to store the extracted important information about the events:
        self.data_extraction_list = []

        # List with the data converted to dictionaries:
        self.out_list = []

        # Initialize OpenAI GPT settings
        self.save_gpt_settings()

    def save_gpt_settings(self,):
            """
            Method to set up the OpenAI GPT settings for the EventExtractor.

            Parameters:
                None

            Returns:
                None            
            """

            #AZURE OPENAI
            #Conect to Azure OpenAI model:
            api_key = os.environ.get('MY_API_KEY')
            print(api_key)
            openai.api_key = api_key
            openai.api_base = "https://pruebaabel1231234152414.openai.azure.com/" # your endpoint
            openai.api_type = 'azure'
            openai.api_version = '2023-05-15' # this may change in the future
            self.deployment_name='gpt35turbo' #This will correspond to the custom name you chose for your deployment when you deployed a model.


    def store_extractor_context(self):
        """
        Method to store example prompts for the information extraction process.

        Parameters:
            None

        Returns:
            None
        """

        self.extractor_context = [
            {
                "role": "user",
                "content": """
                    Et passaré la descripció d'un event. Necessito que em retornis en forma de diccionari de python el títol de l'event, localització, la data i rang d'hores on l'event és atiu. Respon entesos si ho has entés.
                """
            },
            {
                "role": "assistant",
                "content": "Entesos"
            },
            {
                "role": "user",
                "content": """
                    Festa de White Sensation el 10 de juliol de 21:00 a 23:00 h. a la Urbanització California. Els jardins de l'Hotelito seran el lloc per gaudir de la millor música electrònica amb el millor vestimenta de blanc per als assistents.
                """
            },
            {
                "role": "assistant",
                "content": """
                    {"Title":"Festa de White Sensation", "Location": "Jardins de l'Hotelito Urbanització California", "Schedule": "21:00-23:00", "Description: Festa de White Sensation. Gaudeix de la millor música electrònica amb el millor vestimenta de blanc per als assistents.}
                """
            },
            {
                "role": "user",
                "content": """
                    Us convidem al nostre event nocturn de teatre a la plaça del casal de Canyelles el proper 7 de Juliol. De les 20:00 - 22:00 h, els nostres actors i actrius ens transportaran a una història emocionant i intrigant que ens mantindrà pegats a la cadira fins al final del show. \n\nAmb una decoració espectacular on la il·luminació serà el punt clau per crear l'ambient adequat, podràs gaudir del millor teatre a l'aire lliure. La història és una trama de misteri on els amants dels escenaris seran el més ben atesos, amb actors i actrius que lluiran disfresses exòtiques i amb un guion ben treballat.\n\nLa plaça del casal de Canyelles és el lloc perfecte per a aquesta nit de teatre, amb un escenari ampli que ens permetrà fer una espectacular representació per a tots els assistents. Els seguidors del teatre tenen una cita ineludible a Canyelles que no es poden perdre. \n\nNo t'hagis de preocupar, si la calor fa que et refregiïs, els nostres amics de Canyelles estaran a la nostra disposició per a mantenir-nos frescos amb begudes en una nit que promet ser la més entretinguda de l'estiu. \n\nNo hi haurà cap mena de dubte, l'event nocturn de teatre a la plaça del casal de Canyelles el 7 de Juliol serà l'event més popular de la nit, i tu no pots faltar. T'esperem per a viure una nit inoblidable de teatre sota les estrelles.
                """
            },
            {
                "role": "assistant",
                "content": """
                    {"Title":Teatre nocturn", "Location": "Plaça del casal de Canyelles", "Schedule": "20:00-22:00", "Description: "Teatre nocturn a la plaça del casal de Canyelles. Els nostres actors i actrius ens transportaran a una història emocionant i intrigant que ens mantindrà pegats a la cadira fins al final del show. \n\nAmb una decoració espectacular on la il·luminació serà el punt clau per crear l'ambient adequat, podràs gaudir del millor teatre a l'aire lliure. La història és una trama de misteri on els amants dels escenaris seran el més ben atesos, amb actors i actrius que lluiran disfresses exòtiques i amb un guion ben treballat.\n\nLa plaça del casal de Canyelles és el lloc perfecte per a aquesta nit de teatre, amb un escenari ampli que ens permetrà fer una espectacular representació per a tots els assistents. Els seguidors del teatre tenen una cita ineludible a Canyelles que no es poden perdre. \n\nNo t'hagis de preocupar, si la calor fa que et refregiïs, els nostres amics de Canyelles estaran a la nostra disposició per a mantenir-nos frescos amb begudes en una nit que promet ser la més entretinguda de l'estiu. \n\nNo hi haurà cap mena de dubte, l'event nocturn de teatre a la plaça del casal de Canyelles el 7 de Juliol serà l'event més popular de la nit, i tu no pots faltar. T'esperem per a viure una nit inoblidable de teatre sota les estrelles"}
                """
            },
            {
                "role": "user",
                "content": """
                    Celebrem l'estiu a la Escola Pública Sant Nicolau de Canyelles el dia 7 de Juliol. Vine amb la família i amics a gaudir d'un dia ple de diversió i entreteniment per a totes les edats.\n\nComençarem a les 10 del matí amb un esmorzar saludable per agafar forces i començar a jugar a les diferents activitats que hem preparat: jocs d'aigua, inflables, tallers de manualitats, face painting i moltes sorpreses més! També tindrem una zona de food trucks perquè pugueu gaudir d'una àmplia varietat de menjar i beguda.\n\nA la tarda, a partir de les 18h, podrem gaudir d'un concert en directe a càrrec d'un grup local de música que tocaran els millors èxits de l'estiu.\n\nLa festa acabarà cap a les 22h, però abans gaudirem d'una una gran festa de la espuma per refrescar-nos i acabar el dia amb una gran somriure. No et perdis aquesta gran festa d'estiu a la Escola Pública Sant Nicolau de Canyelles! Ens veiem allà!
                """
            },
            {
                "role": "assistant",
                "content": """
                    {"Title":"Event d'estiu", "Location": "Escola Pública Sant Nicolau de Canyelles", "Schedule": "10:00-22:00", "Description": "Vine amb la família i amics a gaudir d'un dia ple de diversió i entreteniment per a totes les edats.\n\nComençarem a les 10 del matí amb un esmorzar saludable per agafar forces i començar a jugar a les diferents activitats que hem preparat: jocs d'aigua, inflables, tallers de manualitats, face painting i moltes sorpreses més! També tindrem una zona de food trucks perquè pugueu gaudir d'una àmplia varietat de menjar i beguda.\n\nA la tarda, a partir de les 18h, podrem gaudir d'un concert en directe a càrrec d'un grup local de música que tocaran els millors èxits de l'estiu.\n\nLa festa acabarà cap a les 22h, però abans gaudirem d'una una gran festa de la espuma per refrescar-nos i acabar el dia amb una gran somriure. No et perdis aquesta gran festa d'estiu a la Escola Pública Sant Nicolau de Canyelles! Ens veiem allà!"}
                """
            },
            {
                "role": "user",
                "content": """
                    Us convidem a participar en les emocionants Jornades de les Ciències Naturals que es celebraran el proper 7 de juliol al camp de futbol de la urbanització Vora Sitges. Aquesta és una oportunitat única per explorar, estimular la curiositat i apropar-te a la ciència a través de diferents activitats.\n\nL'esdeveniment està programat per començar a les 10 del matí i continuar fins a les 6 de la tarda, amb moltes activitats emocionants pel camí. Vindràs a participar en tallers interactius que inclouen experimentació i aprenentatge sobre l'entorn natural que ens envolta.\n\nEls tallers seran dirigits pels nostres guies especialitzats en ciències naturals, que us oferiran les seves habilitats i coneixements expert per ajudar-vos a explorar i descobrir el món que ens rodeja. A més, també tindrem a disposició una àmplia gamma de material didàctic que us ajudarà en la comprensió dels nous conceptes.\n\nEl camp de futbol de Vora Sitges és un lloc ideal per celebrar les nostres Jornades de les Ciències Naturals, i estem segurs que les instal·lacions ens proporcionaran un espai ample i adequat per realitzar totes les activitats programades.\n\nEl transport no serà un problema, ja que hi hauran autobusos de línia que us acostaran fins al lloc de l'esdeveniment, i us garantim un ambient segur i agradable per a tothom.\n\nPer si no en teniu prou, també hi haurà una àmplia gamma d'alimentació i beguda disponible per a tothom. Si us plau, no oblideu dur roba còmoda i, és clar, una gran dosi de curiositat!\n\nAixí doncs, no espereu més i uneix-te a les nostres emocionants Jornades de les Ciències Naturals al camp de futbol de Vora Sitges el proper 7 de juliol. Aquest esdeveniment us obre les portes a un món meravellós de descobriments i aprendizatge, i no voldreu perdre-vos'l!
                """
            },
            {
                "role": "assistant",
                "content": """
                    {"Title": Jornades de les Ciències Naturales", "Location": "Camp de futbol de la urbanització Vora Sitges", "Schedule": "10:00-18:00", "Description": Emocionants Jornades de les Ciències Naturals. Aquesta és una oportunitat única per explorar, estimular la curiositat i apropar-te a la ciència a través de diferents activitats.\n\nL'esdeveniment està programat per començar a les 10 del matí i continuar fins a les 6 de la tarda, amb moltes activitats emocionants pel camí. Vindràs a participar en tallers interactius que inclouen experimentació i aprenentatge sobre l'entorn natural que ens envolta.\n\nEls tallers seran dirigits pels nostres guies especialitzats en ciències naturals, que us oferiran les seves habilitats i coneixements expert per ajudar-vos a explorar i descobrir el món que ens rodeja. A més, també tindrem a disposició una àmplia gamma de material didàctic que us ajudarà en la comprensió dels nous conceptes.\n\nEl camp de futbol de Vora Sitges és un lloc ideal per celebrar les nostres Jornades de les Ciències Naturals, i estem segurs que les instal·lacions ens proporcionaran un espai ample i adequat per realitzar totes les activitats programades.\n\nEl transport no serà un problema, ja que hi hauran autobusos de línia que us acostaran fins al lloc de l'esdeveniment, i us garantim un ambient segur i agradable per a tothom.\n\nPer si no en teniu prou, també hi haurà una àmplia gamma d'alimentació i beguda disponible per a tothom. Si us plau, no oblideu dur roba còmoda i, és clar, una gran dosi de curiositat!\n\nAixí doncs, no espereu més i uneix-te a les nostres emocionants Jornades de les Ciències Naturals al camp de futbol de Vora Sitges el proper 7 de juliol. Aquest esdeveniment us obre les portes a un món meravellós de descobriments i aprendizatge, i no voldreu perdre-vos'l!}

                """
            },
        ]

    def extract(self):
        """
        Method to extract important information (location, date, and time) from the descriptions of events.

        Parameters:
            None

        Returns:
            None
        """

        print("Extracting events data...")

        for event in self.descriptions_events_today:
            # Create a temporal context:
            temp_extractor_context = self.extractor_context.copy()

            # add the description of the event to the whole promt:
            event_dict = {
                "role": "user",
                "content": event
            }

            temp_extractor_context.append(event_dict)
            
            try:
                completion = openai.ChatCompletion.create(
                    deployment_id=self.deployment_name,
                    messages=temp_extractor_context
                )
                reply_content = completion.choices[0].message.content

                self.data_extraction_list.append(reply_content)
            except Exception as e:
                print("\nERROR extracting data in context:\n", temp_extractor_context)
                print("\nRaw ERROR:", e)
                

    def append_geolocation(self,):
        """
        Method to append geolocation (latitude and longitude) information to the extracted data.

        Parameters:
            None

        Returns:
            None
        """

        # Create a new empty list to store the geolocated events.
        new_list = []

        #Iterate through the list of extracted events
        for event in self.data_extraction_list:
            # Convert the event to lowercase to perform a case-insensitive search.            
            event_lower = event.lower()
            # Iterate through the dictionary of coordinates.
            for substring, coordinates in self.coord_dict.items():
                # If the substring is found in the event (ignoring case):
                if substring in event_lower:
                    # Get the coordinates associated with the substring and add them to the existing JSON
                    lat, lon = coordinates
                    event = event[:-1] + f', "lat": {lat}, "lon": {lon}}}'
                    break # Exit the loop once the match has been found.
            
            # Add the modified event to the new list.
            new_list.append(event) 
        
        # Replace the extracted data list with the new list containing the geolocation information.
        self.data_extraction_list = new_list

    def literal_eval(self):
        """
        Method to convert the extracted data from strings to dictionaries using ast.literal_eval.

        Parameters:
            None

        Returns:
            List: A list containing the extracted data converted into dictionaries.
        """

        for event in self.data_extraction_list:

            # Replace the '\' because in some cases chat gpt puts some of this characters in the strings
            event = event.replace("\\", "")
            try:
                self.out_list.append(ast.literal_eval(event))
            except Exception as e:
                print("\nERROR converting events data into dictionaries:\n", event)

    def save_output(self,):
        path_dir = os.path.join(self.project_root, "Event_generator", "data")
        
        # Create the directory if it does not exist
        os.makedirs(path_dir, exist_ok=True)

        path = os.path.join(path_dir, "fake_events.csv")

         # Open CSV file in write mode
        with open(path, 'w', newline='') as archivo_csv:
            # Get first dictionary keys as column names
            columnas = self.out_list[0].keys()

            # Create a DictWriter object to write to the CSV file
            writer = csv.DictWriter(archivo_csv, fieldnames=columnas)

            # Write the header with the column names
            writer.writeheader()

            # Write data from each dictionary as rows to CSV file
            for diccionario in self.out_list:
                writer.writerow(diccionario)

def main(project_root):

    event_generator = EventGenerator()
    event_generator.generate_events()

    descriptions_events_today = event_generator.descriptions_events_today
    print("Nº of events generated: ", len(descriptions_events_today))

    
    event_extractor = EventExtractor(descriptions_events_today, project_root)
    event_extractor.store_extractor_context()
    event_extractor.extract()
    event_extractor.append_geolocation()

    event_extractor.literal_eval()

    event_extractor.save_output()
  

#main(project_root=r"c:\Users\abelb\Desktop\Gamification - main test")





