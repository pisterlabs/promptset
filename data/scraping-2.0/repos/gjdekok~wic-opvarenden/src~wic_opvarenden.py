# Reading and working with files
import os
import csv

# Reading Page XML-files and coordinates
from pagexml.parser import parse_pagexml_file
from shapely.geometry import Polygon, box

# Searching within deed texts
from fuzzy_search.fuzzy_phrase_searcher import FuzzyPhraseSearcher
from fuzzy_search.fuzzy_phrase_model import PhraseModel
import re

# Using OpenAI (optional)
import openai
import json
import dotenv

# Making it faster with parallel processing
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO)

# load api key from .env file
dotenv.load_dotenv()
openai.api_key = os.getenv("API_KEY")

def process_chunk(deeds_chunk):
    sailor_extractor = SailorExtractor(deeds_chunk)
    sailors_chunk = sailor_extractor.extract_sailors()
    return sailors_chunk

# Utility functions

def overlap(box_coord, target_coord, margin=0):
    """
    Function to determine if coordinates overlap. Uses Shapely's Polygon class.
    The function returns 1 if there is overlap, otherwise 0.
    
    box_coord: the coordinates of the box (i.e. a textline).
    target_coord: the coordinates of the target (i.e. a name).
    margin: the margin by which the target is enlarged for better matching.
    
    """

    # Make Shapely polygon from box coordinates
    polygon = Polygon(box_coord)

    #Check if there is overlap
    return polygon.intersects(target_coord)

def calculate_bounding_rectangle(list_of_coordinates):
    """
    Function to calculate the overarching bounding rectangle of a list of coordinates.
    """
    all_points = [point for coordinates in list_of_coordinates for point in coordinates]

    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)

    bounding_rectangle = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]] #List of lists instead of tuples for easy transfer to JS code later

    return bounding_rectangle

def check_for_empty(input_data):
    if isinstance(input_data, str):
        # Checks if string is alphanumeric
        if input_data.isalnum():
            return input_data
        else:
            return None
    # Checks if input is a list
    elif isinstance(input_data, list):
        # Goes through the list and checks each element
        new_list = [x if x.isalnum() else None for x in input_data]
        # If the list contains only Nones, return None
        if all(x is None for x in new_list):
            return None
        else:
            return new_list
    else:
        raise TypeError("Input must be a string or a list of strings.")
    

# Classes

class Person:
    """
    The Person class represents a person mentioned in a deed with a unique URI, label (name), and coordinates.
    """

    def __init__(self, person_uri, label, coordinates):
        self.person_uri = person_uri
        self.label = label
        self.coordinates = coordinates


class Location:
    """
    The Location class represents a location mentioned in a deed with a unique URI, label (name), and coordinates.
    """

    def __init__(self, location_uri, label, coordinates):
        self.location_uri = location_uri
        self.label = label
        self.coordinates = coordinates


class Deed:
    """
    The Deed class represents a notarial deed with associated metadata, persons, and locations.
    It contains methods for retrieving the corresponding PageXML and extracting the first few lines of the deed.
    """

    # Add notarial deeds as a class attribute
    # This is a dict with deed URIs as keys and metadata as values
    notarial_deeds = {}
    with open('../data/index/records.csv') as f:
        for line in csv.DictReader(f):
            notarial_deeds[line['id']] = dict(line)


    def __init__(self, deed_uri, date=None, subject=None, persons=None, locations=None):
        self.deed_uri = deed_uri
        self.persons = []
        self.locations = []

        if date:
            self.date = date

        if subject:
            self.subject = subject

        if persons:
            for person_uri, person_data in persons.items():
                person = Person(
                    person_uri,
                    person_data['label'],
                    person_data['coordinates']
                )
                self.persons.append(person)

        if locations:
            for location_uri, location_data in locations.items():
                location = Location(
                    location_uri, 
                    location_data['label'],
                    location_data['coordinates']
                )
                self.locations.append(location)


    def get_pagexml(self):
        """
        Locates a deed from the instance's deed_uri.
        It looks for the deed in a local folder.
        Returns a parsed version of the Page XML of the first deed page if successful, otherwise None.
        """
        begin_scanname = None

        # The deed should be in the index of notarial deeds ...
        if self.deed_uri in self.notarial_deeds:
            begin_scanname = self.notarial_deeds[self.deed_uri]['begin_scanname']

        if not begin_scanname:
            return None

        # Convert scanname to xml-filename
        scan = begin_scanname[:-6]
        page = int(begin_scanname[-6:])
        filename = str(scan) + str(page).zfill(6) + ".xml"

        # Look for the file in the local folder
        if not os.path.exists(f'../data/pagexml/{filename}'): 
            raise FileNotFoundError

        # # Get image URL
        # if filename in image_url:
        #     this_image_url = image_url[filename]
        # else:
        #     this_image_url = None

        current = parse_pagexml_file(f'../data/pagexml/{filename}')

        return current


    def get_first_lines(self, pagexml):
        """
        Returns the first few lines of a deed. 
        It tries to estimate the start of the deed by looking at its begin_coordinates.
        If there are no known begin coordinates, the function returns None.

        pagexml: the deed PageXML
        """

        deed = self.deed_uri
        if deed in self.notarial_deeds:
            begin_coordinates = self.notarial_deeds[deed]['begin_coordinates']
            begin_coordinates = [int(x) for x in begin_coordinates.split(',')]
            dimensions = pagexml.coords.points
            width = dimensions[1][0]
            height = dimensions[2][1]

        else:
            return None

        # Set the box size and estimate the middle of the scan
        box_depth = 1500
        halfpage_mark = width / 2 - 150
        
        # If the begin coordinates are on the left side of the page, the first lines are on the left side
        if begin_coordinates[0] < (halfpage_mark):
            right_boundary = halfpage_mark
            first_line_box = box(begin_coordinates[0], begin_coordinates[1], halfpage_mark, begin_coordinates[1]+box_depth)
        
        # If the begin coordinates are on the right side of the page, the first lines are on the right side
        else:
            right_boundary = width
            first_line_box = box(begin_coordinates[0], begin_coordinates[1], right_boundary, begin_coordinates[1]+box_depth)
       
        # Now that we have the box, we can extract the text (using the overlap function)
        fulltext = []
        fullcoords = []
        for line in pagexml.get_lines():
            textlinecoord = line.coords.points
            if line.text is not None:

                if overlap(textlinecoord, first_line_box, margin=100):
                    fulltext.append(line.text)
                    fullcoords.append(textlinecoord)

        fulltext = " ".join(fulltext)
        
        return(fulltext, fullcoords, [width, height])
    





class Sailor:
    """
    The Sailor class represents a sailor with a reference to the corresponding deed, and metadata such as URI, name, location, role, organization, and ship name.
    It also contains a dictionary 'check' to store algorithm input for manual verification of the output.
    """

    def __init__(self, deed, sailor_uri=None, name=None, location=None, location_uri=None, location_htr=None, role=None, role_htr=None, organization=None, organization_htr=None, shipname=None, shipname_htr=None):
        self.deed = deed
        self.sailor_uri = sailor_uri
        self.name = name
        self.location = location
        self.location_uri = location_uri
        self.location_htr = location_htr
        self.role = role
        self.role_htr = role_htr
        self.organization = organization
        self.organization_htr = organization_htr
        self.shipname = shipname
        self.shipname_htr = shipname_htr
        self.check = {} # Dictionary to store algorithm input for manual check




class DeedProcessor:
    def __init__(self):
        self.deeds = []
        # Read index of location names in all notarial deeds
        # This file includes information on the coordinates of the location names
        self.locations = {}
        with open('../data/index/locations.csv') as f:
            for c, line in enumerate(csv.DictReader(f)):
                # Get deed URI
                id = line['id'].split('?location')
                deed_uri = id[0]
                try:
                    location_id = id[1].lstrip('=')
                except:
                    pass
                
                if deed_uri in self.locations:
                    uricount += 1
                else:
                    uricount = 0
                    self.locations[deed_uri] = {}  
                
                self.locations[deed_uri][location_id] = {}
                self.locations[deed_uri][location_id]['label'] = line['label']
                self.locations[deed_uri][location_id]['xywh'] = line['xywh']
                self.locations[deed_uri][location_id]['scanname'] = line['scanname']


    def load_from_csv(self, csv_file):
        temp_index = {}
        with open(csv_file) as f:
            for c, line in enumerate(csv.DictReader(f)):
                temp_index[c] = dict(line)

        deed_index = {}

        for _, v in temp_index.items():
            uri = v['akteIndex']

            if uri not in deed_index:
                deed_index[uri] = {}
                deed_index[uri]['date'] = v['date']
                deed_index[uri]['subject'] = v['onderwerpsomschrijving']
                deed_index[uri]['persons'] = {}

            person = v['person']

            if person not in deed_index[uri]['persons']:
                deed_index[uri]['persons'][person] = {}
                deed_index[uri]['persons'][person]['label'] = v['personName']
                deed_index[uri]['persons'][person]['coordinates'] = [(v['coordinates'], v['scanName'])]
            else:
                deed_index[uri]['persons'][person]['coordinates'].append((v['coordinates'], v['scanName']))


     
        # Add location information per deed        
        for k,v in deed_index.items():
            
            if k in self.locations:
                
                # Add location dict for this key to index_schaef_per_deed
                current_location = self.locations[k]
                deed_index[k]['locations'] = {}

                for loc in current_location:
                    loc_uri = str(k) + '?location=' + str(loc)
                    deed_index[k]['locations'][loc_uri] = {}
                    deed_index[k]['locations'][loc_uri]['label'] = current_location[loc]['label']
                    deed_index[k]['locations'][loc_uri]['coordinates'] = [(current_location[loc]['xywh'], current_location[loc]['scanname'])]

        # Convert deed_index to Deed objects and append to self.deeds
        for deed_uri, deed_data in deed_index.items():
            date = deed_data.get('date', None)
            subject = deed_data.get('subject', None)
            persons = deed_data.get('persons', None)
            locations = deed_data.get('locations', None)
            deed = Deed(deed_uri, date, subject, persons, locations)
            self.deeds.append(deed)
 
    def get_all_deeds(self):
        return self.deeds

class Sailor:
    """
    The Sailor class represents a sailor with a reference to the corresponding deed, and metadata 
    such as URI, name, location, role, organization, and ship name. It also contains a dictionary 
    'check' to store algorithm input for manual verification of the output.
    """

    def __init__(self, deed, sailor_uri=None, name=None, location=None, location_uri=None, location_htr=None, role=None, role_htr=None, organization=None, organization_htr=None, shipname=None, shipname_htr=None, creditor_name=None, creditor_uri=None, debt_htr=None):
        self.deed = deed
        self.sailor_uri = sailor_uri
        self.name = name
        self.location = location
        self.location_uri = location_uri
        self.location_htr = location_htr
        self.role = role
        self.role_htr = role_htr
        self.organization = organization
        self.organization_htr = organization_htr
        self.shipname = shipname
        self.shipname_htr = shipname_htr
        self.creditor_name = creditor_name
        self.creditor_uri = creditor_uri
        self.debt_htr = debt_htr
        self.check = {}


class SailorExtractor:
    """
    The SailorExtractor class is responsible for extracting sailors from a list of Deed objects.
    It contains a method 'extract_sailors' which processes the deeds and extracts Sailor instances 
    with the corresponding metadata.
    """

    def __init__(self, deeds):
        self.deeds = deeds
        
        self.roles = ["soldaat", "bootsgezel", "adelborst", "lanssmissaat", "bosschieter", "lansmissaat", "timmerman", "barbier", "onderbarbier", 
                      "soldat", "sergeant", "secretaris", "commissaris", "tamboer", "korporaal", "chirurgijn"]
        self.orgs = ["WIC", "W.I.C.", "VOC", "OIC", "V.O.C.", "Groenlantse Comp", "Admiraliteit", "directeurs", "heren directeuren", "heeren directeuren", "particulier"]
        
        self.schepen_set = self._prepare_schepen_set()

        # Config for the various formulaic pattern searchers

        # Config for formulaic phrase searcher ("varende voor")
        self.config = {
            "char_match_threshold": 0.8,
            "ngram_threshold": 0.6,
            "levenshtein_threshold": 0.7,
            "ignorecase": True,
            "ngram_size": 2,
            "skip_size": 2,
        }

        # Config for instance (name, location) searcher
        self.config2 = {
            "char_match_threshold": 0.6,
            "ngram_threshold": 0.5,
            "levenshtein_threshold": 0.5,
            "ignorecase": True,
            "ngram_size": 2,
            "skip_size": 2,
        }

   
    def _prepare_schepen_set(self):
        """Private method to prepare a set of ship names."""
        schepen_set = set()
        filename = '../data/index/schaef_ships.csv'

        with open(filename, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                if row:
                    scheepsnaam = row[0]
                    scheepsnaam = scheepsnaam.rstrip().lower()
                    schepen_set.add(scheepsnaam)
        return schepen_set
    

    def test_sailors(self):
        """
        Test function
        """
        self.test_config = {
            "char_match_threshold": 0.8,
            "ngram_threshold": 0.6,
            "levenshtein_threshold": 0.7,
            "ignorecase": True,
            "ngram_size": 4,
            "skip_size": 2,
        }

        # Initialize logger
        logger = logging.getLogger(__name__)
        
        # Set signal phrases for sailor searcher
        sailor_searcher = FuzzyPhraseSearcher(self.config)
        sailor_phrases = ["varende voor"]
        sailor_model = PhraseModel(phrases=sailor_phrases)
        sailor_searcher.index_phrase_model(sailor_model)

        # Set signal phrases for sailor searcher
        tsailor_searcher = FuzzyPhraseSearcher(self.test_config)
        tsailor_phrases = ["varende voor"]
        tsailor_model = PhraseModel(phrases=tsailor_phrases)
        tsailor_searcher.index_phrase_model(tsailor_model)

        extracted_sailors = []

        number_to_check = len(self.deeds)
        c = 0

        for deed in self.deeds:
            
            c += 1

            try:
                pagexml = deed.get_pagexml()
            except FileNotFoundError:
                # Log the error message
                logger.error(f"Process {multiprocessing.current_process().name}: The file for the deed with number {c} has not been found.")
                continue
            
            text, fullcoords, dimensions = deed.get_first_lines(pagexml)

            if not text:
                text = ""
               
            # ANALYZING FIRST LINES OF DEED USING FUZZY MATCHING
            # We will now take a closer look at the first lines of the deed. The sailor_searcher object will
            # look for text patterns that might signal the mention of a sailor ("varende voor")
            # Let's take the most likely match, if there is no match, skip this deed    
            match = max(sailor_searcher.find_matches(text), default=None, key=lambda x: x.levenshtein_similarity)

            # If there is no match, sometimes "varende voor" is attached to other words due to HTR errors
            # Try finding this phrase first w/o Phrasesearcher and add spaces before and after the phrase
            if not match:
                if "varende voor" in text:
                    # Add an extra space before and after the phrase
                    text = text.replace("varende voor", " varende voor ")
                    match = max(sailor_searcher.find_matches(text), default=None, key=lambda x: x.levenshtein_similarity)

                elif "varende" in text:
                    text = text.replace("varende", " varende voor ")
                    match = max(sailor_searcher.find_matches(text), default=None, key=lambda x: x.levenshtein_similarity)

            # Try searching for an alternative text pattern
            if not match:
                match = max(tsailor_searcher.find_matches(text), default=None, key=lambda x: x.levenshtein_similarity)

                if match:
                    print(text)
                
    def extract_sailors(self):
        """
        Extracts Sailor instances from the list of Deed objects provided during the SailorExtractor initialization.
        
        This method processes each deed, retrieves the corresponding PageXML, and extracts the first few lines of the deed.
        It then performs fuzzy matching to identify sailors and their metadata, such as name, location, role, organization, and ship name.
        Finally, it returns a list of extracted Sailor instances with the corresponding metadata and additional information for manual checks.
        
        Returns:
            list: A list of Sailor instances extracted from the deeds.
        """

        # Initialize logger
        logger = logging.getLogger(__name__)
        
        # Set signal phrases for sailor searcher
        sailor_searcher = FuzzyPhraseSearcher(self.config)
        sailor_phrases = ["varende voor", "vare voor", "estvarende voor", "oouvarende voor", "van voor", "voren opt' schip"]
        sailor_model = PhraseModel(phrases=sailor_phrases)
        sailor_searcher.index_phrase_model(sailor_model)

        # Alternative sailor searcher if the first one doesn't find anything
        sailor_alt_searcher = FuzzyPhraseSearcher(self.config)
        sailor_alt_phrases = ["onder capitein", "onder cap", "gaende voor", "varen na", "varen voor", "opt schip", "op t' schif"]
        sailor_alt_model = PhraseModel(phrases=sailor_alt_phrases)
        sailor_alt_searcher.index_phrase_model(sailor_alt_model)

        # Set signal phrases for creditor searcher
        creditor_searcher = FuzzyPhraseSearcher(self.config)
        creditor_phrases = ["wesen aan"]
        creditor_model = PhraseModel(phrases=creditor_phrases)
        creditor_searcher.index_phrase_model(creditor_model)

        name_searcher = FuzzyPhraseSearcher(self.config2)
        location_searcher = FuzzyPhraseSearcher(self.config2)

        # Set no sailor job phrases for sailor job searcher (after these phrases, the job is not mentioned)
        no_sailor_job_phrases = ["voren opt' schip", "onder capitein", "onder cap", "varen na", "opt schip", "op t' schif"]

        # Set pattern to extract ship names, organizations and debt amounts using regex
        ship_pattern = r"(?:op|opt|op't|op t) ('t )?(\b(?:schip|Schip)\b\s*(?:de|d')?\s*\w+(?:\s*\w+)?)"
        org_pattern = r"(?:\b(?:Oost|West|west)[-\s]?Ind(?:e|ische)(?:[.]? Comp.?)?\b)"
        debt_pattern = r"de\s?(s|z)om{1,2}e van(.*?(?:(?=\sover)|[es]?[a]?r?:?\s?g[lu]?\.?|car[oe]?l[ui]?s? g[lu]?[sd]?.?))"

        extracted_sailors = []

        number_to_check = len(self.deeds)
        c = 0

        for deed in self.deeds:
            
            c += 1

            # Log progress information with timestamps and process ID
            logger.info(f"Process {multiprocessing.current_process().name}: Extracting sailors from deed {c} of {number_to_check}")
            
            try:
                pagexml = deed.get_pagexml()
            except FileNotFoundError:
                # Log the error message
                logger.error(f"Process {multiprocessing.current_process().name}: The file for the deed with number {c} has not been found.")
                continue
            
            text, fullcoords, dimensions = deed.get_first_lines(pagexml)

            if not text:
                text = ""
               
            # Get a list of sailors and locations mentioned in this deed
            name_to_uri = {person.label: person.person_uri for person in deed.persons}
            location_to_uri = {location.label.replace('?', ''): location.location_uri for location in deed.locations} # Remove question marks from location names as fuzzy_search doesn't like them

            # ANALYZING FIRST LINES OF DEED USING FUZZY MATCHING
            # We will now take a closer look at the first lines of the deed. The sailor_searcher object will
            # look for text patterns that might signal the mention of a sailor ("varende voor")
            # Let's take the most likely match, if there is no match, skip this deed    
            match = max(sailor_searcher.find_matches(text), default=None, key=lambda x: x.levenshtein_similarity)

            # If there is no match, sometimes "varende voor" is attached to other words due to HTR errors
            # Try finding this phrase first w/o Phrasesearcher and add spaces before and after the phrase
            if not match:
                if "varende voor" in text:
                    # Add an extra space before and after the phrase
                    text = text.replace("varende voor", " varende voor ")
                    match = max(sailor_searcher.find_matches(text), default=None, key=lambda x: x.levenshtein_similarity)

                elif "varende" in text:
                    text = text.replace("varende", " varende voor ")
                    match = max(sailor_searcher.find_matches(text), default=None, key=lambda x: x.levenshtein_similarity)

            # Try searching for an alternative text pattern
            if not match:
                match = max(sailor_alt_searcher.find_matches(text), default=None, key=lambda x: x.levenshtein_similarity)
            
            # Reset variables
            interesting_text = ""
            interesting_text_after = ""
            sailor_name = None
            sailor_uri = None
            sailor_location = None
            sailor_location_uri = None
            sailor_location_htr = None
            role = None
            sailor_role_htr = None
            org = None
            org_htr = None
            shipname = None
            shipname_htr = None
            creditor_name = None
            creditor_uri = None
            debt_htr = None

            if match:
                
                interesting_text = text[match.offset-70:match.offset].replace("Schaef", "") # If Schaef himself is mentioned, delete mention to avoid confusing him with a sailor
                interesting_text_after = text[match.offset:match.offset+95]

                # Let's see if the one of the names mentioned in the index is mentioned in the deed as a sailor
                name_searcher = FuzzyPhraseSearcher(self.config2)
                name_phrases = list(name_to_uri.keys())
                name_model = PhraseModel(phrases=name_phrases)
                name_searcher.index_phrase_model(name_model)
                best_sailor_name_match = max(name_searcher.find_matches(interesting_text), default=None, key=lambda x: x.levenshtein_similarity)
                
                if best_sailor_name_match:
                    sailor_name = best_sailor_name_match.phrase.phrase_string
                    sailor_uri = name_to_uri.get(sailor_name)
            
                # Now let's see if we can find the birthplace for this sailor. It is likely the place mentioned in the 
                # 'interesting text'. So let's compare this text to the indexed locations from this deed
                # create a list of domain keywords and phrases
                location_searcher = FuzzyPhraseSearcher(self.config2)
                location_phrases = list(location_to_uri.keys())
                location_model = PhraseModel(phrases=location_phrases)
                location_searcher.index_phrase_model(location_model)
                best_location_match = max(location_searcher.find_matches(interesting_text), default=None, key=lambda x: x.levenshtein_similarity)

                # If no location is found, try 'Amsterdam' (which was not manually indexed)
                if not best_location_match:
                    location_phrases = ['amsterdam']
                    location_model = PhraseModel(phrases=location_phrases)
                    location_searcher.index_phrase_model(location_model)
                    best_location_match = max(location_searcher.find_matches(interesting_text), default=None, key=lambda x: x.levenshtein_similarity)
                            
                if best_location_match:
                    sailor_location = best_location_match.phrase.phrase_string
                    sailor_location_uri = location_to_uri.get(sailor_location)
            
                # Let's also see if we can extract the role using only the HTR, without the index
                if match.phrase.phrase_string not in no_sailor_job_phrases:
                    relevant_fragment = interesting_text_after.lstrip(match.string)
                    if len(relevant_fragment) > 0: # Check if the list is not empty
                        sailor_role_htr = relevant_fragment.split()[0]
                        sailor_role_htr = check_for_empty(sailor_role_htr)
                        if sailor_role_htr:
                            if len(sailor_role_htr) == 0:
                                sailor_role_htr = None

                # Also try the organization using regex
                org_htr = re.findall(org_pattern, interesting_text_after, re.IGNORECASE)
                if len(org_htr) > 0:
                    org_htr = org_htr[0]
                else:
                    org_htr = None

                # Now let's see if we can extract the location using only the HTR, without the index
                relevant_fragment = interesting_text.split('van')
                if len(relevant_fragment) > 1: # Check if the list is not empty
                    relevant_fragment = relevant_fragment[-1].split()
                    if len(relevant_fragment) > 0: # Check if the string is not empty
                        if relevant_fragment[0].lower() == "st.":
                            sailor_location_htr = relevant_fragment[0] + " " + relevant_fragment[1]

                        elif len(relevant_fragment) > 2 and relevant_fragment[1].lower() == "in":
                            sailor_location_htr = relevant_fragment[0] + " " + relevant_fragment[1] + " " + relevant_fragment[2]

                        else:
                            sailor_location_htr = relevant_fragment[0]

                # Finally, let's try and extract the name of the ship from the HTR, using regex
                shipname_htr = re.findall(ship_pattern, text, re.IGNORECASE)
                if shipname_htr:
                    shipname_htr = shipname_htr[0][1]
                    shipname_htr = re.sub(r'\b(schip|in)\b', '', shipname_htr, flags=re.IGNORECASE).strip()
                    shipname_htr = re.sub(r"\b\w*diens\w*\b", '', shipname_htr, flags=re.IGNORECASE).strip()
                else:
                    shipname_htr = None

            # The subject may have information on the role of the sailor, the organization he was working for and the shipname
            if hasattr(deed, "subject"):
                role = next((role for role in self.roles if role in deed.subject.lower()), None)
                org = next((org for org in self.orgs if org.lower() in deed.subject.lower()), None)
                possible_ship_mention = deed.subject.lower().split("schip")
                if len(possible_ship_mention) > 1:
                    shipname = next((shipname for shipname in self.schepen_set if shipname in possible_ship_mention[1].lower()), None)            
      
            # Now let's try and find the creditor and the debt amount
            match = max(creditor_searcher.find_matches(text), default=None, key=lambda x: x.levenshtein_similarity)

            if match:
                possible_creditor_mention = text[match.offset:match.offset+50]
                print(possible_creditor_mention)
                best_creditor_name_match = max(name_searcher.find_matches(possible_creditor_mention), default=None, key=lambda x: x.levenshtein_similarity)
                print(best_creditor_name_match)
                
                if best_creditor_name_match:
                    creditor_name = best_creditor_name_match.phrase.phrase_string
                    creditor_uri = name_to_uri.get(creditor_name)

                possible_debt_mention = text[match.offset+20:match.offset+200]
                print(possible_debt_mention)
                debt_match = re.search(debt_pattern, possible_debt_mention, re.IGNORECASE)
                if debt_match:
                    debt_htr = debt_match.group(2).strip()

            # Create Sailor instances with the extracted information
            sailor = Sailor(
                deed=deed,
                name= sailor_name,
                sailor_uri=sailor_uri,
                location=sailor_location,
                location_uri=sailor_location_uri,
                location_htr=sailor_location_htr,
                role = role,
                role_htr = sailor_role_htr,
                organization = org,
                organization_htr = org_htr,
                shipname = shipname,
                shipname_htr = shipname_htr,
                creditor_name = creditor_name,
                creditor_uri = creditor_uri,
                debt_htr = debt_htr
            )

            sailor.check =  {
                'interesting_text': interesting_text,
                'interesting_text_after': interesting_text_after,
                'possible_person_labels': list(name_to_uri.keys()),
                'possible_location_labels': list(location_to_uri.keys()),
                'subject': deed.subject if hasattr(deed, "subject") else "",
                'full_text': text,
                'full_coords': calculate_bounding_rectangle(fullcoords) if fullcoords else None,
                'dimensions': dimensions if dimensions else None,
                # 'image_url': this_image_url
            }

            extracted_sailors.append(sailor)
        
        return extracted_sailors
        
    def parallel_extract_sailors(self):

        num_cores = multiprocessing.cpu_count() - 1
        if num_cores <= 0:
            num_cores = 1  # Ensure at least one core is used

        if len(self.deeds) <= num_cores:
            chunk_size = 1  # Process each deed in a separate chunk
        else:
            chunk_size = len(self.deeds) // num_cores

        deeds_chunks = [self.deeds[i:i + chunk_size] for i in range(0, len(self.deeds), chunk_size)]

        with multiprocessing.Pool(processes=num_cores) as pool:
            sailor_chunks = pool.map(process_chunk, deeds_chunks)

        extracted_sailors = [sailor for sailor_chunk in sailor_chunks for sailor in sailor_chunk]

        return extracted_sailors



    def extract_sailors_ai(self):
        """
        Extracts Sailor instances from the list of Deed objects provided during the SailorExtractor initialization. 
        
        This method processes each deed, retrieves the corresponding PageXML, and extracts the first few lines of the deed.
        It then uses OpenAI's GPT-3 API to extract the structured data.
        Finally, it returns a list of extracted Sailor instances with the corresponding metadata and additional information for manual checks.
        
        Returns:
            list: A list of Sailor instances extracted from the deeds.
        """

        extracted_sailors = []

        number_to_check = len(self.deeds)
        c = 0
        orgs_short = ["VOC", "WIC"]

        functions = [
        {
            "name": "extract_sailor_info",
            "description": "Extract metadata information from a given text about a sailor's debt situation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_of_sailor": {
                        "type": "string",
                        "description": "Name of the sailor/debtor",
                    },
                    "ship": {
                        "type": "string",
                        "description": "Ship in which he sails (if mentioned)",
                    },
                    "birthplace": {
                        "type": "string",
                        "description": "Birthplace of the sailor",
                    },
                    "job": {
                        "type": "string",
                        "description": "Job of the sailor (i.e. bosschieter, matroos)",
                    },
                    "creditor": {
                        "type": "string",
                        "description": "Name of the creditor",
                    },
                    "captain": {
                        "type": "string",
                        "description": "Name of the captain",
                    },
                    "debt_amount_source": {
                        "type": "string",
                        "description": "Debt amount from source",
                    },
                    "debt_amount_int": {
                        "type": "integer",
                        "description": "Debt amount in integer (best guess based on mentioned amount)",
                    },
                },
            }
        }
    ]

        for deed in self.deeds:
            
            c += 1
            print(f"Extracting sailors from deed {c} of {number_to_check}", end="\r")
            pagexml = deed.get_pagexml()
            if not pagexml:
                continue
            
            text, fullcoords, dimensions = deed.get_first_lines(pagexml)

            if not text:
                continue            
            
            # Get a list of sailors and locations mentioned in this deed
            name_to_uri = {person.label: person.person_uri for person in deed.persons}
            location_to_uri = {location.label.replace('?', ''): location.location_uri for location in deed.locations}

            # Generate the prompt
            prompt = f"""   
                    Below is the Dutch text of a notarial deed (imperfect output of an HTR model). Extract:
                    -the sailor
                    -his function
                    -where he is from
                    -to whom he owes money (likely owed in carolus guldens or car: gls)
                    -how much (exactly as written in the deed and in integers)
                    -for what company
                    -on which ship (if applicable)
                    -under what captain (if applicable)

                    For names and locations, only choose from the lists below (unless the name/location in the deed really doesn't resemble any in the lists).
                    Indexed names: {list(name_to_uri.keys())}
                    Indexed locations: {list(location_to_uri.keys())}

                    For organizations, only choose from the list below (unless the organization in the deed really doesn't resemble any in the list). Anything with 'oost' is VOC, with 'west' is WIC.
                    Indexed organizations: {orgs_short}

                    If a ship name is mentioned in the metadata, pick that one if it resembles the ship name in the deed.
                    Indexed metadata: {deed.subject if hasattr(deed, "subject") else ""}
                    
                    *****
                    Deed text:

                    {text}
            """
           
            print(prompt)

            # messages = []
            # messages.append({"role": "system", "content": "Only use the fields as indicated."})
            # messages.append({"role": "user", "content": prompt})
            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=messages, 
            #     functions=functions
            # )
            # metadata_returned = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])

            # print(metadata_returned)

        return extracted_sailors
        