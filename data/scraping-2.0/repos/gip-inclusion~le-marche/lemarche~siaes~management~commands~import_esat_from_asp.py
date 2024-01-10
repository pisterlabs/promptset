import csv
import os
import time

import openai
from django.conf import settings
from django.core.management.base import BaseCommand

from lemarche.siaes import constants as siae_constants
from lemarche.siaes.models import Siae
from lemarche.utils.apis.geocoding import get_geocoding_data
from lemarche.utils.constants import DEPARTMENT_TO_REGION, department_from_postcode
from lemarche.utils.data import rename_dict_key


FILE_NAME = "Annuaire_ESAT_20230717.csv"
FILE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/" + FILE_NAME
FIELD_NAME_LIST = [
    "Département",
    "Dénomination",
    "N° de Siret",
    "Adresse",
]

# openai client configuration
openai.organization = settings.OPENAI_ORG
openai.api_base = settings.OPENAI_API_BASE
openai.api_key = settings.OPENAI_API_KEY
aimodel = settings.OPENAI_MODEL


def read_csv():
    esat_list = list()

    with open(FILE_PATH) as csv_file:
        # Header : "Département","Dénomination","N° de Siret","Adresse"
        csvreader = csv.DictReader(csv_file, delimiter=",")
        for index, row in enumerate(csvreader):
            esat_list.append(row)

    return esat_list


class Command(BaseCommand):
    """
    Usage: poetry run python manage.py import_esat_from_asp
    """

    def handle(self, *args, **options):
        print("-" * 80)
        esat_list = read_csv()

        print("Importing ESAT FROM ASP...")
        progress = 0

        already_exits = 0
        address_changes = 0
        news = 0

        for index, esat in enumerate(esat_list):
            progress += 1
            if (progress % 50) == 0:
                print(f"{progress}...")

            esat_siret = esat["N° de Siret"]
            esat_denom = esat["Dénomination"]
            if siae := Siae.objects.filter(siret=esat_siret).first():
                already_exits += 1

                address_in_db = f"{siae.address} {siae.post_code} {siae.city}".strip().lower()
                address_in_file = esat["Adresse"].strip().lower()

                # IA used to check if the address has really changed
                if address_in_file != address_in_db:
                    prompt = (
                        f'By answering yes or no, tell me if these two addresses, "{address_in_db}" '
                        f'and "{address_in_file}", refer to the same place ?'
                    )
                    messages = [{"role": "user", "content": prompt}]

                    has_answered = False
                    while not has_answered:
                        try:
                            chat_completion = openai.ChatCompletion.create(
                                model=aimodel, temperature=0.5, max_tokens=150, messages=messages, request_timeout=15
                            )
                            has_answered = True
                            result = chat_completion.to_dict_recursive()
                            if result["choices"][0]["message"]["content"].strip().startswith("No"):
                                address_changes += 1
                                self.update_esat_address(siae, address_in_file)
                        except:  # noqa E722
                            print("OpenAI API Timeout, sleep before retry")
                            time.sleep(3)
                print(f"{esat_denom} ({esat_siret}) addess change : {address_in_db} -> {address_in_file}")
            else:
                news += 1
                print(f"{esat_denom} ({esat_siret}) is a new !")
                self.import_esat(esat)

            # avoid DDOSing APIs
            time.sleep(1)

        print(
            f"Done with {already_exits} already_exits ({address_changes} addresses updated) and {news} new esat added."
        )

    def update_esat_address(self, siae, address):
        geocoding_data = get_geocoding_data(address)
        if geocoding_data:
            print(geocoding_data)
            siae.address = geocoding_data["address"]
            siae.post_code = geocoding_data["post_code"]
            siae.city = geocoding_data["city"]
            siae.department = department_from_postcode(geocoding_data["post_code"])
            siae.region = DEPARTMENT_TO_REGION[siae.department]
            siae.coords = geocoding_data["coords"]
            siae.save()
        else:
            print(f"Geocoding not found,{siae.name},{address}")

    def import_esat(self, esat):  # noqa C901
        # store raw dict
        esat["import_source"] = "esat_asp"
        esat["import_raw_object"] = esat.copy()

        # defaults
        esat["kind"] = siae_constants.KIND_ESAT
        esat["source"] = siae_constants.SOURCE_ESAT
        esat["geo_range"] = siae_constants.GEO_RANGE_DEPARTMENT

        # basic fields
        rename_dict_key(esat, "Dénomination", "name")
        esat["name"].strip()
        esat["name"] = esat["name"].replace("  ", " ")
        rename_dict_key(esat, "N° de Siret", "siret")
        esat["siret_is_valid"] = True

        full_address = esat.pop("Adresse")

        # create object
        try:
            print("Create new esat..")
            [esat.pop(key) for key in ["import_source", "Département"]]
            print(esat)
            siae = Siae.objects.create(**esat)
            self.update_esat_address(siae, full_address)
        except Exception as e:
            print(e)
            print(esat)

        # avoid DDOSing APIs
        time.sleep(0.3)
