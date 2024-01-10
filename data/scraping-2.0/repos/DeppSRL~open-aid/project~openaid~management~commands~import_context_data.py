# -*- coding: utf-8 -*-

import logging
from optparse import make_option
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned
from django.core.management.base import LabelCommand
from openaid.codelists.models import Recipient
import json

__author__ = 'guglielmo'



class Command(LabelCommand):
    """
    Read a json file containing iso codes and statistical info from World Bank
    and import these info into the Recipient model.

    The json file is exported from the FTM project, through dumpdata

        python manage.py dumpdata --indent=4 ftm.Territorio > ../territori.json

    And the matching is done through the PK field,
    which corresponds to the codice field in the Recipient module
    """
    args = '<json_file json_file ...>'
    help = 'Speficica i json file da importare'

    logger = logging.getLogger('management')

    option_list = LabelCommand.option_list + (
        make_option('--dry-run',
                    action='store_true',
                    dest='dryrun',
                    default=False,
                    help='do not actually write into the DB'),
    )

    def handle_label(self, json_filename, **options):
        json_data=open(json_filename).read()
        countries = json.loads(json_data)

        verbosity = options['verbosity']
        if verbosity == '0':
            self.logger.setLevel(logging.ERROR)
        elif verbosity == '1':
            self.logger.setLevel(logging.WARNING)
        elif verbosity == '2':
            self.logger.setLevel(logging.INFO)
        elif verbosity == '3':
            self.logger.setLevel(logging.DEBUG)

        for country in countries:
            country_code = country['pk']
            country_fields = country['fields']
            try:
                if not options['dryrun']:
                    recipient = Recipient.objects.get(code=country_code)
                    recipient.iso_alpha2 = country_fields['iso_alpha2']
                    recipient.iso_code = country_fields['iso_code']

                    if country_fields['popolazione'] is not None:
                        recipient.popolazione = country_fields['popolazione']

                    if country_fields['crescita_popolazione'] is not None:
                        recipient.crescita_popolazione = country_fields['crescita_popolazione']

                    if country_fields['pil'] is not None:
                        recipient.pil = country_fields['pil']

                    if country_fields['pil_procapite'] is not None:
                        recipient.pil_procapite = country_fields['pil_procapite']

                    recipient.save()
                    self.logger.info(u"Aggiornato il Recipient corrispondente a {0} ({1}).".format(country_fields['label'], country_code))
            except ObjectDoesNotExist:
                self.logger.warning(u"Impossibile trovare un Recipient corrispondente a {0} ({1}).".format(country_fields['label'], country_code))
                continue
            except MultipleObjectsReturned:
                self.logger.warning(u"Trovati più di un Recipient corrispondente a {0} ({1})".format(country_fields['label'], country_code))
                continue
