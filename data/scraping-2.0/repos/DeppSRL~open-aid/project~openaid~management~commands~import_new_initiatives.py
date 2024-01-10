# coding=utf-8
__author__ = 'stefano'
import logging
from django.core.exceptions import ObjectDoesNotExist
from openpyxl import load_workbook
from django.core.management.base import BaseCommand
from openaid.projects.models import Initiative
from openaid.codelists.models import Recipient, Sector


class Command(BaseCommand):
    option_list = BaseCommand.option_list

    help = 'import data for new initiatives from xls. 4 dec 2015 only'
    logger = logging.getLogger('openaid')
    stash_codici = []

    known_recipients = {
        'Rep. Centro-Africana': '231',
        'Siria (Rep. Araba)': '573',
        'Bosnia E Erzegovina': '64',
        'Non Ripartibile': '998',
        'Yemen (Rep.)': '580',
        'Sahel': '289',
    }

    def get_code(self, row):
        code = None
        zfill_code = None
        value = row[0].value
        if type(value) == int:
            code = value
        if type(value) == float:
            try:
                code = int(value)
            except TypeError:
                return None, None

        zfill_code = str(code).zfill(6)
        return code, zfill_code

    def check_uniqueness(self, ws):
        ret = False
        for row_counter, row in enumerate(ws.rows):
            if row_counter == 0:
                continue
            code, zfill_code = self.get_code(row)
            if code is None:
                continue

            if zfill_code in self.stash_codici:
                self.logger.error("Row:{} - Codice '{}' non univoco!".format(row_counter, code))
                ret = True
            else:
                self.stash_codici.append(zfill_code)
        return ret

    def import_iniziative(self, ws):
        for row_counter, row in enumerate(ws.rows):
            if row_counter == 0:
                continue

            code, zfill_code = self.get_code(row)
            if code is None:
                continue
            try:
                Initiative.objects.get(code=zfill_code)
            except ObjectDoesNotExist:
                pass
            else:
                self.logger.error("Initiative {} already in DB, skip".format(zfill_code))
                continue

            recipient_stringa = row[1].value.strip().title()
            try:
                recipient = Recipient.objects.get(name_it=recipient_stringa)
            except ObjectDoesNotExist:
                rcode = self.known_recipients[recipient_stringa]
                try:
                    recipient = Recipient.objects.get(code=rcode)
                except ObjectDoesNotExist:
                    self.logger.error(u"Recipient {} does not exist".format(recipient_stringa))
                    continue

            self.logger.debug(u"Recipient {} found".format(recipient_stringa))

            initiative = Initiative()
            initiative.code = zfill_code
            initiative.recipient_temp = recipient
            initiative.title_it = row[2].value
            initiative.total_project_costs = row[7].value
            initiative.grant_amount_approved = row[6].value
            initiative.loan_amount_approved = row[5].value

            purpose_code = str(row[4].value).replace(".0","")
            try:
                initiative.purpose_temp = Sector.objects.get(code=purpose_code)
            except ObjectDoesNotExist:
                self.logger.error("Purpose {} not in DB, skip".format(purpose_code))
                continue
            else:
                self.logger.debug("Purpose {} found".format(purpose_code))
                if initiative.purpose_temp.parent is None:
                    self.logger.error("Purpose '{}' does NOT have a parent (sector), skip".format(purpose_code))
                    continue
            self.logger.info("Inserted initiative {}".format(initiative.code))
            initiative.save()



    def handle(self, *args, **options):
        verbosity = options['verbosity']
        input_filename = 'resources/fixtures/IniziativeRichiesta021215.xlsx'
        if verbosity == '0':
            self.logger.setLevel(logging.ERROR)
        elif verbosity == '1':
            self.logger.setLevel(logging.WARNING)
        elif verbosity == '2':
            self.logger.setLevel(logging.INFO)
        elif verbosity == '3':
            self.logger.setLevel(logging.DEBUG)

        self.logger.info(u"Opening input file: {}".format(input_filename))
        input_file = open(input_filename, 'rb')
        input_workbook = load_workbook(input_file, data_only=True, use_iterators=True)
        ws_da_importare = input_workbook['No Openaid Si Excel']

        self.logger.info("Checking uniqueness of codes in the file")
        # check that codes are unique in the whole file, initiatives cannot be repeated
        ret1 = self.check_uniqueness(ws_da_importare)

        if ret1:
            self.logger.critical("Codes are not unique in the file. Quitting")
            exit()
        else:
            self.logger.info("All codes are unique")

        self.logger.info("Import iniziative")
        # deal with in corso initiatives
        self.import_iniziative(ws_da_importare)
        self.logger.info(u"Finish")