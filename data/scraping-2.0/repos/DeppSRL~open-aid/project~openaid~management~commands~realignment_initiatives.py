# coding=utf-8
__author__ = 'stefano'
import logging
from optparse import make_option
from pprint import pprint
from django.core.exceptions import ObjectDoesNotExist
from openpyxl import load_workbook
from django.core.management.base import BaseCommand
from openaid.projects.models import Initiative


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--file',
                    dest='file',
                    default='',
                    help='path to input file'),
        make_option('--dry-run',
                    action='store_true',
                    dest='dryrun',
                    default=False,
                    help='do not actually write into the DB'),
    )

    help = 'realign initatives with those in the xls input file'
    logger = logging.getLogger('openaid')
    stash_codici = []
    completed_only_xls = []
    completed_in_xls = []
    corso_only_xls = []
    corso_in_xls = []
    dryrun = True

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


    def convert_list_to_string(self, list):
        return ",".join(list)

    def check_uniqueness(self,ws):
        ret = False
        for row_counter, row in enumerate(ws.rows):
            if row_counter == 0:
                continue
            code, zfill_code = self.get_code(row)
            if code is None:
                continue

            if zfill_code in self.stash_codici:
                self.logger.error("Row:{} - Codice '{}' non univoco!".format(row_counter,code))
                ret = True
            else:
                self.stash_codici.append(zfill_code)
        return ret

    def examinate_completed(self, ws):
        for row_counter, row in enumerate(ws.rows):
            if row_counter == 0:
                continue

            code, zfill_code = self.get_code(row)
            if code is None:
                continue

            if zfill_code not in self.completed_in_xls:
                self.completed_in_xls.append(zfill_code)
            try:
                initiative = Initiative.objects.get(code=zfill_code)
            except ObjectDoesNotExist:
                self.completed_only_xls.append(zfill_code)
                continue
            else:
                total = row[3].value
                grant = row[4].value
                loan = row[5].value
                initiative.status_temp = '100'
                initiative.total_project_costs = total
                initiative.loan_amount_approved = loan
                initiative.grant_amount_approved = grant
                if self.dryrun is False:
                    initiative.save()


    def examinate_in_corso(self, ws):
        for row_counter, row in enumerate(ws.rows):
            if row_counter == 0:
                continue

            code, zfill_code = self.get_code(row)
            if code is None:
                continue
            if zfill_code not in self.corso_in_xls:
                self.corso_in_xls.append(zfill_code)

            try:
                initiative = Initiative.objects.get(code=zfill_code)
            except ObjectDoesNotExist:
                self.corso_only_xls.append(zfill_code)
                continue
            else:
                if initiative.status_temp == '100':
                    self.logger.info("IN CORSO: update status iniziativa:{} to Not available".format(zfill_code))
                    initiative.status_temp = '-'
                    if self.dryrun is False:
                        initiative.save()

    def log_completed(self):
        # print out codes present ONLY in XLS
        if len(self.completed_only_xls) > 0:
            self.logger.error("COMPLETED: codes only in XLS:{}".format(self.convert_list_to_string(self.completed_only_xls)))

        # print out codes present ONLY in DB
        completed_missing_xls = Initiative.objects.filter(status_temp='100').exclude(code__in=self.completed_in_xls).order_by('code').values_list('code',flat=True)
        if len(completed_missing_xls) > 0:
            self.logger.error("COMPLETED: codes only in DB:{}".format(self.convert_list_to_string(completed_missing_xls)))

    def log_in_corso(self):
        # print out codes present ONLY in XLS
        if len(self.corso_only_xls) > 0:
            self.logger.error("IN CORSO: codes only in XLS:{}".format(self.convert_list_to_string((self.corso_in_xls))))

        # print out codes present ONLY in DB
        self.logger.debug("There are {} initiatives in corso in xls".format(len(self.corso_in_xls)))
        corso_missing_xls = Initiative.objects.all().exclude(status_temp='100').exclude( code__in=self.corso_in_xls).order_by('code').values_list('code',flat=True)
        if len(corso_missing_xls) > 0:
            self.logger.error("IN CORSO: codes only in DB:{}".format(self.convert_list_to_string((corso_missing_xls))))

    def check_subsets(self):
        #     check what are the codes only in the XLS, and then check which are the codes only in the DB
        codes_db = set(Initiative.objects.all().order_by('code').values_list('code',flat=True))
        codes_xls = set(self.stash_codici)

        stringa_db = self.convert_list_to_string(codes_db-codes_xls)
        stringa_xls = self.convert_list_to_string(codes_xls-codes_db)

        self.logger.info("DB-XLS:{}".format(stringa_db))
        self.logger.info("XLS-DB:{}".format(stringa_xls))

    def handle(self, *args, **options):
        verbosity = options['verbosity']
        self.dryrun = options['dryrun']
        input_filename = options['file']
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
        input_workbook = load_workbook(input_file, data_only=True, use_iterators = True)
        ws_esecuzione_con_scheda = input_workbook['In esecuzione con scheda']
        ws_esecuzione_senza_scheda = input_workbook['In esecuzione senza scheda']
        ws_completed = input_workbook['Chiuse']

        self.logger.info("Checking uniqueness of codes in the file")
        # check that codes are unique in the whole file, initiatives cannot be repeated
        self.logger.info("Checking iniziative esecuzione con scheda")
        ret1 = self.check_uniqueness(ws_esecuzione_con_scheda)
        self.logger.info("Checking iniziative esecuzione senza scheda")
        ret2 = self.check_uniqueness(ws_esecuzione_senza_scheda)
        self.logger.info("Checking iniziative completed")
        ret3 = self.check_uniqueness(ws_completed)

        if ret1 or ret2 or ret3:
            self.logger.critical("Codes are not unique in the file. Quitting")
            exit()
        else:
            self.logger.info("All codes are unique")

        self.check_subsets()
        # deal with completed initiatives
        self.logger.info("Examinate COMPLETED sheet")
        self.examinate_completed(ws_completed)
        self.logger.info("Examinate IN CORSO sheet")
        # deal with in corso initiatives
        self.examinate_in_corso(ws_esecuzione_con_scheda)
        self.examinate_in_corso(ws_esecuzione_senza_scheda)
        # log the results
        self.log_completed()
        self.log_in_corso()

        self.logger.info(u"finish")
