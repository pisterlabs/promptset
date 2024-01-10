__author__ = 'stefano'
# coding=utf-8
from collections import OrderedDict
import logging
import zipfile
from pprint import pprint
from optparse import make_option
from datetime import datetime
from django.core.management.base import BaseCommand
from openaid.utils import UnicodeDictWriter
from openaid.projects.models import Activity
from openaid.codelists.models import Recipient


class Command(BaseCommand):

    help = 'Esporta i crs zippati'
    encoding = 'utf-8'
    output_filename = 'CRS_6_{}.csv'
    logger = logging.getLogger('openaid')

    option_list = BaseCommand.option_list + (
        make_option('--years',
                    dest='years',
                    default='',
                    help='Years to fetch. Use one of this formats: 2012 or 2003-2006 or 2002,2004,2006'),
        make_option('--compress',
                    dest='compress',
                    action='store_true',
                    default=False,
                    help="Generate compressed zip archive"),
    )

    # mapping fields from CSV name to DB name
    field_map = OrderedDict([
        ('year', 'year'),
        ('donorcode', 'project__agency__donor__code'),
        ('donorname', 'project__agency__donor__name'),
        ('agencycode', 'project__agency__code'),
        ('agencyname', 'project__agency__name'),
        ('crsid', 'project__crsid'),
        ('projectnumber', 'number'),
        ('initialreport', 'report_type'),
        ('recipientcode', 'project__recipient__code'),
        ('recipientname', 'project__recipient__name'),
        ('regioncode', 'project__recipient__parent__code'),
        ('regioname', 'project__recipient__parent__name'),
        ('incomegroupcode', 'project__recipient__income_group'),
        ('flowcode', 'flow_type'),
        ('bi_multi', 'bi_multi'),
        ('finance_t', 'project__finance_type'),
        ('aid_t', 'project__aid_type'),
        ('usd_commitment', 'commitment_usd'),
        ('usd_disbursement', 'disbursement_usd'),
        ('commitment_national', 'commitment'),
        ('disbursement_national', 'disbursement'),
        ('shortdescription', 'project__description'),
        ('projecttitle', 'project__title'),
        ('purposecode', 'project__sector__code'),
        ('purposename', 'project__sector__name'),
        ('sectorcode', 'project__sector__parent__code'),
        ('sectorname', 'project__sector__parent__name'),
        ('channelcode', 'project__channel__code'),
        ('channelname', 'project__channel__name'),
        ('channelreportedname', 'channel_reported__name'),
        ('geography', 'geography'),
        ('expectedstartdate', 'expected_start_date'),
        ('completiondate', 'completion_date'),
        ('longdescription', 'long_description'),
        ('gender', 'project__markers__gender'),
        ('environment', 'project__markers__environment'),
        ('trade', 'project__markers__trade'),
        ('pdgg', 'project__markers__pd_gg'),
        ('FTC', 'is_ftc'),
        ('PBA', 'is_pba'),
        ('investmentproject', 'is_investment'),
        ('biodiversity', 'project__markers__biodiversity'),
        ('climateMitigation', 'project__markers__climate_mitigation'),
        ('climateAdaptation', 'project__markers__climate_adaptation'),
        ('desertification', 'project__markers__desertification'),
        ('commitmentdate', 'commitment_date'),
        ('numberrepayment', 'number_repayment'),
        ('grantelement', 'grant_element'),
        ('openaid id', 'project__pk'),
    ])

    # fields needed in the csv in the correct order
    csv_fieldset = OrderedDict([
        ('year', 'year'),
        ('donorcode', 'donorcode'),
        ('donorname', 'donorname'),
        ('agencycode', 'agencycode'),
        ('agencyname', 'agencyname'),
        ('crsid', 'crsid'),
        ('projectnumber', 'projectnumber'),
        ('initialreport', 'initialreport'),
        ('recipientcode', 'recipientcode'),
        ('recipientname', 'recipientname'),
        ('regioncode', 'regioncode'),
        ('regioname', 'regioname'),
        ('incomegroupcode', 'incomegroupcode'),
        ('incomegroupname', 'incomegroupname'),
        ('flowname', 'flowname'),
        ('bi_multi', 'bi_multi'),
        ('finance_t', 'finance_t'),
        ('aid_t', 'aid_t'),
        ('usd_commitment', 'usd_commitment'),
        ('usd_disbursement', 'usd_disbursement'),
        ('currencycode', 'currencycode'),
        ('commitment_national', 'commitment_national'),
        ('disbursement_national', 'disbursement_national'),
        ('shortdescription', 'shortdescription'),
        ('projecttitle', 'projecttitle'),
        ('purposecode', 'purposecode'),
        ('purposename', 'purposename'),
        ('sectorcode', 'sectorcode'),
        ('sectorname', 'sectorname'),
        ('channelcode', 'channelcode'),
        ('channelname', 'channelname'),
        ('channelreportedname', 'channelreportedname'),
        ('geography', 'geography'),
        ('expectedstartdate', 'expectedstartdate'),
        ('completiondate', 'completiondate'),
        ('longdescription', 'longdescription'),
        ('gender', 'gender'),
        ('environment', 'environment'),
        ('trade', 'trade'),
        ('pdgg', 'pdgg'),
        ('FTC', 'FTC'),
        ('PBA', 'PBA'),
        ('investmentproject', 'investmentproject'),
        ('biodiversity', 'biodiversity'),
        ('climateMitigation', 'climateMitigation'),
        ('climateAdaptation', 'climateAdaptation'),
        ('desertification', 'desertification'),
        ('commitmentdate', 'commitmentdate'),
        ('numberrepayment', 'numberrepayment'),
        ('grantelement', 'grantelement'),
        ('openaid id', 'openaid id'),
    ])

    def write_file(self, activity_set, filename):
        f = open(filename, "w")

        udw = UnicodeDictWriter(f, fieldnames=self.csv_fieldset.keys(), encoding=self.encoding)
        udw.writerow(self.csv_fieldset)

        for activity in activity_set:
            udw.writerow(activity)
        f.close()


    def date_to_string(self, date):
        date_format = '%d/%m/%Y'
        if date is not None:
            try:
                return datetime.strftime(date, date_format)
            except ValueError:
                self.logger.error("Wrong date value:{}".format(date))
                return ''
        return ''


    def manipulate(self, activity_set):
        # maps the field names for export using the field map (example: "pk" -> "openaid id")
        # adds "display name to few fields"
        # substitute "None" values with ""
        # adds "currencycode" field
        # converts date to string

        mapped_activities = []
        for activity in activity_set:

            mapped_activity = OrderedDict()
            # get income group displayname and flowname
            incomegroupname = ''
            flowname = ''
            if activity['project__recipient__income_group'] != None and activity[
                'project__recipient__income_group'] != '':
                incomegroupname = Recipient.INCOME_GROUPS[activity['project__recipient__income_group']]

            if activity['flow_type'] != None and activity['flow_type'] != '':
                flowname = Activity.FLOW_TYPES[activity['flow_type']]

            # convert dates to string

            activity['expected_start_date'] = self.date_to_string(activity['expected_start_date'])
            activity['completion_date'] = self.date_to_string(activity['completion_date'])
            activity['commitment_date'] = self.date_to_string(activity['commitment_date'])

            for csv_key in self.csv_fieldset.keys():
                value = ''
                db_key = self.field_map.get(csv_key,None)
                if db_key is None:
                    if csv_key == 'currencycode':
                        value = '918'
                    elif csv_key == 'incomegroupname':
                        value = incomegroupname
                    elif csv_key == 'flowname':
                        value = flowname
                    else:
                        raise Exception

                else:
                    value = activity[db_key]

                    if db_key in ['commitment_usd','disbursement_usd','disbursement','commitment'] and type(value) is float:

                        #use different annotation for money values
                        value = format(value, '.12f')

                if value is None:
                    value = u''
                elif value is True:
                    value = u'1'
                elif value is False:
                    value = u'0'

                if type(value) is int or type(value) is float:
                    value = str(value)

                mapped_activity[csv_key] = value

            mapped_activities.append(mapped_activity)
        return mapped_activities

    def compress(self, filename):

        zipfilename = filename+".zip"
        self.logger.info("Compressed file {}".format(zipfilename))
        with zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED) as myzip:
            myzip.write(filename)

    def export(self, year, compress):
        # gets activity from DB, manipulates data, wrties to file
        activity_set = Activity.objects.all().order_by('year','crsid')

        activity_set = activity_set.filter(year=int(year))

        activity_set = activity_set.values(*self.field_map.values())

        activity_set = self.manipulate(activity_set)
        self.logger.info("Exported {} lines".format(len(activity_set)))
        filename = self.output_filename.format(year)
        self.write_file(activity_set, filename)
        if compress is True:
            self.compress(filename)


    def handle(self, *args, **options):

        ###
        # years
        ###
        years = options['years']
        compress = options['compress']

        if not years:
            raise Exception("Missing years parameter")

        if "-" in years:
            (start_year, end_year) = years.split("-")
            years = range(int(start_year), int(end_year) + 1)
        else:
            years = [int(y.strip()) for y in years.split(",") if 2001 < int(y.strip()) < 2020]

        if not years:
            raise Exception("No suitable year found in {0}".format(years))

        self.logger.info("Processing years: {0}".format(years))

        for year in years:
            try:
                self.export(year, compress)
            except KeyboardInterrupt:
                self.logger.error("Command execution aborted.")

        self.logger.info("Finished exporting")
