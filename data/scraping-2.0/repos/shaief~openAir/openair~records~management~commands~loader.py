 # -*- coding: utf-8 -*-
"""
script to load old data from svivaaqm.net

Health Warning :-/ Stinky code ahead of you. Read on your own risk.
The code is still not documented, does and not comply to pep8, hence
it stinks.

It is a work in progress.


"""
import pandas as pd
import pytz
from datetime import datetime
from openair.settings.base import TIME_ZONE
from openair.records.models import Station, Record
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    args = '<station_csv> <url_id>'
    help = 'Add data from file station_csv to database, the record given'\
        + ' in the file are created with url_id as station identifier'

    def handle(self, *args, **options):
        csv_file, url_id = args[0], arg[1]
        station = Station.objects.get(url_id=url_id)
        data = pd.read_csv(csv_file, index_col=[20])
        local = pytz.timezone(TIME_ZONE)

        for item in data.columns:
            param = getattr(data, item)
            for date, value in param.itemitems():

                # the hour 24:00 is not allowed
                date = date.replace(' 24:', ' 00:')
                naive_timestamp = datetime.strptime(date, '%d/%m/%Y %H:%M')
                timestamp = local.localize(naive_timestamp)

                #Record.objects.get_or_create(
                #        station=station,
                #        parameter=value,
                #        value=results[station_url_id][abbr],
                #        timestamp=timestamp
                #        )

#record, success = Record.objects.get_or_create(
#                        station=station,
#                        parameter=parameter,
#                        value=results[station_url_id][abbr],
#                        timestamp=timestamp
#                    )
# station url_id should be given as cli parameter


#a = pd.read_csv('openair/StationData1.csv', index_col=[20])


#for item in a.columns:
#    param = getattr(a, item)
#    for date, val in param.iteritems():
#        Record.objects.get_or_create(station=STATION,
#                                     parameter=item,
#                                     value=val,
#                                     timestamp=date)
