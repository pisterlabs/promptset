from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
import pytz
from openair.settings.base import TIME_ZONE
from openair.records.models import Station
from openair.records.scraper import scrape_station_info, ScrapingError


class Command(BaseCommand):
    help = 'After run_scraper use this command to scrape stations \
           information and populate the DB'

    def handle(self, *args, **options):

        local = pytz.timezone(TIME_ZONE)

        # run over all of the stations
        for station in Station.objects.all():

            try:
                results = scrape_station_info(station.url_id)
            except ScrapingError, e:
                print(e)
                continue

            # print(station.url_id)
            # for k in results.keys():
            #     print(u'{}\t{}'.format(k, results[k]))

            invalid_dates = ['', '01/01/2000']
            invalid_height = 0

            station.name = results['name']
            station.location = results['location']
            station.owners = results['owners']

            if not results['date_of_founding'] in invalid_dates:
                naive_datetime = datetime.strptime(
                    results['date_of_founding'], '%d/%m/%Y'
                )

                date_of_founding = local.localize(naive_datetime)
                station.date_of_founding = date_of_founding

            station.lon = results['lon']
            station.lat = results['lat']

            if results['height']  != invalid_height:
                station.height = results['height']

            station.save()

        return

        # run over the received stations
        for station_url_id in results.keys():

            station, success = Station.objects.get_or_create(
                url_id=station_url_id,
                zone=zone
            )

            naive_datetime = datetime.strptime(
                results[station_url_id].pop('timestamp'),
                '%d/%m/%Y %H:%M'
            )

            timestamp = local.localize(naive_datetime)

            for abbr in results[station_url_id].keys():

                # don't process this value if there is none
                if results[station_url_id][abbr] is None:
                    continue

                parameter, success = Parameter \
                    .objects.get_or_create(abbr=abbr)

                # don't create records if there are already there
                record, success = Record.objects.get_or_create(
                    station=station,
                    parameter=parameter,
                    value=results[station_url_id][abbr],
                    timestamp=timestamp
                )

                record.save()
