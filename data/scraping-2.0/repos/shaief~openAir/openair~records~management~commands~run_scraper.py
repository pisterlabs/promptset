from datetime import datetime
from django.core.management.base import BaseCommand  # , CommandError
import pytz
from openair.settings.base import TIME_ZONE
from openair.records.models import Zone, Station, Parameter, Record
from openair.records.scraper import scrape_zone


class Command(BaseCommand):
    help = 'Run the scraper on the zones to fill the DB'

    def handle(self, *args, **options):

        local = pytz.timezone(TIME_ZONE)

        # run over all of the zones
        for zone in Zone.objects.all():

            results = scrape_zone(zone.url_id)

            # run over the received stations
            for station_url_id in results.keys():

                station, success = Station.objects.get_or_create(
                    url_id=station_url_id,
                    zone=zone
                )

                timestamp_str = results[station_url_id] \
                    .pop('timestamp')

                # the hour 24:00 is not allowed but yet exists
                timestamp_str = timestamp_str.replace(
                    ' 24:', ' 00:'
                )

                naive_timestamp = datetime.strptime(
                    timestamp_str, '%d/%m/%Y %H:%M'
                )

                timestamp = local.localize(naive_timestamp)

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
