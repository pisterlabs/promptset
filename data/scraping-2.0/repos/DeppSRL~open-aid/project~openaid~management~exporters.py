# coding=utf-8
import StringIO
import csv
import time
import zipfile
import datetime
from os import path, rename
import sys
import csvkit
from django.conf import settings
from openaid.codelists import models as codelists
from openaid.projects import mapping
from openaid.projects.models import Activity
from openaid.contexts import START_YEAR, END_YEAR


MEDIA_EXPORT_PATH = path.join(settings.MEDIA_ROOT, 'crs')

CODELISTS_CSV_MAP = dict([(getattr(cl, 'code_list_csv_field', '%scode' % cl.code_list), cl) for cl in codelists.CODE_LISTS])

EXPORTED_FIELDS = (
    ['openaid_id', ] +
    mapping.ACTIVITY_FIELDS_MAP.keys() +
    mapping.CHANNEL_REPORTED_MAP.keys() +
    mapping.MARKERS_FIELDS_MAP.keys() 
    # CODELISTS_CSV_MAP.keys() +
    # ['eur_commitment', 'eur_disbursement']
)

CODELISTS_FIELDS = []
for cl in CODELISTS_CSV_MAP.keys():
    CODELISTS_FIELDS.append(cl)
    if cl.endswith('_t'):
        CODELISTS_FIELDS.append(cl + 'name')
    else:
        CODELISTS_FIELDS.append(cl.replace('code', 'name'))
EXPORTED_FIELDS += CODELISTS_FIELDS
EXPORTED_FIELDS += ['eur_commitment', 'eur_disbursement']


def serialize_activity(activity):
    # act = {'openaid_id': activity.pk}
    # act.update(dict.fromkeys(EXPORTED_FIELDS, ''))
    act = dict.fromkeys(EXPORTED_FIELDS, '')
    act['openaid_id'] = activity.pk
    for field in EXPORTED_FIELDS:

        if field in mapping.ACTIVITY_FIELDS_MAP:
            act[field] = getattr(activity, mapping.ACTIVITY_FIELDS_MAP[field], '')
            if isinstance(act[field], bool):
                act[field] = '1' if act[field] else '0'
            elif isinstance(act[field], datetime.datetime):
                act[field] = act[field].date().isoformat()
            # elif isinstance(act[field], float):
            #     act[field] = repr(act[field]).replace('.', ',')

        elif field in mapping.MARKERS_FIELDS_MAP:
            act[field] = getattr(activity.markers, mapping.MARKERS_FIELDS_MAP[field], '') or ''

        elif field in CODELISTS_FIELDS:

            if field.endswith('name'):
                if field.endswith('_tname'):
                    codelist = CODELISTS_CSV_MAP[field.replace('name', '')]
                else:
                    codelist = CODELISTS_CSV_MAP[field.replace('name', 'code')]
                codelist_item = getattr(activity, codelist.code_list, None) or ''
                act[field] = codelist_item.name_en if codelist_item else ''
            else:
                codelist = CODELISTS_CSV_MAP[field]
                codelist_item = getattr(activity, codelist.code_list, None) or ''
                act[field] = codelist_item.code if codelist_item else ''

        elif field in mapping.CHANNEL_REPORTED_MAP:
            act[field] = activity.channel_reported.name if activity.channel_reported else ''

    for euros in ['commitment', 'disbursement']:
        act['eur_%s' % euros] = getattr(activity, euros, '')

    return act

def generate_file_path(year, to_backup=False):
    csv_path = path.join(MEDIA_EXPORT_PATH, 'CRS_%s_%s' % (settings.OPENAID_CRS_DONOR, year))
    if not to_backup:
        return '%s.csv.zip' % csv_path
    return '%s_%s.csv.zip' % (csv_path, time.strftime('%Y%m%d-%H%M%S'))

def export_activities_by_year(year):

    print 'Start export for year: %s' % year

    zip_file_path = generate_file_path(year)
    if path.isfile(zip_file_path):
        # backup
        backup_file_path = generate_file_path(year, to_backup=True)
        rename(zip_file_path, backup_file_path)
        print 'Backup old export: %s' % backup_file_path

    output = StringIO.StringIO() ## temp output file
    csv_writer = csvkit.DictWriter(output, EXPORTED_FIELDS, quoting=csv.QUOTE_ALL)
    csv_writer.writeheader()
    i = 0
    for i, activity in enumerate(Activity.objects.filter(year=year) if year != 'all' else Activity.objects.all(), start=1):
        csv_writer.writerow(serialize_activity(activity))
        print "\r%d" % i,
        sys.stdout.flush()

    csv_file = zipfile.ZipFile(zip_file_path, 'w')
    csv_file.writestr(path.basename(zip_file_path)[:-4], output.getvalue())

    print '%d Activities for year %s exported in: %s' % (i, year, zip_file_path)


def export_activities(year=None):
    """
    Questo task si occupa di generare i file csv contenenti le Activities.
    Se sono già state generate, si preoccupa di fare un backup e di rigenerare nuovamente i dati.
    """

    print 'Exporting Activities for %s Donor from %d to %d' % (
        settings.OPENAID_CRS_DONOR, START_YEAR, END_YEAR
    )
    if not year:
        for year in range(START_YEAR, END_YEAR+1):
            export_activities_by_year(year)
    else:
        export_activities_by_year(year)
        # p = Process(export_activities_by_year, args=(year, ))
        # jobs.append(p)
        # p.start()
