# coding=utf-8
import time
import csvkit
from optparse import make_option
from django.core.management.base import LabelCommand, CommandError
from django.conf import settings
from django.db.models.query_utils import Q
from openaid.projects.translation import ActivityTranslationOptions
from openaid.projects import models


def prepare_fields(fields):

    def order(v):
        try:
            return {
                'code': 0,
                'parent': 1,
                'acronym': 2,
                'group': 2,
                'title': 3,
                'name': 3,
                'description': 5,
            }[v]
        except KeyError:
            if v.startswith('name_') or v.startswith('title_'):
                return 4
            if v.startswith('description_'):
                return 6
            return v

    return sorted(fields[:], key=order)

class Command(LabelCommand):
    args = 'csvfile'
    help = 'Esporta la traduzione del delle Activity. ' \
           'Per ogni language code e per ogni field traducibile per l\'Activity'

    option_list = LabelCommand.option_list + (
        make_option('-f', '--field',
            action='store', dest='field', default=ActivityTranslationOptions.fields[0],
            help="Field to translate"),
        make_option('-l', '--lang',
            action='store', dest='lang',
            help="Import only this language."),
    )

    def handle_label(self, crs_file, **options):

        start_time = time.time()
        i = 0
        field = options['field']

        languages = [lang[0].split('-')[0] for lang in settings.LANGUAGES]
        if options['lang']:
            if options['lang'] not in languages:
                raise CommandError("Invalid language code '%s'. Try: %s" % (options['lang'], ', '.join(languages)))
            languages = [options['lang'], ]

        all_fields = [field, ]
        for lang in languages:
            all_fields.append('%s_%s' % (field, lang))
        all_fields = prepare_fields(all_fields)

        self.stdout.write('FIELD: %s' % field)
        self.stdout.write('LANGUAGES: %s' % languages)
        self.stdout.write('ALL FIELDS: %s' % all_fields)

        with open(crs_file, 'w') as crs_file:

            writer = csvkit.DictWriter(crs_file, all_fields)
            writer.writeheader()
            for i, activity in enumerate(models.Activity.objects.exclude(**{field: ''}).order_by(*all_fields).values(*all_fields).distinct(*all_fields)):
                writer.writerow(activity)
                self.stdout.write("\rExported activities %d" % (i,), ending='')
                self.stdout.flush()


        self.stdout.write("\nTotal rows: %d" % i)
        self.stdout.write("Execution time: %d seconds" % (time.time() - start_time))
