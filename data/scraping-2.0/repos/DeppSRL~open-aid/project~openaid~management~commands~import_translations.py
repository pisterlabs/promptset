# coding=utf-8
import time
import csvkit
from optparse import make_option
from django.core.management.base import LabelCommand, CommandError
from django.conf import settings
from openaid.projects.translation import ActivityTranslationOptions
from openaid.projects.forms import text_cleaner
from openaid.projects import models


class Command(LabelCommand):
    args = 'csvfile'
    help = 'Importa la traduzione del delle Activity. ' \
           'Per ogni language code e per ogni field traducibile per l\'Activity'

    option_list = LabelCommand.option_list + (
        make_option('-f', '--field',
            action='store', dest='field', default=ActivityTranslationOptions.fields[0],
            help="Field to translate"),
        make_option('-l', '--lang',
            action='store', dest='lang',
            help="Import only this language."),
        make_option('-o', '--override',
            action='store_true', dest='override',
            help="Override old values."),
    )

    def handle_label(self, crs_file, **options):

        start_time = time.time()
        i = 0
        translations = 0
        field = options['field']

        languages = [lang[0].split('-')[0] for lang in settings.LANGUAGES]
        if options['lang']:
            if options['lang'] not in languages:
                raise CommandError("Invalid language code '%s'. Try: %s" % (options['lang'], ', '.join(languages)))
            languages = [options['lang'], ]

        self.stdout.write('FIELD: %s' % field)
        self.stdout.write('LANGUAGES: %s' % languages)

        with open(crs_file, 'r') as crs_file:

            rows = csvkit.DictReader(crs_file, encoding='utf-8')

            for i, row in enumerate(rows, start=1):
                updates, matches = self.translate(row, field, languages, override=options['override'])
                if matches == 0:
                    self.stdout.write("\rRow %d non corrisponde a nessuna Activity" % (i))
                else:
                    self.stdout.write("\r%s: Translated activities %d    " % (i, updates), ending='')
                    self.stdout.flush()
                translations += updates


        self.stdout.write("\nTotal rows: %d" % i)
        self.stdout.write("Execution time: %d seconds" % (time.time() - start_time))

    def translate(self, row, field, languages, override=False):

        # translations ={}
        updates = 0
        matches = 0


        field_value = text_cleaner(row[field])
        if field_value:

            for activity in models.Activity.objects.filter(**{
                '%s__iexact' % field: field_value
            }):
                i = 0
                matches += 1
                for lang in languages:

                    translated_field = '%s_%s' % (field, lang)
                    if translated_field not in row.keys():
                        continue

                    translated_field_value = text_cleaner(row[translated_field])

                    if (translated_field_value and override) or not getattr(activity, translated_field):
                        if getattr(activity, translated_field) != translated_field_value:
                            setattr(activity, translated_field, translated_field_value)
                            i +=1
                if i > 0:
                    activity.save()
                    updates += i

        return updates, matches