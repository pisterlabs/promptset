# coding=utf-8
from optparse import make_option
from os import path
import time
import csvkit
from django.core.management.base import LabelCommand, CommandError
from django.conf import settings
from openaid.codelists import models
from openaid.codelists.translation import CodeListTranslationOptions
from openaid.projects.forms import text_cleaner


class Command(LabelCommand):

    ACTIONS = ['import', 'reload', 'clear', 'translate', 'stats', 'update']

    args = '|'.join(ACTIONS)
    help = 'Execute an action on code lists.'

    option_list = LabelCommand.option_list + (
        make_option('-f', '--field',
            action='store', dest='field',
            help="Comma separated of fields to check"),
        make_option('-l', '--lang',
            action='store', dest='lang',
            help="Import only this language."),
        make_option('-o', '--override',
            action='store_true', dest='override', default=False,
            help="Override old values."),
        make_option('-c', '--codelist',
            action='store', dest='codelist',
            help="Use only this codelist."),
    )

    def handle_label(self, action, **options):

        assert action in self.ACTIONS

        if action == 'stats':
            self.handle_stats(**options)
            return

        elif action == 'clear':
            self.delete_code_lists()

        elif action == 'translate':
            self.translate_codelists(**options)

        elif action in ('reload', 'import'):
            counters = self.get_code_list_counters()
            already_imported = any([c for cl, c in counters])
            if action == 'import' and already_imported:
                self.stderr.write('\nError: Code lists already exists. Try to use `manage.py codelists reload`.')
                return

            if already_imported:
                self.delete_code_lists()

            for codelist in models.CODE_LISTS:
                self.import_codelist(codelist)

        elif action == 'update':
            for codelist in models.CODE_LISTS:
                if options['codelist'] == codelist.code_list:
                    self.update_codelist(codelist)

    def handle_stats(self, **options):
        for codelist, count in self.get_code_list_counters():
            self.stdout.write('%s: %d' % (codelist.code_list, count))

    def translate_codelists(self, **options):

        if options['codelist']:
            codelists = [models.CODE_LISTS_DICT[c] for c in options['codelist'].split(',') if c in models.CODE_LIST_NAMES]
        else:
            codelists = models.CODE_LISTS

        if options['field']:
            fields = [f.strip() for f in options['field'].split(',')]
        else:
            fields = list(CodeListTranslationOptions.fields)

        languages = [lang[0].split('-')[0] for lang in settings.LANGUAGES]
        if options['lang']:
            if options['lang'] not in languages:
                raise CommandError("Invalid language code '%s'. Try: %s" % (options['lang'], ', '.join(languages)))
            languages = [options['lang'], ]

        self.stdout.write('CODELISTS: %s' % codelists)
        self.stdout.write('FIELDS: %s' % fields)
        self.stdout.write('LANGUAGES: %s' % languages)

        for codelist in codelists:

            translations = 0
            self.stdout.write('Import translations for: %s' % codelist)

            with open(path.join(settings.RESOURCES_PATH, 'codelists', '%s.csv' % codelist.code_list), 'r') as crs_file:

                rows = csvkit.DictReader(crs_file, encoding='utf-8')

                for i, row in enumerate(rows, start=1):

                    if codelist.code_list == 'agency' and int(row.get('donor', '0')) != settings.OPENAID_CRS_DONOR:
                        continue

                    updates, matches = self.translate(codelist, row, fields, languages, override=options['override'])
                    if matches == 0:
                        self.stdout.write("\rRow %d non corrisponde a nessuna %s" % (i, codelist))
                    else:
                        self.stdout.write("\r%s: Translated codelists %d    " % (i, updates), ending='')
                        self.stdout.flush()
                    translations += updates

            self.stdout.write('\nImported %s translations' % translations)

    def translate(self, codelist, row, fields, languages, override=False):

        updates = 0
        matches = 0
        try:
            obj = codelist.objects.get(code=row['code'])
        except codelist.DoesNotExist:
            self.stderr.write('Code "%s" not found in %s' % (row['code'], codelist))
            return 0, 0
        for field in filter(lambda f: f in row.keys(), fields):

            i = 0
            matches += 1
            for lang in languages:

                translated_field = '%s_%s' % (field, lang)
                if translated_field not in row.keys():
                    continue

                translated_field_value = text_cleaner(row[translated_field])

                if (translated_field_value and override) or not getattr(obj, translated_field):
                    if getattr(obj, translated_field) != translated_field_value:
                        setattr(obj, translated_field, translated_field_value)
                        i +=1
            if i > 0:
                obj.save()
                updates += i
            elif row[field] and not getattr(obj, field):
                setattr(obj, field, row[field])
                obj.save()

        return updates, matches

    ## utils
    def import_codelist(self, codelist):

        i = 0
        start_time = time.time()
        self.stdout.write('\n### IMPORT CODELIST: %s' % codelist.code_list)
        csv_path = path.join(settings.RESOURCES_PATH, 'codelists', '%s.csv' % codelist.code_list)
        reader = csvkit.DictReader(open(csv_path))
        for row in reader:
            # agency has donor
            if codelist.code_list == 'agency':
                # skip other agencies (code is unique)
                if int(row['donor']) != settings.OPENAID_CRS_DONOR:
                    continue
                row['donor'] = models.Donor.objects.get(code=row['donor'])
            if row.has_key('parent'):
                try:
                    row['parent'] = codelist.objects.get(code=row['parent'])
                except codelist.DoesNotExist:
                    row['parent'] = None
            codelist.objects.create(**row)
            i += 1

        self.stdout.write("Total rows: %d" % i)
        self.stdout.write("Execution time: %d seconds" % (time.time() - start_time))
        self.stdout.write('###\n')


    def delete_code_lists(self):
        answer = raw_input('Are you sure? (Yes/No)')
        if answer.lower() in ('yes', 'y'):
            for CodeList in self.get_code_lists():
                self.stdout.write('\nDeleting %s' % CodeList.code_list)
                self.stdout.write(': %d' % CodeList.objects.count())
                CodeList.objects.all().delete()

    def update_codelist(self, codelist):

        i = c = 0
        start_time = time.time()
        self.stdout.write('\n### UPDATE CODELIST: %s' % codelist.code_list)
        csv_path = path.join(settings.RESOURCES_PATH, 'codelists', '%s.csv' % codelist.code_list)
        reader = csvkit.DictReader(open(csv_path))
        for row in reader:
            # agency has donor
            if codelist.code_list == 'agency':
                # skip other agencies (code is unique)
                if int(row['donor']) != settings.OPENAID_CRS_DONOR:
                    continue
                row['donor'] = models.Donor.objects.get(code=row['donor'])
            if row.has_key('parent'):
                try:
                    row['parent'] = codelist.objects.get(code=row['parent'])
                except codelist.DoesNotExist:
                    row['parent'] = None
            _, created = codelist.objects.get_or_create(
                code=row['code'],
                defaults=row
            )
            i += 1
            if created:
                c += 1

        self.stdout.write("Total rows: %d" % i)
        self.stdout.write("Created items: %d" % c)
        self.stdout.write("Execution time: %d seconds" % (time.time() - start_time))
        self.stdout.write('###\n')

    def get_code_list_counters(self):
        return [(cl, cl.objects.count()) for cl in models.CODE_LISTS]

    def get_code_lists(self):
        return models.CODE_LISTS

