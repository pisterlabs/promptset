# coding=utf-8
from optparse import make_option
import time
import csvkit
from django.core.management.base import LabelCommand, CommandError
from openaid.codelists import models as codelist_models
from openaid.projects import models
from openaid.projects import forms
from openaid.projects import mapping


class Command(LabelCommand):
    args = '<crs_file crs_file ...>'
    help = 'Speficica i CRS file da lavorare'

    option_list = LabelCommand.option_list + (
        make_option('-c', '--clean',
            action='store_true', dest='clean', default=False,
            help="Clean old activities and projects before import crs file."),
    )

    def delete_projects(self):
        answer = raw_input('Are you sure? (Yes/No)')
        if answer.lower() in ('yes', 'y'):
            self.stdout.write('Deleting %s activities' % models.Activity.objects.count())
            models.Activity.objects.all().delete()
            self.stdout.write('Deleting %s projects' % models.Project.objects.count())
            models.Project.objects.all().delete()
            return True
        return False

    def handle_label(self, crs_filename, **options):
        """
        Gli argomenti forniti sono i nomi dei file CRS da lavorare
        """
        start_time = time.time()
        i = 0

        if options.get('clean') and not self.delete_projects():
            raise CommandError("Import aborted")

        self.all_codelists = dict([
            (cl.code_list, dict(cl.objects.values_list('code', 'pk')))
            for cl in codelist_models.CODE_LISTS
        ])

        rows = projects = activities = 0

        try:
            with open(crs_filename, 'r') as crs_file:
                for rows, activity in enumerate(csvkit.DictReader(crs_file), start=1):
                    activity, new_project = self.load_activity(activity, rows)
                    if activity:
                        activities += 1
                        self.stdout.write("\rImported row: %d" % (activities), ending='')
                        self.stdout.flush()
                        if new_project:
                            projects += 1
        except KeyboardInterrupt:
            self.stdout.write("\nCommand execution aborted.")
        finally:
            self.stdout.write("\nTotal projects: %d" % projects)
            self.stdout.write("Total activities: %d" % activities)
            self.stdout.write("Total rows: %d" % rows)
            self.stdout.write("Execution time: %d seconds" % (time.time() - start_time))

    def load_activity(self, row, i):
        # fix report_type
        if not row['report_type']:
            row['report_type'] = 0
        # fix empty flow
        if row['flow_type'] == '':
            row['flow_type'] = 0
        elif row['flow_type'] == '10':
            row['flow_type'] = '100'
        elif row['flow_type'] == '20':
            return 0, 0
        # fix agency
        if row['agency_code'] in ['10', '11']:
            row['agency_code'] = '7'


        for field, field_value in row.items():
            if field.endswith('_date'):
                row[field] = '-'.join(reversed(field_value.replace('/', '-').split('-')))
            elif field in ['commitment', 'disbursement']:
                row[field] = field_value.strip().replace('.', '').replace(',', '.')
                if row[field] == '-':
                    row[field] = 0.0
                row[field] = float(row[field]) / 1000

        # 1. creo l'Activity
        activity_form = mapping.create_mapped_form(forms.ActivityForm, row, mapping.OrderedDict([
            ('crsid', 'crsid'),
            ('year', 'year'),
            ('title', 'title'),
            ('number', 'number'),
            ('description', 'description'),
            ('report_type', 'report_type'),
            ('is_ftc', 'is_ftc'),
            ('is_pba', 'is_pba'),
            ('is_investment', 'is_investment'),
            ('geography', 'geography'),
            ('bi_multi', 'bi_multi'),
            ('flow_type', 'flow_type'),
            ('expected_start_date', 'expected_start_date'),
            ('completion_date', 'completion_date'),
            ('commitment_date', 'commitment_date'),
            ('commitment', 'commitment'),
            ('disbursement', 'disbursement'),
        ]))
        if not activity_form.is_valid():
            self.stderr.write('\nError on row %s:\n%s' % (i, activity_form.errors.as_text()))
            return 0, 0
        activity = activity_form.save()

         # 2. creo i markers
        markers_form = mapping.create_mapped_form(forms.MarkersForm, row, {
            'biodiversity_mark': 'biodiversity',
            'climate_adaptation_mark': 'climate_adaptation',
            'climate_mitigation_mark': 'climate_mitigation',
            'desertification_mark': 'desertification',
            'environment_mark': 'environment',
            'gender_mark': 'gender',
            'pd_gg_mark': 'pd_gg',
            'trade_mark': 'trade',
        })
        if markers_form.is_valid():
            activity.markers = markers_form.save()

        # 3. associo il channel reported
        channel_reported_form = mapping.create_mapped_form(forms.ChannelReportedForm, row, {
            'channel_reported': 'name'
        })
        if channel_reported_form.is_valid():
            crn, _ = models.ChannelReported.objects.get_or_create(name=channel_reported_form.instance.name)
            activity.channel_reported = crn

        # 4. aggiungo le code lists
        for codelist in codelist_models.CODE_LISTS:
            if codelist.code_list == 'donor':
                continue
            code_field = '%s_code' % codelist.code_list
            code_value = row[code_field]
            if code_value:
                try:
                    pk = self.all_codelists[codelist.code_list][code_value]
                    setattr(activity, '%s_id' % codelist.code_list, pk)
                except KeyError:
                    self.stderr.write('\nError: cannot find %s with code "%s" (row: %s)' % (codelist.code_list, code_value, i))

        # 5. la associo ad un project
        activity.project, project_created = models.Project.objects.get_or_create(**{
            'crsid': activity.crsid,
            'recipient_id': activity.recipient_id,
            'defaults': {
                'start_year': activity.year,
                'end_year': activity.year,
            }
        })

        activity.save()
        activity.project.update_from_activities(save=True)

        return activity, project_created
