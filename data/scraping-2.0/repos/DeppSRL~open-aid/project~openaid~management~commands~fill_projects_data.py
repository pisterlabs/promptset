# coding=utf-8
from optparse import make_option
import time
import csvkit
from django.core.management.base import NoArgsCommand, CommandError
from openaid.codelists import models as codelist_models
from openaid.projects import models



class Command(NoArgsCommand):

    help = 'Collect data from project activity'

    def handle_noargs(self, **options):

        projects = 0

        for project in models.Project.objects.all():

            projects += 1

            for activity in project.activity_set.all().order_by('year'):

                if activity.title:
                    project.title = activity.title

                if activity.long_description:
                    project.description = activity.long_description

                for codelist in ('agency', 'aid_type', 'channel', 'finance_type', 'sector'):

                    self.fill_project_codelist(project, activity, codelist)

                self.merge_markers(project, activity.markers)

            project.markers.save()
            project.save()

            self.stdout.write("\rFilled project: %d" % projects, ending='')
            self.stdout.flush()

    def merge_markers(self, project, markers):

        if project.markers is None:
            project.markers = models.Markers.objects.create()

        for field in project.markers.names:
            mark = getattr(markers, field, None)
            if mark is not None and getattr(project.markers, field) != mark:
                setattr(project.markers, field, mark)

    def fill_project_codelist(self, project, activity, codelist):

        codelist_value = getattr(activity, codelist, False)

        if not codelist_value or codelist_value is getattr(project, codelist):
            return

        setattr(project, codelist, codelist_value)

