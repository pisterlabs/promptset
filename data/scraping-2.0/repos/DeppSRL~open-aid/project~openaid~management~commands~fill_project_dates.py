__author__ = 'stefano'
# coding=utf-8
import logging
from django.core.management.base import BaseCommand
from openaid.projects.models import Activity, Project


class Command(BaseCommand):

    help = 'Fills project.expected_start_date and project.completion_date based on activity set best date'
    logger = logging.getLogger('openaid')

    def get_best_dates(self, project, activities):

        for activity in activities:
            if activity.expected_start_date is None and activity.completion_date is None:
                continue

            else:
                self.logger.debug("dates found -year:{}, start:{}, end:{}".format(activity.year, activity.expected_start_date,activity.completion_date ))
                project.expected_start_date = activity.expected_start_date
                project.completion_date = activity.completion_date
                break

        return project


    def handle(self, *args, **options):
        verbosity = options['verbosity']
        if verbosity == '0':
            self.logger.setLevel(logging.ERROR)
        elif verbosity == '1':
            self.logger.setLevel(logging.WARNING)
        elif verbosity == '2':
            self.logger.setLevel(logging.INFO)
        elif verbosity == '3':
            self.logger.setLevel(logging.DEBUG)


        # loop over projects and set the best dates found in its activities for start/end
        all_projects = Project.objects.all().order_by('crsid')
        for project in all_projects:
            self.logger.debug("Project:'{}'".format(project))
            project.expected_start_date = None
            project.completion_date = None

            activity_set = Activity.objects.filter(project=project).order_by('-year')
            if activity_set.count() > 0:
                project = self.get_best_dates(project, activity_set)

            project.save()

        self.logger.info("Finished updating {} projects".format(all_projects.count()))
