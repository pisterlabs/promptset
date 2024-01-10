__author__ = 'stefano'
# coding=utf-8
from datetime import datetime, date
import logging
from django.core.management.base import BaseCommand
from openaid.projects.models import Activity


class Command(BaseCommand):

    help = 'Fixes activity dates removing Hours and minutes'
    logger = logging.getLogger('openaid')


    def check_invalid_date(self, date_obj):
        if date_obj is None:
            return date_obj

        treshold_date = date(1970, 1, 1)
        if date_obj <= treshold_date:
            return None
        else:
            return date_obj


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


        # loop over activity, fix hour and minute setting hour to 00:00
        # then check if the date is valid: if date < 1.1.1970 set date to null

        all_activities = Activity.objects.all().order_by('year')
        for act in all_activities:
            # check valid dates
            # expected_start_date
            # completion_date
            # commitment_date
            act.expected_start_date = self.check_invalid_date(act.expected_start_date)
            act.completion_date = self.check_invalid_date(act.completion_date)
            act.commitment_date = self.check_invalid_date(act.commitment_date)
            # act.save()
        self.logger.info("Finished ")
