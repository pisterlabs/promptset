# coding=utf-8
from django.core.exceptions import ObjectDoesNotExist

__author__ = 'stefano'
import logging
from django.core.management.base import BaseCommand
from openaid.codelists.models import Channel


class Command(BaseCommand):
    help = 'Adds Channels 22501 and 22502 for import CRS 2014'
    logger = logging.getLogger('openaid')

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


        self.logger.info(u"Start procedure")
        try:
            parent = Channel.objects.get(code='22000')
        except ObjectDoesNotExist:
            self.logger.critical("Cant find parent with code 22000, quit")
            exit()

        self.logger.info("Adding 22501")
        c = Channel()
        c.name = 'OXFAM - provider country office'
        c.name_en = 'OXFAM - provider country office'
        c.code ='22501'
        c.parent = parent
        c.acronym = 'OXFAM'
        c.save()
        self.logger.info("Adding 22502")
        c = Channel()
        c.name = 'Save the Children - donor country office'
        c.name_en = 'Save the Children - donor country office'
        c.code ='22502'
        c.parent = parent
        c.save()
        self.logger.info(u"Finished")