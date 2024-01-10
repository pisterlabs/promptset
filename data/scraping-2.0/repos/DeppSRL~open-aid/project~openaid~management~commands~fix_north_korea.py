# coding=utf-8
from collections import OrderedDict
from django.core.exceptions import ObjectDoesNotExist

__author__ = 'stefano'
import logging
from django.core.management.base import BaseCommand
from openaid.codelists.models import Recipient


class Command(BaseCommand):
    help = 'Fixes North Korea recipient obj'
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

        nk = None
        self.logger.info(u"Start procedure to fix North Korea")
        try:
            nk = Recipient.objects.get(code='740')
        except ObjectDoesNotExist:
            self.logger.critical("Object with code 740 not present in db. quit")
            exit()
        nk.iso_code = 'PRK'
        nk.iso_alpha2 = 'KP'
        nk.popolazione = 24895705
        nk.crescita_popolazione = 0.5
        nk.pil = None
        nk.pil_procapite= None
        nk.save()

        self.logger.info(u"Finished")