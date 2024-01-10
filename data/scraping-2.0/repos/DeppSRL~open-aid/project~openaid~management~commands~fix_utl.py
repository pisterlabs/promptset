# coding=utf-8
import logging

from django.core.management.base import NoArgsCommand
from openaid.codelists.models import Recipient

from openaid.projects.models import Utl



class Command(NoArgsCommand):

    help = 'Fix utl with id 28,29,30 recipient list'
    logger = logging.getLogger('openaid')

    def handle_noargs(self, **options):

        # set UTL with 28,29,30 all the recipients as recipient sets

        all_recipients = Recipient.objects.all()
        self.logger.info("Start...")
        utl_28 = Utl.objects.get(pk=28)
        utl_28.recipient_set = all_recipients
        utl_28.save()
        self.logger.info("Updated utl with pk 28")
        utl_29 = Utl.objects.get(pk=29)
        utl_29.recipient_set = all_recipients
        utl_29.save()
        self.logger.info("Updated utl with pk 29")
        utl_30 = Utl.objects.get(pk=30)
        utl_30.recipient_set = all_recipients
        utl_30.save()
        self.logger.info("Updated utl with pk 30")
        self.logger.info("Done")