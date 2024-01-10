# coding=utf-8
from collections import OrderedDict

__author__ = 'stefano'
import logging
from django.core.management.base import BaseCommand
from openaid.projects.models import Project, Activity


class Command(BaseCommand):
    help = 'Prints out CRS ID which are not uniques in the DB'
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
        # crs id not univoques for all proj
        repetitions = OrderedDict()
        repetitions_2014 = OrderedDict()
        all_crsid = Project.objects.all().order_by('crsid').distinct('crsid').values_list('crsid',flat=True)
        self.logger.info("*********** all projects **************")
        for crsid in all_crsid:
            proj_set = Project.objects.filter(crsid=crsid)
            if len(proj_set)>1:
                repetitions[crsid] = None
                repetitions[crsid] = proj_set

                self.logger.debug('Code "{}" is repeated {} times'.format(crsid, len(proj_set)))
        self.logger.info("Repeated CRS for all DB:{} codes:{}".format(len(repetitions.keys()),",".join(repetitions.keys())))

        # crs id not univoques for 2014
        crsid_projects_2014 = Activity.objects.filter(year=2014).order_by('project__crsid').values_list('project__crsid', flat=True).distinct()
        self.logger.info("*********** 2014 **************")
        for crsid in crsid_projects_2014:
            proj_set = Project.objects.filter(crsid=crsid)
            if len(proj_set)>1:
                repetitions_2014[crsid] = None
                repetitions_2014[crsid] = proj_set
                self.logger.debug('Code "{}" is repeated {} times'.format(crsid, len(proj_set)))

        self.logger.info("Repeated CRS for 2014: {} codes:{}".format(len(repetitions_2014.keys()),",".join(repetitions_2014.keys())))

        self.logger.info(u"Finished")