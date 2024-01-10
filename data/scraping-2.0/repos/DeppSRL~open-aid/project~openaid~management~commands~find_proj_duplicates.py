# coding=utf-8
__author__ = 'stefano'
import logging
from django.core.management.base import BaseCommand
from openaid.projects.models import Initiative, Project


class Command(BaseCommand):
    help = 'Prints out project belonging to the same initiative that have different not-null values for certain fields.'
    logger = logging.getLogger('openaid')
    # fields to check arranged in a way to be used as a filter
    filters = [
        {'description_it__isnull':False},
        {'description_en__isnull':False},
        {'recipient__isnull':False},
        {'outcome_it__isnull':False},
        {'outcome_en__isnull':False},
        {'beneficiaries_it__isnull':False},
        {'beneficiaries_en__isnull':False},
        {'beneficiaries_female__isnull':False},
        {'status__isnull':False},
        {'is_suspended__isnull':False},
        {'other_financiers_it__isnull':False},
        {'other_financiers_en__isnull':False},
        {'loan_amount_approved__isnull':False},
        {'grant_amount_approved__isnull':False},
        {'counterpart_authority_it__isnull':False},
        {'counterpart_authority_en__isnull':False},
        {'email__isnull':False},
        {'location_en__isnull':False},
        {'location_it__isnull':False},
        {'sector__isnull':False}
    ]

    text_fields = [
        'description_it',
        'description_en',
        'outcome_it',
        'outcome_en',
        'beneficiaries_it',
        'beneficiaries_en',
        'other_financiers_en',
        'other_financiers_it',
        'counterpart_authority_it',
        'counterpart_authority_en',
        'email',
        'location_it',
        'location_en',
    ]

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
        import datetime
        thresold_date = datetime.datetime.strptime('13042015', '%d%m%Y').date()
        for filt in self.filters:
            key = filt.keys()[0]
            fieldname = key.replace("__isnull","")

            for init in Initiative.objects.all().order_by('code'):
                project_set = Project.objects.filter(initiative=init, last_update__gte=thresold_date).filter(**filt).order_by('pk')
                # if the field is a text/char field, excludes the "" values
                if fieldname in self.text_fields:
                    project_set = project_set.exclude(**{fieldname:''})
                    
                count = project_set.count()
                if count > 1:
                    # get the different values
                    field_values = project_set.values_list(fieldname,flat=True)
                    first_value = None
                    different_flag = False
                    for idx,fv in enumerate(field_values):
                        if idx == 0:
                            first_value = fv
                            continue
                        if fv != first_value:
                            different_flag = True
                            break

                    if different_flag is True:
                        proj_pks = ",".join([str(x) for x in project_set.values_list('pk',flat=True)])
                        self.logger.error(u"Field:{}, count:{}, Initiative.code:{}, Projects.pk:{}".format(fieldname, count, init.code, proj_pks))

        self.logger.info(u"Finished")