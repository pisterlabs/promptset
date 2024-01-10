# coding=utf-8

__author__ = 'stefano'
import logging
from django.core.exceptions import ObjectDoesNotExist
from openaid.codelists.models import Recipient
from django.core.management.base import BaseCommand
from openaid.projects.models import Initiative, Project, Activity

# This one-time procedure get data from some projects and transfer it to selected Initiative.

class Command(BaseCommand):
    help = 'Fills Initiative few empty initiatives with selected projs data'
    logger = logging.getLogger('openaid')


    # maps the field name between project (keys) and initiative (value)
    field_map = {
        'description_it': 'description_temp_it',
        'description_en': 'description_temp_en',
        'recipient': 'recipient_temp',
        'outcome_it': 'outcome_temp_it',
        'outcome_en': 'outcome_temp_en',
        'beneficiaries_it': 'beneficiaries_temp_it',
        'beneficiaries_en': 'beneficiaries_temp_en',
        'beneficiaries_female': 'beneficiaries_female_temp',
        'status': 'status_temp',
        'is_suspended': 'is_suspended_temp',
        'other_financiers_it': 'other_financiers_temp_it',
        'other_financiers_en': 'other_financiers_temp_en',
        'counterpart_authority_it': 'counterpart_authority_temp_it',
        'counterpart_authority_en': 'counterpart_authority_temp_en',
        'email': 'email_temp',
        'location_it': 'location_temp_it',
        'location_en': 'location_temp_en',
        'sector': 'purpose_temp',
    }

    status_order = ['100', '75', '50', '25', '0', '-']

    def get_most_advanced_status(self, project_set):
        # gets more advanced status in the project set
        for status in self.status_order:
            if project_set.filter(status=status).count() > 0:
                return status
        return '-'

    def update_fields(self, initiative, project):

        # loops on every field that has to be updated and updates if the conditions apply
        for project_fieldname, initiative_fieldname in self.field_map.iteritems():

            if project_fieldname == 'sector':
                field_value = getattr(project, project_fieldname)
                if field_value is not None:
                    # only consider sector value that are LEAF nodes, no children
                    if field_value.get_children().count() != 0:
                        self.logger.error(
                            "Initiative:{}. Cannot copy SECTOR VALUE: {} from Project, this Sector is not a leaf node! SKIP".format(
                                initiative, field_value))
                        continue

            elif project_fieldname == 'status':
                field_value = getattr(project, project_fieldname)
                # STATUS: if the proj.status is == 100 => Almost completed
                # translates the value to 90 for Almost completed in Initiative
                # because in Initiative there is a status for "COMPLETED' which has value=100
                if field_value == '100':
                    field_value = '90'
            else:
                field_value = getattr(project, project_fieldname)

            if field_value is not None:
                initiative.__setattr__(initiative_fieldname, field_value)

        return initiative

    def update_related_objects(self, initiative, project):

        # updates documents and photo set getting the photos and docs from the projects
        for doc in project.document_set.all():
            initiative.document_set.add(doc)

        for photo in project.photo_set.all():
            initiative.photo_set.add(photo)


        # updates reports and problems with initiative link (project link will later be removed by migrations)
        for r in project.report_set.all():
            r.initiative = initiative
            r.save()

        for prob in project.problem_set.all():
            prob.initiative = initiative
            prob.save()

        return initiative

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

        source_projects = {
            27009: {'recipient': "Tunisia", 'initiative_pk': 8348},
            19918: {'recipient': "Tunisia", 'initiative_pk': 7713},
            19913: {'recipient': "Tunisia", 'initiative_pk': 7705},
            22988: {'recipient': "Tunisia", 'initiative_pk': 7704},
            12522: {'recipient': "Tunisia", 'initiative_pk': 6819},
            19923: {'recipient': "Tunisia", 'initiative_pk': 2794},
            27673: {'recipient': "Vietnam", 'initiative_pk': 8170},
            20011: {'recipient': "Vietnam", 'initiative_pk': 7721},
            19105: {'recipient': "Egypt", 'initiative_pk': 3785},
        }

        self.logger.info(u"Start procedure")
        for pk, value in source_projects.iteritems():
            recipient = Recipient.objects.get(name=value['recipient'])

            try:
                initiative = Initiative.objects.get(pk=value['initiative_pk'], recipient_temp=recipient)
            except ObjectDoesNotExist:
                self.logger.error("Cannot find initiative with pk:{} and recipient:{}, skip".format(value['initiative_pk'], recipient.name))
                continue
            try:
                project = Project.objects.get(pk=pk, recipient=recipient)
            except ObjectDoesNotExist:
                self.logger.error("Cannot find project with pk:{} and recipient:{}, skip".format(pk, recipient.name))
                continue
            else:
                self.logger.info("Working on proj:{}, initiative:{}".format(pk, initiative.pk))

                initiative = self.update_fields(initiative, project)
                initiative.save()
                project.initiative=initiative
                project.save()
                initiative = self.update_related_objects(initiative, project)
                initiative.save()
                
        self.logger.info(u"Finish")