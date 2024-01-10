# coding=utf-8
__author__ = 'stefano'
import logging
from django.core.management.base import BaseCommand
from django.db.transaction import set_autocommit, commit
from openaid.projects.models import Initiative, Project, Activity

# This one-time procedure get data from projects and transfer it to Initiative.
# for the logic of this mapping look for "Mappatura Project - initiative" on Google drive

class Command(BaseCommand):
    help = 'Fills Initiative selected fields with data from the most recent Project.' \
           ' Plus links Reports, Problems, Doc, Photos objs to Initiative'
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

    status_order = ['100','75','50','25','0','-']

    def get_most_advanced_status(self, project_set):
        # gets more advanced status in the project set
        for status in self.status_order:
            if project_set.filter(status=status).count() > 0:
                return status
        return '-'

    def update_fields(self, initiative):

        project_set = Project.objects.filter(initiative=initiative).order_by('-last_update')
        if project_set.count() == 0:
            return initiative

        # gets the project with last update most recent
        project_last_update = project_set[0]
        project_last_activity = project_set[0]
        activity_set = Activity.objects.filter(project__initiative=initiative).order_by('-year')
        #gets project with most recent activity connected
        if activity_set.count() > 0:
            project_last_activity_pk = Activity.objects.filter(project__initiative=initiative).order_by('-year').values_list('project',flat=True)[0]
            project_last_activity = Project.objects.get(pk=project_last_activity_pk)

        # loops on every field that has to be updated and updates if the conditions apply
        for project_fieldname, initiative_fieldname in self.field_map.iteritems():

            if project_fieldname == 'sector':
                field_value = getattr(project_last_activity, project_fieldname)
                if field_value is not None:
                    # only consider sector value that are LEAF nodes, no children
                    if field_value.get_children().count() != 0:
                        self.logger.error("Initiative:{}. Cannot copy SECTOR VALUE: {} from Project, this Sector is not a leaf node! SKIP".format(initiative,field_value))
                        continue

            if project_fieldname == 'recipient':
                field_value = getattr(project_last_activity, project_fieldname)
            elif project_fieldname == 'status':
                field_value = self.get_most_advanced_status(project_set)
                # STATUS: if the proj.status is == 100 => Almost completed
                # translates the value to 90 for Almost completed in Initiative
                # because in Initiative there is a status for "COMPLETED' which has value=100
                if field_value == '100':
                    field_value = '90'
            else:
                field_value = getattr(project_last_update, project_fieldname)

            if field_value is not None:
                initiative.__setattr__(initiative_fieldname, field_value)

        return initiative

    def update_related_objects(self, initiative):

        # updates documents and photo set getting the photos and docs from the projects
        initiative.document_set = initiative.documents()
        initiative.photo_set = initiative.photos()

        # updates reports and problems with initiative link (project link will later be removed by migrations)
        for r in initiative.reports():
            r.initiative = initiative
            r.save()

        for prob in initiative.problems():
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


        set_autocommit(False)
        self.logger.info(u"Start procedure")
        for index, initiative in enumerate(Initiative.objects.all().order_by('code')):
            self.logger.debug(u"Update Initiative:'{}'".format(initiative))

            initiative = self.update_fields(initiative)

            initiative = self.update_related_objects(initiative)

            initiative.save()
            #         commits every N initiatives
            if index % 500 == 0:
                self.logger.info(u"Reached Initiative:'{}'".format(initiative))
                commit()

        #             final commit
        commit()
        set_autocommit(True)
        self.logger.info(u"Finished updating {} initiatives".format(Initiative.objects.all().count()))