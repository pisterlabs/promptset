# coding=utf-8
__author__ = 'stefano'
import logging
from django.core.exceptions import ObjectDoesNotExist
from pprint import pprint
from django.core.management.base import BaseCommand
from openaid.projects.models import Initiative, NewProject


class Command(BaseCommand):
    help = 'Fills Initiative selected fields with data from data from NewProject table plus Photos'
    logger = logging.getLogger('openaid')


    # maps the field name between new project (keys) and initiative (value)
    field_map = {
        'title_it': 'title_it',
        'title_en': 'title_en',
        'description_it': 'description_temp_it',
        'description_en': 'description_temp_en',
    }

    special_cases = {
        "9377 (Fondo Esperti e Fondo in Loco) - F.ROT/AID 99/009/01":"009377",
        "0010521":"010521",
    }

    # the followings initiatives code only need to copy photos, not overwrite fields
    only_copy_photos = [
        '009331',
        '010074',
        '010214',
        '010311',
        '010345',
        '010232',
    ]

    def update_fields(self, new_project, initiative):
        # loops on every field that has to be updated and updates if the conditions apply
        for newproject_fieldname, initiative_fieldname in self.field_map.iteritems():

            field_value = getattr(new_project, newproject_fieldname)

            if field_value is not None:
                initiative.__setattr__(initiative_fieldname, field_value)

        return initiative

    def copy_photos(self, new_project, initiative):

        # updates photo set getting the photos from the new projects
        initiative.photo_set.add()
        for pic in new_project.photo_set.all():
            initiative.photo_set.add(pic)
        return initiative

    def get_code(self,new_project):
        # gets correct 6 chars code from np.number
        code = new_project.number.strip()

        # deals with hand picked special cases
        if code in self.special_cases.keys():
            return self.special_cases[code]

        code = code.replace("AID ","")
        if code is None or code == '':
            return code
        code_split_slash = code.split("/")
        code_split_dot = code.split(".")

        if len(code_split_slash)>1:
            if len(code_split_slash[0]) <=6:
                code = code_split_slash[0]
            else:
                return -1
        elif len(code_split_dot)>1:
            if len(code_split_dot[0]) <=6:
                code = code_split_dot[0]
            else:
                return -1
        elif len(code) <= 6:
            code = code
        else:
            # malformed code
            return -1
        # check that the string contains only numbers
        if code.isdigit():
            return code.zfill(6)
        else:
            return -1

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

        counters = {'null':0, 'existing':0, 'malformed':0, 'processed':0}

        self.logger.info(u"Start procedure")
        for index, new_project in enumerate(NewProject.objects.all().order_by('pk')):
            self.logger.debug(u"Update new proj with pk:'{}'".format(new_project.pk))

            code = self.get_code(new_project)
            if code is not None and code != '' and code != -1:
                initiative, created = Initiative.objects.get_or_create(code=code)
                if created is False:
                    self.logger.debug(u"Initiative with code:'{}' already exist".format(code))
                    counters['existing'] += 1
                else:
                    self.logger.info(u"Created Initiative with code:'{}'".format(code))

            else:
                if code == None or code == '':
                    self.logger.error(u"NewProj:{} has code None or ''. Skip".format(new_project.pk))
                    counters['null'] += 1
                elif code == -1:
                    self.logger.error(u"NewProj:{} - Code '{}' is malformed, cannot process. Skip".format(new_project.pk, new_project.number))
                    counters['malformed'] += 1
                continue

            if initiative is None:
                self.logger.critical("Initiative cannot be None here. ERROR")
                exit()

            if initiative.code not in self.only_copy_photos:
                initiative = self.update_fields(new_project, initiative)

            initiative = self.copy_photos(new_project, initiative)
            initiative.save()
            counters['processed'] += 1

        self.logger.info(u"Finished transfering {} new proj".format(counters['processed']))
        pprint(counters)