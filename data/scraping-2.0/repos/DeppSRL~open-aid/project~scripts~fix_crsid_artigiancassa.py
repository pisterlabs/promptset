# coding=utf-8
from __future__ import unicode_literals
import csvkit
from django.db import IntegrityError, transaction
from django.db.models import Count
from openaid.codelists.models import Agency
from openaid.projects.models import Project, Activity, Initiative
from collections import Counter


__author__ = 'joke2k'


def update_crsids(filename):
    for row in csvkit.DictReader(open(filename)):
        activity_id = row['openaid_id']
        new_crsid = row['CRSID-OK']
        initiative_number = row['Initiative number']
        # clean number
        if len(initiative_number.split(' ')) > 0:
            initiative_number = initiative_number.split(' ')[0]
        if len(initiative_number) > 0:
            initiative_number = initiative_number.zfill(6)
        project_number = '/'.join([initiative_number, row['projectnumber']])

        updates_markers = False

        try:
            activity = Activity.objects.get(pk=activity_id)
        except Activity.DoesNotExist:
            print '- Impossibile trovare Activity.pk = %s' % activity_id
            continue

        try:

            new_project = Project.objects.get(crsid=new_crsid, recipient__code=activity.recipient.code)

            try:
                conclict_activity = new_project.activity_set.get(year=activity.year)

                if conclict_activity == activity:
                    continue

                _, updates_markers = conclict_activity.merge(activity, save=False)
                activity, conclict_activity = conclict_activity, activity
                print '- Cancello %s dopo il merge in %s' % (repr(conclict_activity), repr(activity))
                conclict_activity.delete()

            except Activity.DoesNotExist:
                pass

        except Project.DoesNotExist:

            new_project = Project.objects.create(
                crsid=new_crsid,
                recipient=activity.recipient,
                start_year=activity.year,
                end_year=activity.year,
                number=project_number,
            )

            print ('- Nuovo progetto per Activity %s non trovato con newCRSID:%s' % (
                repr(activity), new_crsid))

        finally:
            activity.crsid = new_crsid
            activity.project = new_project
            activity.number = project_number
            if updates_markers:
                activity.markers.save()
            if project_number:
                activity.number = project_number
                new_project.number = project_number
                try:
                    initiative = Initiative.objects.get(code=new_project.number.split('/')[0])
                    new_project.initiative = initiative
                except Initiative.DoesNotExist:
                    print '- Nessuna Initiative trovata con codice: %s' % (project_number)
            activity.save()

            new_project.update_from_activities(save=True)

            #print '- %s aggiornata' % repr(activity)



    # cancello tutti i progetti senza Activity
    qs = Project.objects.annotate(activities=Count('activity')).filter(activities=0)
    print 'Cancello %s Project senza Activity' % (
        qs.count(),
    )
    qs.delete()


def run(filename='Artigiancassa.csv'):

    update_crsids(filename)
