from __future__ import unicode_literals
from openaid.projects.models import Project

__author__ = 'joke2k'


def generate_crsid():
    generate_crsid.X += 1
    return 'X%d' % generate_crsid.X
generate_crsid.X = 0


def run():

    for p in Project.objects.filter(crsid=''):
        activities = sorted(p.activities(), key=lambda x: x.year)

        p.crsid = generate_crsid()
        p.start_year = p.end_year = activities[0].year
        p.save()
        print 'Project %s has new crsid: %s' % (p, p.crsid)

        if len(activities) <= 1:
            continue

        for a in activities[1:]:
            new_p = Project.objects.create(
                crsid=generate_crsid(),
                recipient=p.recipient,
                start_year=a.year,
                end_year=a.year,
            )
            a.project = new_p
            a.save()
            print 'Created new project %s for activity %s' % (new_p.crsid, a)

        #print p.pk, p.recipient, p._activities_map('disbursement'), map(lambda x: x.code, p._activities_map('agency'))