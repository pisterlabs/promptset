from csvkit import DictReader
from openaid.projects.models import Project


def run():

    titoli = DictReader(open('last_titles.csv'))

    num_to_title = dict([(row['NUM.PROG.'].strip(), row['TITOLO PROG.'].strip()) for row in titoli])

    for project in Project.objects.all():
        for activity in project.activity_set.all().order_by('year'):
            project.number = activity.number
        if project.number in num_to_title:
            project.title_it = num_to_title[project.number]
        project.save()
