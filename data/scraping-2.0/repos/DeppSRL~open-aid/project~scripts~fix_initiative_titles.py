from openaid.projects.models import Initiative


def run():

    for i in Initiative.objects.all():

        i.title_en = i.title_it
        i.save()