import csvkit


def run():
    from openaid.projects.models import Initiative, Project
    for i, row in enumerate(csvkit.DictReader(open('initiatives_full.csv')), start=1):

        initiative, created = Initiative.objects.get_or_create(
            code=row['code'].zfill(6),
            defaults={
                'title_it': row['title'],
                'country': row['country'] if row['country'] != '(vuoto)' else '',
                'total_project_costs': row['total'],
                'grant_amount_approved': row['grant'],
                'loan_amount_approved': row['loan'],
            }
        )

        projects = Project.objects.filter(number__startswith='%s/' % initiative.code).update(initiative=initiative)

        print '%d] Created %s%s' % (i, repr(initiative), (' associated with %d projects' % projects) if projects else '')