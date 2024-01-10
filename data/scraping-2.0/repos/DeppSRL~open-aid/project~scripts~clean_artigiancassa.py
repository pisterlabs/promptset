from django.db.models import Count
from openaid.codelists.models import Agency
from openaid.projects.models import Initiative, Activity


def run():

    artigiancassa = Agency.objects.get(code=9)

    # aggiorno i valori con NULL se sono impostati su 0.0
    print 'Aggiorno i commitment di %d activities' % Activity.objects.filter(commitment=0.0).count()
    Activity.objects.filter(commitment=0.0).update(commitment=None)
    print 'Aggiorno i disbursement di %d activities' % Activity.objects.filter(disbursement=0.0).count()
    Activity.objects.filter(disbursement=0.0).update(disbursement=None)

    # rimozione Activities senza disbursement E commitment
    print 'Rimuovo %d activities di artigiancassa con disbursement E commitment nulli' % artigiancassa.activity_set.filter(
        disbursement__isnull=True,
        commitment__isnull=True
    ).count()
    artigiancassa.activity_set.filter(
        disbursement__isnull=True,
        commitment__isnull=True
    ).delete()

    # rimozione delle Initiatives senza Activities
    print 'Rimouvo %d initiatives di artigiancassa senza progetti' % Initiative.objects.filter(
        project__agency=artigiancassa,
    ).annotate(
        tot=Count('project__activity')
    ).filter(tot=0).count()
    Initiative.objects.filter(
        project__agency=artigiancassa,
    ).annotate(
        tot=Count('project__activity')
    ).filter(tot=0).delete()

    # rimozione dei Projects senza Activities
    print 'Rimuovo %d projects di artgiancassa senza activities' % artigiancassa.project_set.annotate(tot=Count('activity')).filter(tot=0).count()
    artigiancassa.project_set\
        .annotate(tot=Count('activity'))\
        .filter(tot=0)\
        .delete()