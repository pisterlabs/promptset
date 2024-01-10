from django.conf import settings
from django.db.models import Min, Max, Count
from faq.models import Question
from openaid.pages import urls as pages_urls
from .codelists import models
from .projects import models as projects_models
from .projects.forms import FacetedProjectSearchForm
from openaid import utils

YEAR_FIELD = 'selected_year'
YEAR_GET_FIELD = 'year'
START_YEAR = projects_models.Activity.objects.aggregate(Min('year'))['year__min']
END_YEAR = projects_models.Activity.objects.aggregate(Max('year'))['year__max']

YEARS_RANGE_FIELD = 'years'
YEARS = range(START_YEAR, END_YEAR+1)

def project_context(request):

    recipients = []
    for root_recipient in models.Recipient.objects.root_nodes().prefetch_related('children'):
        children = models.Recipient.objects.add_related_count(
            root_recipient.get_children(),
            projects_models.Activity,
            'recipient', 'activity_count', cumulative=True
        )
        setattr(root_recipient, 'activity_count', sum(map(lambda x: x.activity_count, children)))
        if len(children) == 1:
            root_recipient = children[0]
            children = []
        recipients.append(
            (root_recipient, children)
        )

    sectors = models.Sector.objects.add_related_count(
        models.Sector.objects.root_nodes(),
        projects_models.Activity,
        'sector', 'activity_count', cumulative=True
    )

    agencies = models.Agency.objects.annotate(activity_count=Count('activity')).all()

    aid_types = models.AidType.objects.add_related_count(
        models.AidType.objects.root_nodes(),
        projects_models.Activity,
        'aid_type', 'activity_count', cumulative=True
    )
    
    return {
        'project_name': settings.PROJECT_NAME,
        'available_languages': map(lambda x: x[0], settings.LANGUAGES),
        'footer_sections': pages_urls.footer_sections,
        'recipients': sorted(recipients, key=lambda r: r[0].code),
        'sectors': sectors,
        'agencies': agencies,
        'aid_types': aid_types,
        YEAR_FIELD: utils.sanitize_get_param(int, request.GET.get(YEAR_GET_FIELD), END_YEAR, top=END_YEAR, length=4),
        YEARS_RANGE_FIELD: YEARS,
        'search_form': FacetedProjectSearchForm(request.GET),
        'addthis_profile': settings.ADDTHIS_PROFILE,
        'donor_code': settings.OPENAID_CRS_DONOR,
        'site_full_url': request.build_absolute_uri('/')[:-1],
        'page_full_url': request.build_absolute_uri(),
        'faq_list': Question.objects.all()[:4],
        'top_elements_number': settings.TOP_ELEMENTS_NUMBER,
    }