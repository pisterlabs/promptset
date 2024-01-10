from __future__ import absolute_import
from django import template
from django.conf import settings
from django.contrib.humanize.templatetags.humanize import intcomma
from django.db.models import Sum
from django.template.defaultfilters import floatformat
from openaid.projects import models as projects_models
from openaid.codelists import models as codelists_models
from openaid import contexts

register = template.Library()

@register.filter(is_safe=True)
def currency(amount):
    return intcomma(floatformat((amount or 0.0) * settings.OPENAID_MULTIPLIER, 0))

@register.filter(is_safe=True)
def currency_initiatives(amount):
    return intcomma(floatformat((amount or 0.0) * 1, 0))


@register.filter
def unique(args):
    return set([a for a in args if a])

def _get_code_list_items(instance, model):
    if instance and isinstance(instance, model):
        return instance.get_children()
    return model.objects.root_nodes()

@register.inclusion_tag('commons/main_panel.html', takes_context=True)
def crs_stats(context, instance=None, year=None, show_map=True):
    widget = context.get('widget', False)
    start_year = contexts.START_YEAR
    end_year = contexts.END_YEAR
    year = int(year or context.get(contexts.YEAR_FIELD, None) or end_year)

    filters = {
        'year': year,
    }

    if instance:
        filters['%s__in' % instance.code_list] = instance.get_descendants_pks(include_self=True)

    sectors = _get_code_list_items(instance, codelists_models.Sector)
    agencies = codelists_models.Agency.objects.all() if not isinstance(instance, codelists_models.Agency) else []
    aid_types = _get_code_list_items(instance, codelists_models.AidType)

    statistify = lambda item: (item, item.get_total_commitment(**filters))
    cleaner = lambda items: filter(lambda x: x[1], sorted(map(statistify, items), key=tot_order, reverse=True))
    tot_order = lambda item: item[1]

    activities = projects_models.Activity.objects.all()
    if len(filters.keys()):
        activities = activities.filter(**filters)

    selected_facet = instance.code_list_facet if instance else None

    # commitment_sum = 0 if activities.aggregate(Sum('commitment'))['commitment__sum'] is None else activities.aggregate(Sum('commitment'))['commitment__sum']
    commitment_sum = activities.aggregate(Sum('commitment'))['commitment__sum']
    if commitment_sum is None:
        commitment_sum = 0
    # disbursements_sum = 0 if activities.aggregate(Sum('disbursement'))['disbursement__sum'] is None else activities.aggregate(Sum('disbursement'))['disbursement__sum']
    disbursements_sum = activities.aggregate(Sum('disbursement'))['disbursement__sum']
    if disbursements_sum is None:
        disbursements_sum = 0

    ctx = {
        'selected_year': year,
        'selected_code_list': instance,
        'selected_facet': selected_facet,
        'start_year': start_year,
        'end_year': end_year,
        'sector_stats': cleaner(sectors),
        'agency_stats': cleaner(agencies),
        'aid_stats': cleaner(aid_types),
        'projects_count': activities.distinct('project').count(),
        'commitments_sum': commitment_sum,
        'disbursements_sum': disbursements_sum,
        'years': range(start_year, end_year + 1),
        'show_map': show_map,
        'widget': widget
    }

    ctx['columns'] = 3 if len(ctx['sector_stats']) and len(ctx['agency_stats']) and len(ctx['aid_stats']) else 2
    if instance and hasattr(instance, 'parent') and instance.parent and instance.code_list == 'sector':
        ctx['columns'] = 2
        ctx['sector_stats'] = None

    main_organizations = projects_models.Organization.objects.filter(parent__isnull=True)
    if not selected_facet:
        # multilateral aid have only to consider the top categories, otherwise the amount is doubled
        multi_projects = projects_models.AnnualFunds.objects.filter(organization__in=main_organizations,year=year).aggregate(
            multi_commitments_sum=Sum('commitment'),
            multi_disbursements_sum=Sum('disbursement'),
        )
        multi_commitments_sum = multi_projects['multi_commitments_sum']
        multi_disbursements_sum = multi_projects['multi_disbursements_sum']

        # adds the % commitment and % disbursement for every organization to display in the template
        ctx.update(multi_projects)

        ctx.update({
            'total_commitments_sum': (multi_commitments_sum or 0.0) + commitment_sum,
            'total_disbursements_sum': (multi_disbursements_sum or 0.0) + disbursements_sum,
        })

        # drilldown pie code
        ctx.update({
            'multi_stats_commitment': projects_models.AnnualFunds.get_multilateral_data(year=year, type='commitment'),
            'multi_stats_disbursement': projects_models.AnnualFunds.get_multilateral_data(year=year, type='disbursement')
        })

    ctx['years_values'] = []
    for year in ctx['years']:
        if instance:
            year_commitment = instance.get_total_commitment(year=year)
            year_disbursement = instance.get_total_disbursement(year=year)
        else:
            # Calculate data for homepage (no codelist selected)
            multi_projects = projects_models.AnnualFunds.objects.filter(
                organization__in=main_organizations,
                year=year).aggregate(
                    multi_commitments_sum=Sum('commitment'),
                    multi_disbursements_sum=Sum('disbursement'),
                )
            year_commitment = projects_models.Activity.objects.filter(year=year).aggregate(Sum('commitment'))['commitment__sum']
            year_commitment += (multi_projects['multi_commitments_sum'] or 0.0)
            year_disbursement = projects_models.Activity.objects.filter(year=year).aggregate(Sum('disbursement'))['disbursement__sum']
            year_disbursement += (multi_projects['multi_disbursements_sum'] or 0.0)

        ctx['years_values'].append([
            year,
            (year_commitment or 0.0) * settings.OPENAID_MULTIPLIER,
            (year_disbursement or 0.0) * settings.OPENAID_MULTIPLIER
        ])

    return ctx
