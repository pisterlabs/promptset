from django.contrib.contenttypes.generic import GenericRelation
from django.core.validators import MaxValueValidator, MinValueValidator
from django.core.exceptions import ValidationError
from django.core.urlresolvers import reverse
from django.db import models, IntegrityError
from django.db.models import Sum
from django.utils.translation import ugettext as _
from django.conf import settings
from model_utils import Choices
from openaid import utils
from openaid.projects import fields
import logging

logger = logging.getLogger(__name__)


class ChannelReported(models.Model):
    name = models.CharField(max_length=1000)

    def __unicode__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Channel reported"


class Markers(models.Model):
    biodiversity = fields.MarkerField()
    climate_adaptation = fields.MarkerField()
    climate_mitigation = fields.MarkerField()
    desertification = fields.DesertificationMarkerField()
    environment = fields.MarkerField()
    gender = fields.MarkerField()
    pd_gg = fields.MarkerField()
    trade = fields.MarkerField()

    @property
    def names(self):
        return ['biodiversity', 'climate_adaptation', 'climate_mitigation', 'desertification', 'environment', 'gender',
                'pd_gg', 'trade']

    def merge(self, markers, save=False):
        updates = 0

        for field in self.names:
            mark = getattr(markers, field, None)
            if mark is not None and getattr(self, field) != mark:
                setattr(self, field, mark)
                updates += 1

        if save and updates > 0:
            self.save()
        return updates

    def __unicode__(self):
        return ("{}" * 8).format(*[getattr(self, name) or '-' for name in self.names])

    class Meta:
        verbose_name_plural = "Markers"


class MarkedModel(models.Model):
    markers = models.ForeignKey(Markers, null=True, blank=True)

    def merge_markers(self, markers, save=False):
        updates = 0

        if self.markers is None:
            self.markers = Markers()
            updates += 1

        updates += self.markers.merge(markers, False)

        if save and updates > 0:
            self.save()
        return updates

    class Meta:
        abstract = True


class CodelistsModel(models.Model):
    recipient = models.ForeignKey('codelists.Recipient', verbose_name=_('Country'), blank=True, null=True)
    agency = models.ForeignKey('codelists.Agency', null=True, blank=True)
    aid_type = models.ForeignKey('codelists.AidType', null=True, blank=True)
    channel = models.ForeignKey('codelists.Channel', null=True, blank=True)
    finance_type = models.ForeignKey('codelists.FinanceType', null=True, blank=True)
    sector = models.ForeignKey('codelists.Sector', verbose_name=_('Main Sector'), null=True, blank=True)


    def merge_codelists(self, activity, save=False):
        updates = 0

        for codelist in ('recipient', 'agency', 'aid_type', 'channel', 'finance_type', 'sector'):

            value = getattr(activity, codelist, None)

            if not value or value == getattr(self, codelist):
                continue

            if codelist == 'recipient':
                logger.warning(
                    'Merge Activity %s in %s: Cambiamento di recipient non previsto (%s o %s?) [update ignorato del recipient]' % (
                        activity,
                        self,
                        value,
                        getattr(self, codelist)
                    ))
                continue

            setattr(self, codelist, value)
            updates += 1

        if save and updates > 0:
            self.save()
        return updates

    class Meta:
        abstract = True


class Project(CodelistsModel, MarkedModel):
    STATUS_CHOICES = Choices(
        ('-', 'Not available'),
        ('0', '0%'),
        ('25', '25%'),
        ('50', '50%'),
        ('75', '75%'),
        ('100', 'Almost completed'),
    )

    initiative = models.ForeignKey('projects.Initiative', blank=True, null=True, on_delete=models.SET_NULL)
    title = models.CharField(max_length=500, blank=True)
    description = models.TextField(_('Abstract'), blank=True)
    crsid = models.CharField(max_length=128, blank=True)
    number = models.CharField(max_length=128, blank=True, verbose_name=_('N. ID DGCS'))
    start_year = models.PositiveSmallIntegerField(validators=[MinValueValidator(1900.0), MaxValueValidator(2100.0)])
    end_year = models.PositiveSmallIntegerField(validators=[MinValueValidator(1900.0), MaxValueValidator(2100.0)])
    expected_start_year = models.IntegerField(blank=True, null=True,
                                              validators=[MinValueValidator(1900.0), MaxValueValidator(2100.0)])
    expected_completion_year = models.IntegerField(blank=True, null=True,
                                                   validators=[MinValueValidator(1900.0), MaxValueValidator(2100.0)])
    has_focus = models.BooleanField(_('Focus'), default=False)
    last_update = models.DateField(_('Last update'), null=True, auto_now=True)
    outcome = models.TextField(_('Main Outcome'), blank=True)
    beneficiaries = models.TextField(_('Beneficiaries'), blank=True)
    beneficiaries_female = models.FloatField(verbose_name=_('of which females (%)'),
                                             help_text=_('Beneficiaries of which females (%)'), blank=True, null=True)

    status = models.CharField(_('Status'), max_length=3, help_text=_('Progress based on Approved vs Disbursed'),
                              choices=STATUS_CHOICES, default='-')
    is_suspended = models.BooleanField(verbose_name=_('suspended'), default=False)
    total_project_costs = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    other_financiers = models.TextField(blank=True, verbose_name=_('Other funders'))
    loan_amount_approved = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    grant_amount_approved = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    counterpart_authority = models.CharField(max_length=500, blank=True)
    email = models.EmailField(_('Officer in charge (email)'), blank=True)
    location = models.TextField(blank=True)
    photo_set = GenericRelation('attachments.Photo')
    document_set = GenericRelation('attachments.Document')

    # start/end date of the project: these values are read from activities values (this should change in the future)
    # or a mng task
    # todo: better modellation of start/end dates through 1:N relationship with a "dates" table
    expected_start_date = models.DateField(blank=True, null=True)
    completion_date = models.DateField(blank=True, null=True)

    def get_initiative(self):
        try:
            return Initiative.objects.get(code=self.number.split('/')[0])
        except Initiative.DoesNotExist:
            return None

    @classmethod
    def get_top_projects(cls, qnt=settings.TOP_ELEMENTS_NUMBER, order_by=None, year=None, **filters):
        if year:
            filters['year'] = year
        projects = Activity.objects.filter(**filters).order_by('project').distinct('project').values('project',
                                                                                                     'commitment')

        def order_by_commitment(project):
            return -1 * ( project.get('commitment') or 0)

        projects = sorted(projects, key=order_by or order_by_commitment)[:qnt]
        return cls.objects.filter(pk__in=map(lambda p: p.get('project'), projects))

    def activities(self, year=None):
        if not hasattr(self, '_activities'):
            self._activities = list(
                self.activity_set.all().prefetch_related('recipient', 'agency', 'aid_type', 'channel', 'finance_type',
                                                         'sector'))
        return filter(lambda a: a.year == year, self._activities) if year else self._activities

    def _activities_map(self, field, activities=None, year=None, skip_none=False):
        activities = activities or self.activities(year=year)
        activities = list(map(lambda a: getattr(a, field), activities)) if activities else []
        if skip_none:
            activities = filter(lambda a: a is not None, activities)
        return activities

    def years_range(self):
        return sorted(self._activities_map('year'))

    def recipients(self):
        return self._activities_map('recipient')

    def agencies(self, year=None):
        return self._activities_map('agency', year=year)

    def aid_types(self):
        return self._activities_map('aid_type', skip_none=True)

    def channels(self):
        return self._activities_map('channel', skip_none=True)

    def finance_types(self):
        return self._activities_map('finance_type', skip_none=True)

    def sectors(self):
        return self._activities_map('sector', skip_none=True)

    def purpose(self, year=None):
        return self._activities_map('sector', year=None, skip_none=True)[0]

    def channel_reported(self, year=None):
        return self._activities_map('channel_reported', year=year)[0]

    def commitments(self, year=None):
        return self._activities_map('commitment', year=year)

    def commitment(self, year=None):
        return sum(self._activities_map('commitment', year=year or self.end_year, skip_none=True), 0.0)

    def total_commitment(self):
        return sum(self._activities_map('commitment', skip_none=True), 0.0)

    def disbursements(self, year=None):
        return self._activities_map('disbursement', year=year)

    def disbursement(self, year=None):
        return sum(self._activities_map('disbursement', year=year or self.end_year, skip_none=True), 0.0)

    def total_disbursement(self):
        return sum(self._activities_map('disbursement', skip_none=True), 0.0)

    def flow_type(self, year=None):
        for a in self.activities(year=year):
            return a.get_flow_type_display()
        return None

    def bi_multi(self, year=None):
        for a in self.activities(year=year):
            return a.get_bi_multi_display()
        return None

    def is_ftc(self, year=None):
        return any(self._activities_map('is_ftc', year=year))

    def is_pba(self, year=None):
        return any(self._activities_map('is_pba', year=year))

    def is_investment(self, year=None):
        return any(self._activities_map('is_investment', year=year))

    def get_absolute_url(self):
        return reverse('projects:project-detail', kwargs={'pk': self.pk})

    def update_from_activities(self, save=False):
        activity_updates = 0
        markers_updates = 0
        for activity in self.activity_set.all().order_by('year'):

            if not self.title_en and activity.title_en:
                self.title_en = activity.title_en
                activity_updates += 1

            if activity.year < self.start_year:
                self.start_year = activity.year
                activity_updates += 1

            if activity.year > self.end_year:
                self.end_year = activity.year
                activity_updates += 1

            activity_updates += self.merge_codelists(activity, False)
            markers_updates += self.merge_markers(activity.markers, False)

            if not self.number or self.number != activity.number:
                self.number = activity.number

        if save:
            markers_updates > 0 and self.markers.save()
            activity_updates > 0 and self.save()

        return activity_updates, markers_updates

    def __unicode__(self):
        return "Project:%s:%s" % (self.crsid, self.recipient)

    def __repr__(self):
        return u"<Project(id=%d, crsid='%s', recipient='%s')>" % (
            self.pk, self.crsid, self.recipient
        )

    class Meta:
        unique_together = (('crsid', 'recipient'),)


class Activity(CodelistsModel, MarkedModel):
    REPORT_TYPES = Choices(
        # prese da resources/crs/Codelist04042014.osd:Nature of submission
        (0, _('Unknown')),
        (1, _('New activity reported')),
        (2, _('Revision')),
        (3, _('Previously reported activity')),
        # increase/decrease of earlier commitment, disbursement on earlier commitment
        (5, _('Provisional data')),
        (8, _('Commitment = Disbursement')),
    )

    FLOW_TYPES = Choices(
        # prese da resources/crs/dsd.xml:CL_CRS1_FLOW
        (0, _('Unknown')),
        (11, _('ODA Grants')),
        (12, _('ODA Grant-Like')),
        (13, _('ODA Loans')),
        (14, _('Other Official Flows')), # (non Export Credit)
        (19, _('Equity Investment')),
        (30, _('Private Grants')),
        (100, _('Official Development Assistance')),
    )

    BI_MULTI_TYPES = Choices(
        # prese da resources/crs/Codelist04042014.osd:Bi_Multi
        (0, _('Unknown')),
        (1, _('Bilateral')),
        (2, _('Multilateral')),
        (3, _('Bilateral, core contributions to NGOs and other private bodies / PPPs')),
        (7, _('Bilateral, ex-post reporting on NGOs\' activities funded through core contributions')),
        (4, _('Multilateral outflows')),
        (6, _('Private sector outflows')),
    )

    project = models.ForeignKey(Project, null=True, blank=True)
    crsid = models.CharField(max_length=128, blank=True)
    year = models.IntegerField(validators=[MinValueValidator(1900.0), MaxValueValidator(2100.0)])
    number = models.CharField(max_length=128, blank=True)
    title = models.CharField(max_length=500, blank=True)
    description = models.TextField(blank=True)
    long_description = models.TextField(blank=True)
    geography = models.CharField(max_length=500, blank=True)
    report_type = models.PositiveSmallIntegerField(_('Nature of submission'), choices=REPORT_TYPES)
    flow_type = models.PositiveSmallIntegerField(_('Flow type'), choices=FLOW_TYPES)
    bi_multi = models.IntegerField(_('Bi/Multilateral'), choices=BI_MULTI_TYPES)

    is_ftc = models.BooleanField(_('Free Standing Technical Cooperation'), default=False)
    is_pba = models.BooleanField(_('Programme Based Approaches'), default=False)
    is_investment = models.BooleanField(_('Investment Project'), default=False)

    # money parameters
    # NOTA BENE: QUI SONO MILIONI DI EURO / DI USD!!!
    commitment = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    commitment_usd = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    disbursement = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    disbursement_usd = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])

    # other parameters
    grant_element = models.FloatField(blank=True, null=True, validators=[MinValueValidator(0.0)])
    number_repayment = models.PositiveIntegerField(blank=True, null=True)
    expected_start_date = models.DateField(blank=True, null=True)
    completion_date = models.DateField(blank=True, null=True)
    commitment_date = models.DateField(blank=True, null=True)

    # external relations
    channel_reported = models.ForeignKey(ChannelReported, blank=True, null=True)

    def merge(self, activity, save=False):

        updates = 0

        for field in ['number', 'title', 'description', 'long_description', 'geography',
                      'number_repayment', 'expected_start_date', 'completion_date', 'commitment_date',
                      'channel_reported',
                      'is_ftc', 'is_pba', 'is_investment']:
            if field in ['title', 'description', 'long_description']:
                for field_i18n in ['_it', '_en']:
                    field_i18n = ''.join([field, field_i18n])
                    if not getattr(self, field_i18n) and getattr(activity, field_i18n):
                        setattr(self, field_i18n, getattr(activity, field_i18n))
                        updates += 1
            else:
                if not getattr(self, field) and getattr(activity, field):
                    setattr(self, field, getattr(activity, field))
                    updates += 1

        for field in ['report_type', 'flow_type', 'bi_multi', ]:
            self_value, activity_value = getattr(self, field), getattr(activity, field)
            if self_value == activity_value:
                continue
            if self_value == 0 and activity_value > 0:
                setattr(self, field, getattr(activity, field))
                updates += 1
            elif getattr(activity, field) > 0 and getattr(self, field) > 0:
                logger.warning('Merge Activity %s in %s: Conflitto sul campo %s (%s o %s?) [update non eseguito]' % (
                    repr(activity), repr(self), field, activity_value, self_value
                ))

        for field in ['commitment', 'commitment_usd', 'disbursement', 'disbursement_usd', 'grant_element']:
            self_value, activity_value = getattr(self, field) or 0.0, getattr(activity, field) or 0.0
            setattr(self, field, self_value + activity_value)
            updates += 1
            if self_value == activity_value:
                logger.warning(
                    'Merge Activity %s in %s: Entrambe le Activity hanno lo stesso valore per il campo %s [update eseguito sommandoli]' % (
                        repr(activity), repr(self), field
                    ))

        markers_updates = self.merge_markers(activity.markers)
        updates += self.merge_codelists(activity, False)

        if save:
            markers_updates > 0 and self.markers.save()
            updates > 0 and self.save()
        return updates, markers_updates

    def save(self, *args, **kwargs):
        if not self.commitment and self.commitment_usd:
            self.commitment = utils.currency_converter(self.commitment_usd, self.year)
        if not self.disbursement and self.disbursement_usd:
            self.disbursement = utils.currency_converter(self.disbursement_usd, self.year)
        return super(Activity, self).save(*args, **kwargs)

    def __unicode__(self):
        return u"{project}:{year}".format(year=self.year, project=self.project)

    def __repr__(self):
        return u"<Activity(id=%d, project=%s, year=%s, commitment=%s, disbursement=%s)>" % (
            self.pk, self.project, self.year, self.commitment, self.disbursement
        )

    class Meta:
        ordering = ('-year', 'number', 'title')
        verbose_name_plural = "Activities"


class Organization(models.Model):
    """
    Organizzazioni a cui vanno i fondi multilaterali.
    I Projects sono relativi ai fondi bilaterali.
    """

    parent = models.ForeignKey('Organization', null=True, blank=True)
    acronym = models.CharField(max_length=24, unique=True, null=False, blank=False, default='')
    name = models.CharField(max_length=255)
    order = models.IntegerField(null=True, blank=True, default=0)

    def __unicode__(self):
        return self.name


class AnnualFunds(models.Model):
    """
    Rappresenta i fondi multilaterali anno per anno delle Organization.
    """
    year = models.PositiveSmallIntegerField()
    organization = models.ForeignKey(Organization)
    commitment = models.FloatField(blank=True, default=0.0)
    disbursement = models.FloatField(blank=True, default=0.0)

    def __unicode__(self):
        return '{} {}: {}/{}'.format(self.organization, self.year, self.commitment, self.disbursement)

    class Meta:
        unique_together = ("year", "organization")
        verbose_name_plural = "Annual funds"

    @staticmethod
    def get_multilateral_data(year, type=None):
        multilateral_data = []

        if type == 'commitment':
            sum_aggregate = {'sum': Sum('commitment')}
        elif type == 'disbursement':
            sum_aggregate = {'sum': Sum('disbursement')}
        else:
            raise Exception

        for main_organization in Organization.objects.filter(parent__isnull=True).order_by('order'):

            main_org_dict = {
                'name': main_organization.name,
                'pk': main_organization.pk,
                'sum': AnnualFunds.objects.filter(year=year, organization=main_organization).aggregate(**sum_aggregate)[
                    'sum'],
                'organizations': []
            }

            organizations = Organization.objects.filter(parent=main_organization).order_by('name')
            for org in organizations:
                main_org_dict['organizations'].append({
                    'name': org.name,
                    'sum': AnnualFunds.objects.filter(year=year, organization=org).aggregate(**sum_aggregate)['sum']
                })

            multilateral_data.append(main_org_dict)

        return multilateral_data


class Utl(models.Model):
    name = models.CharField(max_length=200)
    city = models.CharField(max_length=200)

    user = models.OneToOneField('auth.User', blank=True, null=True, related_name='utl')
    nation = models.OneToOneField('codelists.Recipient', related_name='+')
    recipient_set = models.ManyToManyField('codelists.Recipient', related_name='utl_set')

    class Meta:
        verbose_name_plural = "UTL"


# todo: this has to be removed when the transition to the new Initiative is finished
class TemporaryCheck(models.Model):
    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if self.project is not None and self.initiative is not None:
            if self.project.initiative != self.initiative:
                raise ValidationError(
                    'Object cannot have foreign key to Project AND Initiative which are not connected. Choose one.')
        super(TemporaryCheck, self).save(*args, **kwargs)


class Problem(TemporaryCheck):
    event = models.TextField(_('Unforeseen event'), blank=True)
    impact = models.TextField(blank=True)
    actions = models.TextField(_('Actions carried out'), blank=True)
    project = models.ForeignKey(Project, default=None, null=True, blank=True)
    initiative = models.ForeignKey('projects.Initiative', default=None, null=True, blank=True)


class Report(TemporaryCheck):
    TYPES = Choices(
        (1, _('Technical Assistance/Consultancy and related expenses')),
        (2, _('Works, Supply, Services')),
        (3, _('Contributions')),
        (4, _('Scholarships')),
    )

    PROCUREMENT_PROCEDURES = Choices(
        (1, _('Call for Proposal')),
        (2, _('Direct Contracting/Direct Assignment')),
        (3, _('Competitive Bidding')),
        (4, _('Call for Grant')),
    )

    PROCUREMENT_NOTICE = Choices(
        (1, _('Tender preparation')),
        (2, _('Tender launched')),
        (3, _('Selection procedure')),
        (4, _('Contract awarded')),
    )

    type = models.IntegerField(choices=TYPES, default=None, null=True)
    procurement_procedure = models.IntegerField(choices=PROCUREMENT_PROCEDURES, default=None, null=True)
    procurement_notice = models.IntegerField(choices=PROCUREMENT_NOTICE, default=None, null=True)
    status = models.CharField(max_length=200, blank=True)
    number = models.CharField(verbose_name=_('N. ID DGCS'), max_length=128, blank=True)
    awarding_entity = models.CharField(max_length=1000, blank=True)
    description = models.TextField(blank=True, verbose_name=_('Observation'))
    project = models.ForeignKey(Project, default=None, null=True, blank=True)
    initiative = models.ForeignKey('projects.Initiative', default=None, null=True, blank=True)


    class Meta:
        verbose_name = _('Procurement')
        verbose_name_plural = _('Procurements')


class NewProject(CodelistsModel):
    title = models.CharField(max_length=500, blank=True)
    number = models.CharField(verbose_name=_('N. ID DGCS'), blank=True, max_length=100)
    description = models.TextField(verbose_name=_('Abstract'), blank=True)
    year = models.PositiveSmallIntegerField(null=True, blank=True)
    commitment = models.FloatField(help_text=_('Migliaia di euro'), blank=True, null=True)
    disbursement = models.FloatField(help_text=_('Migliaia di euro'), blank=True, null=True)
    photo_set = GenericRelation('attachments.Photo')

    def get_absolute_url(self):
        return reverse('projects:newproject-detail', kwargs={'pk': self.pk})


class Initiative(models.Model):
    STATUS_CHOICES = Choices(
        ('-', 'Not available'),
        ('0', '0%'),
        ('25', '25%'),
        ('50', '50%'),
        ('75', '75%'),
        ('90', 'Almost completed'),
        ('100', 'Completed'),
    )
    code = models.CharField(_('N.ID Iniziativa DGCS'), max_length=6, unique=True, null=False, blank=False)
    title = models.CharField(max_length=1000, null=True, blank=True, default='')
    # NOTA: VALORI IN EURO
    total_project_costs = models.FloatField(_('Total project costs for Italian Entities'),
                                            help_text=_(
                                                'Value in Euro. Example: for 10.000 Euro insert 10000. Do not insert dots or commas for decimals or thousands'),
                                            blank=True, null=True, validators=[MinValueValidator(0.0), ])
    loan_amount_approved = models.FloatField(help_text=_(
        'Value in Euro. Example: for 10.000 Euro insert 10000. Do not insert dots or commas for decimals or thousands'),
                                             blank=True, null=True, validators=[MinValueValidator(0.0), ])
    grant_amount_approved = models.FloatField(help_text=_(
        'Value in Euro. Example: for 10.000 Euro insert 10000. Do not insert dots or commas for decimals or thousands'),
                                              blank=True, null=True, validators=[MinValueValidator(0.0), ])

    # new fields
    # last update field is an imported /insered field about the last update of the record
    last_update_temp = models.DateField(_('Data aggiornamento scheda'), blank=True, null=True, default=None)
    description_temp = models.TextField(_('Abstract'), blank=True)
    recipient_temp = models.ForeignKey('codelists.Recipient', verbose_name=_('Country'), blank=True, null=True)
    outcome_temp = models.TextField(_('Main Outcome'), blank=True)
    purpose_temp = models.ForeignKey('codelists.Sector', verbose_name=_('Purpose code'), null=True, blank=True)
    beneficiaries_temp = models.TextField(_('Beneficiaries'), blank=True)
    beneficiaries_female_temp = models.FloatField(verbose_name=_('of which females (%)'),
                                                  help_text=_('Beneficiaries of which females (%)'), blank=True,
                                                  null=True,
                                                  validators=[MinValueValidator(0.0), MaxValueValidator(100.0)])

    status_temp = models.CharField(_('Status'), max_length=3, help_text=_('Progress based on Approved vs Disbursed'),
                                   choices=STATUS_CHOICES, default='-')
    is_suspended_temp = models.BooleanField(verbose_name=_('suspended'), default=False)
    start_year = models.PositiveSmallIntegerField(null=True, blank=True, default=None,
                                                  validators=[MinValueValidator(1900.0), MaxValueValidator(2100.0)])
    end_year = models.PositiveSmallIntegerField(null=True, blank=True, default=None,
                                                validators=[MinValueValidator(1900.0), MaxValueValidator(2100.0)])
    other_financiers_temp = models.TextField(blank=True, verbose_name=_('Other funders'))
    counterpart_authority_temp = models.CharField(_('Counterpart authority'), max_length=500, blank=True)
    email_temp = models.EmailField(_('Officer in charge (email)'), blank=True)
    location_temp = models.TextField(_('Location'), blank=True)
    # ATTACHMENTS
    photo_set = GenericRelation('attachments.Photo')
    document_set = GenericRelation('attachments.Document')
    # created at / updated at: automatical fields for create/update time. not shown in backend
    updated_at = models.DateTimeField(auto_now=True, blank=True, null=True, default=None)
    created_at = models.DateTimeField(auto_now_add=True, blank=True, null=True, default=None)
    has_focus = models.BooleanField(_('Focus'), default=False)

    @classmethod
    def get_top_initiatives(cls, is_home=False, **filters):
        # excludes from top initiatives those sectors that are for staff wages and other
        excluded_sectors = settings.OPENAID_INITIATIVE_PURPOSE_EXCLUDED

        # selects the base set of Initiatives for this case, applying various filters
        base_set = Initiative.objects. \
            exclude(status_temp='100'). \
            exclude(purpose_temp__code__in=excluded_sectors). \
            filter(**filters). \
            distinct()

        # if it's the home page skip the ordering by FOCUS, otherwise use the FOCUS ordering with total proj costs
        top_initiatives_not_null = base_set.exclude(total_project_costs__isnull=True)
        if not is_home:
            top_initiatives_not_null = top_initiatives_not_null.order_by('-has_focus', '-total_project_costs')
        else:
            top_initiatives_not_null = top_initiatives_not_null.order_by('-total_project_costs')

        top_initiatives = list(top_initiatives_not_null)
        # adds up initiatives with NULL cost at the end of the list, if any,
        # this avoids to have initiatives with NULL values on top of the list
        top_initiatives_null = base_set.exclude(total_project_costs__isnull=False).order_by('title')

        if top_initiatives_null.count() > 0:
            top_initiatives.extend(list(top_initiatives_null))

        return top_initiatives

    @property
    def last_update(self):
        dates = list(self._project_fields_map('last_update', skip_none=True))
        if len(dates) == 0:
            return None
        return max(dates)

    def locations(self):
        return list(self._project_fields_map('location', skip_none=True))

    def years_range(self):
        range = set()
        for project in self.projects():
            range.add(project.start_year)
            range.add(project.end_year)

        return sorted(range)

    def years_stats(self):
        for year in self.years_range():
            commitment = disbursement = 0.0
            for project in self.projects():
                commitment += sum([x for x in project.commitments(year=year) if x])
                disbursement += sum([x for x in project.disbursements(year=year) if x])

            yield (year, commitment, disbursement)

    @property
    def description(self):
        try:
            return [p.description for p in self.projects() if p.description][0]
        except IndexError:
            return ''

    @property
    def outcome(self):
        return self._get_first_project_value('outcome')

    def photos(self):
        return list(self._projects_map('photo_set', 'all'))

    @property
    def image(self):
        try:
            return self.photos()[0]
        except IndexError:
            return ''

    def projects(self):
        if not hasattr(self, '_projects'):
            self._projects = list(self.project_set.all().prefetch_related('recipient'))
        return self._projects

    def _projects_map(self, field, callback=None, skip_none=False):
        for project in self.projects():
            method = getattr(project, field)
            for value in method() if not callback else getattr(method, callback)():
                if skip_none and value is None:
                    continue
                yield value

    def documents(self):
        return list(self._projects_map('document_set', 'all'))

    def problems(self):
        return list(self._projects_map('problem_set', 'all'))

    def reports(self):
        return list(self._projects_map('report_set', 'all'))

    def recipients(self):
        return list(set(self._projects_map('recipients')))

    def aid_types(self):
        aid_types = []
        for aid_type in list(self._projects_map('aid_types')):
            aid_types.append(aid_type.get_root())
        return set(aid_types)

    def sectors(self):
        sectors = []
        for sector in list(self._projects_map('sectors')):
            sectors.append(sector.get_root())
        return set(sectors)

    def channels(self):
        channels = []
        for channel in list(self._projects_map('channels')):
            channels.append(channel.get_root())
        return set(channels)

    def finance_type(self):
        return self._get_first_project_value('finance_types')

    def flow_type(self):
        return self._get_first_project_value('flow_type')

    def agency(self):
        return self._get_first_project_value('agencies')

    def _get_first_project_value(self, field, skip_values=None):
        for project in self.projects():
            if not hasattr(project, field):
                continue
            value = getattr(project, field, None)

            if hasattr(value, '__call__'):
                value = value()
            if hasattr(value, '__iter__'):
                for v in value:
                    if v:
                        return v
                else:
                    continue
            if not value:
                continue
            elif skip_values and value in skip_values:
                continue
            return value
        return None

    @property
    def purpose(self):
        return self._get_first_project_value('purpose')

    @property
    def expected_start_date(self):
        return self._get_first_project_value('expected_start_date')

    @property
    def expected_completion_year(self):
        return self._get_first_project_value('expected_completion_year')

    @property
    def is_suspended(self):
        return self._get_first_project_value('is_suspended')

    @property
    def beneficiaries(self):
        return self._get_first_project_value('beneficiaries')

    @property
    def beneficiaries_female(self):
        return self._get_first_project_value('beneficiaries_female')

    @property
    def other_financiers(self):
        return self._get_first_project_value('other_financiers')

    def _project_fields_map(self, field, skip_none=False):
        for project in self.projects():
            value = getattr(project, field)
            if value is None and skip_none:
                continue
            yield value

    @property
    def counterpart_authority(self):
        return self._get_first_project_value('counterpart_authority')

    @property
    def email(self):
        return self._get_first_project_value('email')

    @property
    def status(self):
        return self._get_first_project_value('status', skip_values=['0', '-'])

    @property
    def crsid(self):
        return self._get_first_project_value('crsid')

    @property
    def bi_multi(self):
        return self._get_first_project_value('bi_multi')

    @property
    def is_ftc(self):
        return self._get_first_project_value('is_ftc')

    @property
    def is_pba(self):
        return self._get_first_project_value('is_pba')

    @property
    def is_investment(self):
        return self._get_first_project_value('is_investment')

    def save(self, *args, **kwargs):
        if len(self.code) != 6:
            self.code = self.code.zfill(6)

        if (self.title_it is None or self.title_it == '') and (self.title_en is None or self.title_en == ''):
            raise ValidationError("Initiative must have Italian or English title, fill at least one")

        return super(Initiative, self).save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('projects:initiative-detail', kwargs={'code': self.code})

    def __unicode__(self):

        country = ""
        if self.recipient_temp:
            country = self.recipient_temp.name

        return '%s:%s "%s"' % (self.code, country, self.title)

    def __repr__(self):
        country = ""
        if self.recipient_temp:
            country = self.recipient_temp.name

        return u"<Initiative(id=%d, code=%s, title=\"%s\", country=%s)>" % (
            self.pk, self.code, self.title, country
        )

    class Meta:
        verbose_name = _('Initiative')
        verbose_name_plural = _('Initiatives')
