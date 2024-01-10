import asyncio
import logging
from collections import defaultdict
from datetime import timedelta
from functools import reduce
from io import StringIO

from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db.models import Count, F, Prefetch, Q, Sum
from django.db.transaction import atomic
from django.utils import timezone
from openai.error import RateLimitError
from rest_framework import status
from rest_framework.response import Response
from slack_sdk import WebClient

from jvapp.apis._apiBase import JobVyneAPIView, SUCCESS_MESSAGE_KEY, WARNING_MESSAGES_KEY, get_error_response, \
    get_success_response
from jvapp.apis.geocoding import LocationParser, get_raw_location
from jvapp.apis.job import LocationView
from jvapp.apis.job_subscription import JobSubscriptionView
from jvapp.apis.notification import MessageGroupView
from jvapp.apis.social import SocialLinkView, SocialLinkJobsView
from jvapp.apis.stripe import StripeCustomerView
from jvapp.apis.user import UserView
from jvapp.models.abstract import PermissionTypes
from jvapp.models.employer import ConnectionTypeBit, Employer, EmployerAuthGroup, EmployerConnection, \
    EmployerReferralBonusRule, \
    EmployerReferralRequest, EmployerAts, EmployerSlack, EmployerSubscription, JobDepartment, EmployerJob, \
    EmployerJobApplicationRequirement, EmployerReferralBonusRuleModifier, EmployerPermission, EmployerFile, \
    EmployerFileTag
from jvapp.models.job_seeker import JobApplication
from jvapp.models.location import Location
from jvapp.models.social import SocialLink
from jvapp.models.tracking import MessageThread, MessageThreadContext
from jvapp.models.user import JobVyneUser, PermissionName, UserEmployerPermissionGroup
from jvapp.permissions.employer import IsAdminOrEmployerOrReadOnlyPermission, IsAdminOrEmployerPermission
from jvapp.serializers.employer import get_serialized_auth_group, get_serialized_employer, \
    get_serialized_employer_billing, get_serialized_employer_bonus_rule, get_serialized_employer_file, \
    get_serialized_employer_file_tag, get_serialized_employer_job, \
    get_serialized_employer_referral_request
from jvapp.utils import csv, ai
from jvapp.utils.ai import PromptError
from jvapp.utils.data import AttributeCfg, coerce_bool, coerce_int, is_obfuscated_string, set_object_attributes
from jvapp.utils.datetime import get_datetime_or_none
from jvapp.utils.email import ContentPlaceholders, get_domain_from_email, send_django_email
from jvapp.utils.sanitize import sanitize_html

__all__ = (
    'EmployerView', 'EmployerJobView', 'EmployerAuthGroupView', 'EmployerUserView', 'EmployerUserActivateView',
    'EmployerSubscriptionView', 'EmployerInfoView',
)

from jvapp.utils.slack import raise_slack_exception_if_error

BATCH_UPDATE_SIZE = 100
logger = logging.getLogger(__name__)


class EmployerView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerOrReadOnlyPermission]
    
    def get(self, request, employer_id=None):
        employer = None
        if employer_id:
            employer_id = int(employer_id)
            employer = self.get_employers(employer_id=employer_id)
        elif employer_key := self.query_params.get('employer_key'):
            try:
                employer = self.get_employers(employer_key=employer_key)
            except Employer.DoesNotExist:
                return get_error_response('This is a bad URL')
        elif social_link_id := self.query_params.get('social_link_id'):
            try:
                social_link = SocialLink.objects.get(id=social_link_id)
            except SocialLink.DoesNotExist:
                return get_error_response('This is a bad URL')
            employer_id = social_link.employer_id
            if not employer_id:
                return get_error_response('This is a bad URL')
            employer = self.get_employers(employer_id=employer_id)
        
        if employer:
            data = get_serialized_employer(
                employer,
                is_employer=self.user and (
                        self.user.is_admin
                        or (self.user.employer_id == employer_id and self.user.is_employer)
                )
            )
            return Response(status=status.HTTP_200_OK, data=data)
        
        employers = list(Employer.objects.all().values('id', 'employer_name'))
        employers.sort(key=lambda e: e['employer_name'])
        return Response(status=status.HTTP_200_OK, data=employers)
    
    @atomic
    def put(self, request, employer_id):
        employer = self.get_employers(employer_id=employer_id)
        if logo := self.files.get('logo'):
            employer.logo = logo[0]
        
        set_object_attributes(
            employer,
            self.data,
            {
                'email_domains': AttributeCfg(is_ignore_excluded=True),
                'notification_email': AttributeCfg(is_ignore_excluded=True),
                'color_primary': AttributeCfg(is_protect_existing=True),
                'color_secondary': AttributeCfg(is_protect_existing=True),
                'color_accent': AttributeCfg(is_protect_existing=True),
                'is_manual_job_entry': AttributeCfg(is_protect_existing=True),
            }
        )
        
        employer.jv_check_permission(PermissionTypes.EDIT.value, self.user)
        employer.save()
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Updated employer data'
        })
    
    @staticmethod
    def get_employers(employer_id=None, employer_key=None, employer_filter=None):
        if employer_id:
            employer_filter = Q(id=employer_id)
        elif employer_key:
            employer_filter = Q(employer_key=employer_key)
        
        employers = Employer.objects \
            .select_related('default_bonus_currency', 'applicant_tracking_system') \
            .prefetch_related(
                'subscription',
                'employee',
                'employee__employer_permission_group',
                'employee__employer_permission_group__permission_group',
                'employee__employer_permission_group__permission_group__permissions',
                'ats_cfg',
                'slack_cfg'
            ) \
            .filter(employer_filter) \
            .annotate(employee_count=Count('employee'))
        
        if employer_id or employer_key:
            if not employers:
                raise Employer.DoesNotExist
            return employers[0]
        
        return employers
    

class EmployerAtsView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerPermission]
    
    @atomic
    def post(self, request):
        if not (employer_id := self.data.get('employer_id')):
            return get_error_response('An employer ID is required')
        
        ats = self.get_new_ats_cfg(self.user, employer_id, self.data['name'])
        self.update_ats(self.user, ats, self.data)
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Successfully created ATS configuration'
        })
    
    @atomic
    def put(self, request):
        if not (ats_id := self.data.get('id')):
            return get_error_response('An ATS ID is required')
        
        ats = EmployerAts.objects.get(id=ats_id)
        self.update_ats(self.user, ats, self.data)
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Successfully updated ATS configuration'
        })
    
    @atomic
    def delete(self, request, ats_id):
        from jvapp.apis.ats import get_ats_api  # Avoid circular import
        ats = EmployerAts.objects.get(id=ats_id)
        ats.jv_check_permission(PermissionTypes.DELETE.value, self.user)
        ats_api = get_ats_api(ats)
        try:
            ats_api.delete_webhooks()
        except ConnectionError:
            # The refresh token may have expired. Since we are deleting the configuration, this doesn't matter
            pass
        ats.delete()
        ats_jobs = EmployerJob.objects.filter(employer_id=ats.employer_id, ats_job_key__isnull=False)
        for job in ats_jobs:
            job.close_date = timezone.now().date()
        EmployerJob.objects.bulk_update(ats_jobs, ['close_date'])
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Successfully deleted ATS configuration'
        })
    
    @staticmethod
    def get_new_ats_cfg(user, employer_id, ats_name):
        existing_ats = EmployerAts.objects.filter(employer_id=employer_id)
        if existing_ats:
            # Check whether there is an existing cfg for the ats that is supposed to be created (e.g. Greenhouse)
            same_ats = next((ats for ats in existing_ats if ats.name == ats_name), None)
            if same_ats:
                return same_ats
            
            # If the ats cfg is for a different provider, we need to delete the old one
            # Example: ATS name is Lever and the existing ATS is for Greenhouse
            if existing_ats:
                for delete_ats in existing_ats:
                    delete_ats.jv_check_permission(PermissionTypes.DELETE.value, user)
                existing_ats.delete()
        
        return EmployerAts(employer_id=employer_id)
    
    @staticmethod
    @atomic
    def update_ats(user, ats, data):
        from jvapp.apis.ats import LeverAts, get_ats_api  # Avoid circular import
        set_object_attributes(ats, data, {
            'name': AttributeCfg(is_ignore_excluded=True),
            'email': AttributeCfg(is_ignore_excluded=True),
            'job_stage_name': AttributeCfg(is_ignore_excluded=True),
            'employment_type_field_key': AttributeCfg(is_ignore_excluded=True),
            'salary_range_field_key': AttributeCfg(is_ignore_excluded=True),
            'access_token': AttributeCfg(is_ignore_excluded=True),
            'refresh_token': AttributeCfg(is_ignore_excluded=True),
            'is_webhook_enabled': AttributeCfg(is_ignore_excluded=True)
        })
        
        if ats.name == LeverAts.NAME and ats.email:
            ats_api = get_ats_api(ats)
            user_data = ats_api.get_or_create_jobvyne_lever_user(ats.email)
            ats.api_key = user_data['id']
        else:
            api_key = data.get('api_key')
            if api_key and not is_obfuscated_string(api_key):
                ats.api_key = api_key
        
        permission_type = PermissionTypes.EDIT.value if ats.id else PermissionTypes.CREATE.value
        ats.jv_check_permission(permission_type, user)
        ats.save()
        
        
class EmployerSlackView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerPermission]
    
    def put(self, request, slack_cfg_id):
        if not (employer_id := self.data.get('employer_id')):
            return get_error_response('An employer ID is required')
        if not self.user.has_employer_permission(PermissionName.MANAGE_EMPLOYER_SETTINGS.value, employer_id):
            return get_error_response('You do not have permission for this operation')
        slack_cfg = EmployerSlack.objects.select_related('employer').get(id=slack_cfg_id)
        self.update_slack_cfg(self.user, slack_cfg, self.data)
        
        # Slack bot needs to be part of the channel to post to it
        client = WebClient(token=slack_cfg.oauth_key)
        if slack_cfg.jobs_post_channel:
            resp = client.conversations_join(channel=slack_cfg.jobs_post_channel)
            raise_slack_exception_if_error(resp)
        if slack_cfg.referrals_post_channel:
            resp = client.conversations_join(channel=slack_cfg.referrals_post_channel)
            raise_slack_exception_if_error(resp)
        
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Updated Slack configuration'
        })
    
    def delete(self, request, slack_cfg_id):
        try:
            slack_cfg = EmployerSlack.objects.get(id=slack_cfg_id)
        except EmployerSlack.DoesNotExist:
            return get_error_response(f'No Slack configuration exists with ID = {slack_cfg_id}')
        if not self.user.has_employer_permission(PermissionName.MANAGE_EMPLOYER_SETTINGS.value, slack_cfg.employer_id):
            return get_error_response('You do not have permission for this operation')
        slack_cfg.jv_check_permission(PermissionTypes.DELETE.value, self.user)
        slack_cfg.delete()
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Slack configuration deleted'
        })
        
    @staticmethod
    @atomic
    def update_slack_cfg(user, slack_cfg, data):
        set_object_attributes(slack_cfg, data, {
            'is_enabled': None,
            'jobs_post_channel': None,
            'jobs_post_dow_bits': None,
            'jobs_post_tod_minutes': None,
            'jobs_post_max_jobs': None,
            'referrals_post_channel': None,
            'modal_cfg_is_salary_required': None,
        })
        slack_cfg.jv_check_permission(PermissionTypes.EDIT.value, user)
        slack_cfg.save()
    
    @staticmethod
    def get_slack_cfg(employer_id):
        slack_cfg = EmployerSlack.objects.select_related('employer').filter(employer_id=employer_id)
        if slack_cfg:
            return slack_cfg[0]
        return None


class EmployerBillingView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerPermission]
    
    def get(self, request, employer_id):
        employer = Employer.objects.get(id=employer_id)
        return Response(status=status.HTTP_200_OK, data=get_serialized_employer_billing(employer))
    
    def put(self, request, employer_id):
        employer = Employer.objects.get(id=employer_id)
        
        # Check permissions
        employer.jv_check_permission(PermissionTypes.EDIT.value, self.user)
        billing_permission = PermissionName.MANAGE_BILLING_SETTINGS.value
        has_billing_permission = self.user.has_employer_permission(billing_permission, employer.id)
        if not has_billing_permission:
            employer._raise_permission_error(billing_permission)
        
        # Street address isn't important to normalize. We are just using the address to determine taxes
        location_text = f'{self.data["city"]}, {self.data["state"]}, {self.data["country"]} {self.data.get("postal_code")}'
        raw_location, _ = get_raw_location(location_text)
        if not raw_location:
            raise ValueError(f'Could not locate address for {location_text}')
        employer.street_address = self.data.get('street_address')
        employer.street_address_2 = self.data.get('street_address_2')
        employer.city = raw_location['city']
        employer.state = raw_location['state']
        employer.country = raw_location['country_short']
        employer.postal_code = raw_location.get('postal_code')
        employer.billing_email = self.data['billing_email']
        employer.save()
        
        StripeCustomerView.create_or_update_customer(employer)
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Billing information updated successfully'
        })


class EmployerReferralRequestView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerPermission]
    
    def get(self, request):
        if not (employer_id := self.query_params.get('employer_id')):
            return get_error_response('An employer ID is required')
        
        referral_requests = EmployerReferralRequest.jv_filter_perm(
            self.user,
            EmployerReferralRequest.objects.filter(employer_id=employer_id)
        )
        return Response(status=status.HTTP_200_OK, data=[
            get_serialized_employer_referral_request(rr) for rr in referral_requests
        ])
    
    def post(self, request):
        if not (employer_id := self.data.get('employer_id')):
            return get_error_response('An employer ID is required')
        
        referral_request = EmployerReferralRequest(employer_id=employer_id)
        self.update_referral_request(self.user, referral_request, self.data)
        error_msg = self.send_referral_requests(self.user, referral_request, self.data)
        if error_msg:
            return get_error_response(error_msg)
        
        return self.get_success_response(self.data)
    
    def put(self, request):
        if not (request_id := self.data.get('request_id')):
            return get_error_response('A request ID is required')
        
        referral_request = EmployerReferralRequest(id=request_id)
        self.update_referral_request(self.user, referral_request, self.data)
        error_msg = self.send_referral_requests(self.user, referral_request, self.data)
        if error_msg:
            return get_error_response(error_msg)
        
        return self.get_success_response(self.data)
    
    @staticmethod
    def get_success_response(data):
        recipient_count = len(data['user_ids'])
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'Sent referral requests to {recipient_count} {"recipients" if recipient_count != 1 else "recipient"}'
        })
    
    @staticmethod
    def send_referral_requests(user, referral_request, data):
        """
        :return: Error message or None
        """
        referral_request.jv_check_permission(PermissionTypes.CREATE.value, user)
        if not (user_ids := data.get('user_ids')):
            return 'No users were specified'
        
        # TODO: Make this a background task since it can take a long time with lots of employees
        recipients = JobVyneUser.objects.filter(id__in=user_ids)
        jobs_list = None
        referral_request_message_thread = EmployerReferralRequestView.get_or_create_referral_request_message_thread(
            referral_request.employer_id,
            referral_request
        )
        referral_links = {
            rl.owner_id: rl for rl in
            SocialLinkView.get_or_create_employee_referral_links(recipients, referral_request.employer)
        }
        for idx, recipient in enumerate(recipients):
            referral_link = referral_links[recipient.id]
            email_body = referral_request.email_body
            
            referral_link_url = referral_link.get_link_url()
            email_body = email_body.replace(
                ContentPlaceholders.JOB_LINK.value,
                referral_link_url
            )
            
            # Jobs are the same regardless of recipient so we only need to fetch them once
            if idx == 0:
                jobs, _ = SocialLinkJobsView.get_jobs_from_social_link(referral_link)
                if not jobs:
                    return 'No jobs were provided'
                job_titles = list({j.job_title for j in jobs})
                job_titles.sort()
                jobs_list = '<ul>'
                jobs_list += ''.join([f'<li>{job_title}</li>' for job_title in job_titles[:5]])
                if len(job_titles) > 5:
                    jobs_list += '<li>And more jobs viewable on the website</li>'
                jobs_list += '</ul>'
            
            email_body = email_body.replace(
                ContentPlaceholders.JOBS_LIST.value,
                jobs_list
            )
            
            email_body = email_body.replace(
                ContentPlaceholders.EMPLOYEE_FIRST_NAME.value,
                recipient.first_name
            )
            email_body = email_body.replace(ContentPlaceholders.EMPLOYEE_LAST_NAME.value, recipient.last_name)
            
            emails = [recipient.email]
            if recipient.business_email:
                emails.append(recipient.business_email)
            
            for email in emails:
                send_django_email(
                    data['email_subject'],
                    'emails/base_general_email.html',
                    to_email=email,
                    django_context={
                        'is_exclude_final_message': True
                    },
                    html_body_content=email_body,
                    employer=referral_request.employer,
                    is_include_jobvyne_subject=False,
                    message_thread=referral_request_message_thread
                )
        
        return None
    
    @staticmethod
    @atomic
    def update_referral_request(user, referral_request, data):
        set_object_attributes(referral_request, data, {
            'email_subject': None,
            'email_body': AttributeCfg(prop_func=lambda val: sanitize_html(val, is_email=True))
        })
        
        permission_type = PermissionTypes.EDIT.value if referral_request.id else PermissionTypes.CREATE.value
        referral_request.jv_check_permission(permission_type, user)
        referral_request.save()
        
    @staticmethod
    def get_or_create_referral_request_message_thread(employer_id, referral_request):
        employer_message_group = MessageGroupView.get_or_create_employer_message_group(employer_id)
        try:
            return MessageThread.objects.get(
                message_thread_context__referral_request=referral_request,
                message_groups=employer_message_group
            )
        except MessageThread.DoesNotExist:
            message_thread = MessageThread()
            message_thread.save()
            message_thread.message_groups.add(employer_message_group)
            MessageThreadContext(
                message_thread=message_thread,
                referral_request=referral_request
            ).save()
            return message_thread


class EmployerSubscriptionView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerOrReadOnlyPermission]
    INACTIVE_STATUSES = [
        EmployerSubscription.SubscriptionStatus.EXPIRED.value,
        EmployerSubscription.SubscriptionStatus.CANCELED.value
    ]
    ACTIVE_STATUS = EmployerSubscription.SubscriptionStatus.ACTIVE.value
    
    def get(self, request, employer_id):
        employer_id = int(employer_id)
        employer = Employer.objects.prefetch_related('subscription').get(id=employer_id)
        subscription = self.get_subscription(employer)
        has_active_subscription = subscription and subscription.status == self.ACTIVE_STATUS
        active_employees = EmployerSubscriptionView.get_active_employees(employer)
        data = {
            'is_active': has_active_subscription,
            'has_seats': has_active_subscription and (active_employees <= subscription.employee_seats)
        }
        if self.user and getattr(self.user, 'is_employer', None) and (self.user.employer_id == employer_id):
            data['subscription_seats'] = subscription.employee_seats if subscription else 0
            data['active_employees'] = active_employees
        return Response(status=status.HTTP_200_OK, data=data)
    
    @staticmethod
    def get_subscription(employer):
        return next(
            (s for s in employer.subscription.all() if s.status not in EmployerSubscriptionView.INACTIVE_STATUSES),
            None
        )
    
    @staticmethod
    def get_active_employees(employer):
        return employer.employee \
            .annotate(
            employer_user_type_bits=Sum('employer_permission_group__permission_group__user_type_bit', distinct=True)) \
            .filter(
            is_employer_deactivated=False,
            has_employee_seat=True,
            employer_user_type_bits__lt=F('employer_user_type_bits') + (
                    1 * F('employer_user_type_bits').bitand(JobVyneUser.USER_TYPE_EMPLOYEE))
        ) \
            .count()


class EmployerJobView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerOrReadOnlyPermission]
    
    def get(self, request, employer_job_id=None):
        if employer_job_id:
            job = self.get_employer_jobs(employer_job_id=employer_job_id)
            rules = EmployerBonusRuleView.get_employer_bonus_rules(self.user, employer_id=job.employer_id)
            data = get_serialized_employer_job(job, rules=rules, is_include_bonus=True)
        elif employer_id := self.query_params.get('employer_id'):
            employer_id = employer_id[0] if isinstance(employer_id, list) else employer_id
            job_filter = Q(employer_id=employer_id)
            if job_title_filter := self.query_params.get('job_title_filter'):
                job_filter &= Q(job_title__iregex=f'^.*{job_title_filter}.*$')
            if city_ids := self.query_params.getlist('city_ids[]'):
                job_filter &= Q(locations__city_id__in=city_ids)
            if state_ids := self.query_params.getlist('state_ids[]'):
                job_filter &= Q(locations__state_id__in=state_ids)
            if country_ids := self.query_params.getlist('country_ids[]'):
                job_filter &= Q(locations__country_id__in=country_ids)
            if department_ids := self.query_params.getlist('department_ids[]'):
                job_filter &= Q(job_department_id__in=department_ids)
            if job_ids := self.query_params.getlist('job_ids[]'):
                job_filter &= Q(id__in=job_ids)
            is_only_closed = coerce_bool(self.query_params.get('is_only_closed'))
            is_include_closed = coerce_bool(self.query_params.get('is_include_closed'))
            jobs, paginated_jobs = self.get_employer_jobs(employer_job_filter=job_filter, is_only_closed=is_only_closed, is_include_closed=is_include_closed)
            rules = EmployerBonusRuleView.get_employer_bonus_rules(self.user,
                                                                   employer_id=employer_id) if self.user else None
            data = [get_serialized_employer_job(j, rules=rules, is_include_bonus=bool(self.user)) for j in jobs]
        else:
            return Response('A job ID or employer ID is required', status=status.HTTP_400_BAD_REQUEST)
        
        return Response(status=status.HTTP_200_OK, data=data)
    
    def put(self, request):
        if not (job_id := self.data.get('id')):
            return Response('A job ID is required', status=status.HTTP_400_BAD_REQUEST)
        
        employer_job = self.get_employer_jobs(employer_job_id=job_id)
        employer_job = self.update_job(self.user, employer_job, self.data)
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'Updated {employer_job.job_title} job'
        })
    
    def post(self, request):
        if not (employer_id := self.data.get('employer_id')):
            return Response('An employer ID is required', status=status.HTTP_400_BAD_REQUEST)
        
        employer_job = self.update_job(self.user, EmployerJob(employer_id=employer_id), self.data,
                                       is_check_duplicate=True)
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'Added {employer_job.job_title} job'
        })
    
    @staticmethod
    @atomic
    def update_job(user, employer_job, data, is_check_duplicate=False):
        # Make sure user can edit jobs
        permission_type = PermissionTypes.EDIT.value if employer_job.id else PermissionTypes.CREATE.value
        employer_job.jv_check_permission(permission_type, user)
        
        # We need to get and create locations to determine whether the job is a duplicate
        location_parser = LocationParser()
        location_ids = tuple(
            location if coerce_int(location) else location_parser.get_location(location).id
            for location in data['locations']
        )
        
        # Make sure this isn't a duplicate before adding locations to job
        if is_check_duplicate:
            job_title = data['job_title']
            potential_job_duplicates = {
                job.get_key(): job for job in
                EmployerJob.objects.prefetch_related('locations').filter(employer_id=employer_job.employer_id,
                                                                         job_title=job_title)
            }
            employer_job = potential_job_duplicates.get(EmployerJob.generate_job_key(job_title, location_ids),
                                                        employer_job)
        
        set_object_attributes(employer_job, data, {
            'job_title': AttributeCfg(is_ignore_excluded=True),
            'open_date': AttributeCfg(is_ignore_excluded=True,
                                      prop_func=lambda x: get_datetime_or_none(x, as_date=True)),
            'close_date': AttributeCfg(is_ignore_excluded=True,
                                       prop_func=lambda x: get_datetime_or_none(x, as_date=True)),
            'salary_interval': AttributeCfg(is_ignore_excluded=True),
            'salary_floor': AttributeCfg(is_ignore_excluded=True),
            'salary_ceiling': AttributeCfg(is_ignore_excluded=True),
            'employment_type': AttributeCfg(is_ignore_excluded=True),
        })
        
        employer_job.job_description = sanitize_html(data['job_description'])
        if salary_currency := data.get('salary_currency'):
            employer_job.salary_currency_name = salary_currency
        
        # Handle job department - if ID is a string, this is a new department for this employer
        if job_department := data['job_department']:
            try:
                employer_job.job_department_id = int(job_department['id'])
            except ValueError:
                # Check whether job department has already been created by a different employer
                job_departments = JobDepartment.objects.filter(name__iexact=job_department['id'])
                if job_departments:
                    employer_job.job_department = job_departments[0]
                else:
                    new_job_department = JobDepartment(name=job_department['id'])
                    new_job_department.save()
                    employer_job.job_department = new_job_department
        
        # Remove existing locations if this is an existing job
        if employer_job.id:
            employer_job.locations.remove()
        
        employer_job.save()
        
        # Add locations
        job_location_model = employer_job.locations.through
        job_location_model.objects.bulk_create(
            [job_location_model(location_id=l, employerjob=employer_job) for l in location_ids],
            ignore_conflicts=True
        )
        
        return employer_job
    
    @staticmethod
    def get_employer_job_filter(
        employer_job_id=None, is_only_closed=False, is_include_closed=False, is_include_future=False,
        is_allow_unapproved=False, lookback_days=None
    ):
        if employer_job_id:
            return Q(id=employer_job_id)
        
        job_filter = Q()
        if not is_allow_unapproved:
            job_filter &= Q(is_job_approved=True)
        if is_only_closed:
            job_filter &= (Q(close_date__isnull=False) & Q(close_date__lt=timezone.now().date()))
        elif not is_include_closed:
            job_filter &= (Q(close_date__isnull=True) | Q(close_date__gt=timezone.now().date()))
        if not is_include_future:
            end_date = timezone.now().date()
            if lookback_days:
                start_date = timezone.now().date() - timedelta(days=30 * 3)
                job_filter &= Q(open_date__range=(start_date, end_date))
            else:
                job_filter &= Q(open_date__lte=end_date)
        
        return job_filter
    
    @staticmethod
    def get_employer_jobs(
            employer_job_id=None, employer_job_filter=None, order_by=None, applicant_user=None,
            is_include_fetch=True, is_only_closed=False, is_include_closed=False, is_include_future=False,
            is_allow_unapproved=False, lookback_days=None, jobs_per_page=25, page_count=1
    ):
        # NOTE: Be careful adding the order_by argument since it may cause a full table scan if not on an index
        standard_job_filter = EmployerJobView.get_employer_job_filter(
            employer_job_id=employer_job_id, is_only_closed=is_only_closed, is_include_closed=is_include_closed,
            is_include_future=is_include_future, is_allow_unapproved=is_allow_unapproved, lookback_days=lookback_days
        )
        
        jobs = EmployerJob.objects.filter(standard_job_filter)
        if employer_job_filter:
            jobs = jobs.filter(employer_job_filter)
        
        # Jobs can be duplicated when we filter based on location
        # Calling "distinct" on the entire data set is extremely inefficient
        # Instead we get unique job ids and then re-query the database with a much smaller paginated subset
        jobs = jobs.values('id')
        if order_by:
            jobs = jobs.order_by(*order_by)
        paginated_jobs = Paginator(jobs, per_page=jobs_per_page)
        page_count = min(page_count, paginated_jobs.num_pages)
        jobs = paginated_jobs.get_page(page_count)
        job_ids = {job['id'] for job in jobs}
        
        if is_include_fetch:
            locations_prefetch = Prefetch(
                'locations', queryset=Location.objects.select_related('city', 'state', 'country')
            )
            
            jobs = (
                EmployerJob.objects
                .select_related(
                    'job_department',
                    'employer',
                    'employer__applicant_tracking_system',
                    'referral_bonus_currency',
                    'salary_currency'
                )
                .prefetch_related(
                    locations_prefetch,
                    'taxonomy',
                    'taxonomy__taxonomy'
                )
                .filter(id__in=job_ids)
            )
            if applicant_user:
                user_application_prefetch = Prefetch(
                    'job_application',
                    queryset=JobApplication.objects.filter(user=applicant_user),
                    to_attr='user_job_applications'
                )
                jobs = jobs.prefetch_related(user_application_prefetch)
            
        if employer_job_id:
            if not jobs:
                raise EmployerJob.DoesNotExist
            return jobs[0]
        
        return jobs, paginated_jobs
    
    
class EmployerConnectionView(JobVyneAPIView):
    
    def get(self, request):
        pass
    
    @staticmethod
    def get_employer_connections(job=None, user_id=None, group_id=None):
        assert any((job, user_id, group_id))
        connection_filter = Q()
        if job:
            connection_filter &= Q(employer_id=job.employer_id)
        if user_id:
            connection_filter &= Q(user_id=user_id)
        if group_id:
            connection_filter &= Q(group_id=None)
            
        return EmployerConnection.objects.filter(connection_filter)
    
    @staticmethod
    def get_and_update_employer_connection(employer_connection, data):
        if not employer_connection.id:
            try:
                employer_connection = EmployerConnection.objects.get(
                    user_id=employer_connection.user_id, employer_id=employer_connection.employer_id
                )
            except EmployerConnection.DoesNotExist:
                pass
        
        hiring_job = None
        if data['connection_type'] == ConnectionTypeBit.HIRING_MEMBER.value:
            data['connection_type'] = ConnectionTypeBit.CURRENT_EMPLOYEE.value
            hiring_job = data.get('job')
        
        set_object_attributes(employer_connection, data, {
            'connection_type': None,
            'is_allow_contact': None
        })
        
        is_new = not employer_connection.id
        employer_connection.save()
        
        if hiring_job:
            employer_connection.hiring_jobs.add(hiring_job)
        elif job := data.get('job'):
            employer_connection.hiring_jobs.remove(job)
        
        return employer_connection, is_new


class EmployerJobApplicationView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerPermission]
    
    def put(self, request):
        if not (application_id := self.data.get('id')):
            return get_error_response('An application ID is required')
        
        application = self.get_application(application_id)
        has_changed = False
        new_application_status = self.data.get('application_status')
        if new_application_status and new_application_status != application.application_status:
            application.application_status = new_application_status
            application.application_status_dt = timezone.now()
            has_changed = True
        
        if has_changed:
            if not self.has_edit_permission(self.user, application):
                return get_error_response('You do not have the appropriate permissions to edit this application')
            application.save()
        
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Updated job application'
        })
        
    
    @staticmethod
    def has_edit_permission(user, application):
        return (
            (application.employer_job.employer_id == user.employer_id)
            and user.has_employer_permission(PermissionName.MANAGE_EMPLOYER_JOBS.value, application.employer_job.employer_id)
        )
    
    @staticmethod
    def get_application(application_id):
        return JobApplication.objects\
            .select_related('employer_job')\
            .get(id=application_id)


class EmployerJobApplicationRequirementView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerOrReadOnlyPermission]
    
    def get(self, request):
        employer_id = coerce_int(self.query_params.get('employer_id'))
        job_id = coerce_int(self.query_params.get('job_id'))
        assert employer_id or job_id
        if employer_id:
            application_requirements = self.get_application_requirements(employer_ids=[employer_id])
            return Response(
                status=status.HTTP_200_OK,
                data=self.get_consolidated_application_requirements(application_requirements)
            )
        else:
            job = EmployerJob.objects.get(id=job_id)
            application_requirements = self.get_application_requirements(job=job)
            consolidated_requirements = self.get_consolidated_application_requirements(application_requirements)
            application_requirements = self.get_job_application_fields(
                job, consolidated_requirements
            )
            return Response(
                status=status.HTTP_200_OK,
                data=application_requirements
            )
    
    @atomic
    def put(self, request):
        if not (employer_id := self.data.get('employer_id')):
            return get_error_response('An employer ID is required')
        
        application_field = self.data.get('application_field')
        application_requirements = EmployerJobApplicationRequirement.objects.filter(
            employer_id=employer_id,
            application_field=application_field
        )
        
        for app_requirement_key, requirement_attr in (
                ('required', 'is_required'), ('optional', 'is_optional'), ('hidden', 'is_hidden')):
            filters = self.data.get(app_requirement_key)
            is_default = self.data['default'].get(requirement_attr)
            requirement = next((r for r in application_requirements if getattr(r, requirement_attr)), None)
            if any((filters['departments'], filters['jobs'])) or is_default:
                if requirement:
                    requirement.filter_departments.clear()
                    requirement.filter_jobs.clear()
                else:
                    requirement = EmployerJobApplicationRequirement(
                        employer_id=employer_id, application_field=application_field,
                        is_required=False, is_optional=False, is_hidden=False, is_locked=False
                    )
                    setattr(requirement, requirement_attr, True)
                    requirement.save()
                if not is_default:
                    if filters['departments']:
                        requirement.filter_departments.add(*[d['id'] for d in filters['departments']])
                    if filters['jobs']:
                        requirement.filter_jobs.add(*[j['id'] for j in filters['jobs']])
            elif requirement:
                # If there is an old default value, we need to get rid of it
                requirement.delete()
        
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'Updated application requirements for {application_field} field'
        })
    
    @staticmethod
    def get_application_requirements(employer_ids=None, job=None):
        assert employer_ids or job
        application_requirement_filter = Q(employer_id__in=employer_ids or [job.employer_id])
        if job:
            application_requirement_filter &= (
                    Q(filter_jobs__id=job.id) | Q(filter_departments__id=job.job_department_id)
                    | (Q(filter_jobs__isnull=True) & Q(filter_departments__isnull=True))
            )
        
        return EmployerJobApplicationRequirement.objects \
            .prefetch_related(
                'employer',
                'employer__default_bonus_currency',
                'filter_departments',
                'filter_jobs',
                'filter_jobs__salary_currency',
                'filter_jobs__referral_bonus_currency'
            ) \
            .filter(application_requirement_filter)
    
    @staticmethod
    def get_consolidated_application_requirements(application_requirements):
        requirement_template = {
            'application_field': None,
            'default': None,
            'is_locked': None,
            'required': None,
            'optional': None,
            'hidden': None
        }
        consolidated_requirements = defaultdict(lambda: {**requirement_template})
        for requirement in application_requirements:
            consolidated_requirement = consolidated_requirements[requirement.application_field]
            consolidated_requirement['application_field'] = requirement.application_field
            filter_jobs = requirement.filter_jobs.all()
            filter_departments = requirement.filter_departments.all()
            if not any([filter_jobs, filter_departments]):
                consolidated_requirement['default'] = {
                    'is_required': requirement.is_required,
                    'is_optional': requirement.is_optional and (not requirement.is_required),
                    'is_hidden': requirement.is_hidden and (not any([requirement.is_required, requirement.is_optional]))
                }
                consolidated_requirement['is_locked'] = requirement.is_locked
            else:
                key = 'required' if requirement.is_required else ('optional' if requirement.is_optional else 'hidden')
                consolidated_requirement[key] = {
                    'departments': [{'id': f.id, 'name': f.name} for f in
                                    filter_departments] if filter_departments else None,
                    'jobs': [{'id': j.id, 'job_title': j.job_title} for j in filter_jobs] if filter_jobs else None
                }
        
        consolidated_requirements = list(consolidated_requirements.values())
        consolidated_requirements.sort(key=lambda x: (x['is_locked'], x['default']['is_required']), reverse=True)
        
        return consolidated_requirements
    
    @staticmethod
    def get_job_application_fields(job: EmployerJob, consolidated_requirements: list):
        application_fields = {}
        for requirement in consolidated_requirements:
            field_key = requirement['application_field']
            if (required := requirement['required']) and EmployerJobApplicationRequirementView.is_job_filter_match(job,
                                                                                                                   required):
                application_fields[field_key] = {'is_required': True}
            elif (optional := requirement['optional']) and EmployerJobApplicationRequirementView.is_job_filter_match(
                    job, optional):
                application_fields[field_key] = {'is_optional': True}
            elif (hidden := requirement['hidden']) and EmployerJobApplicationRequirementView.is_job_filter_match(job,
                                                                                                                 hidden):
                pass
            else:
                application_fields[field_key] = {**requirement['default']}
        return application_fields
    
    @staticmethod
    def is_job_filter_match(job: EmployerJob, requirement_filter: dict):
        return any([
            requirement_filter['jobs'] and next((j for j in requirement_filter['jobs'] if j['id'] == job.id), None),
            requirement_filter['departments'] and next(
                (d for d in requirement_filter['departments'] if d['id'] == job.job_department_id), None)
        ])


class EmployerJobBonusView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerOrReadOnlyPermission]
    
    def put(self, request):
        jobs = EmployerJob.objects.filter(id__in=self.data['job_ids'])
        jobs_to_update = []
        for job in jobs:
            job.referral_bonus = self.data['referral_bonus']
            job.referral_bonus_currency_id = self.data['referral_bonus_currency']
            jobs_to_update.append(job)
        
        EmployerJob.objects.bulk_update(jobs_to_update, ['referral_bonus', 'referral_bonus_currency_id'],
                                        batch_size=1000)
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'Updated referral bonus for {len(jobs)} {"jobs" if len(jobs) > 1 else "job"}'
        })


class EmployerBonusDefaultView(JobVyneAPIView):
    
    @atomic
    def put(self, request):
        if not (employer_id := self.data.get('employer_id')):
            return Response('An employer ID is required', status=status.HTTP_400_BAD_REQUEST)
        employer = EmployerView.get_employers(employer_id=employer_id)
        
        employer.jv_check_permission(PermissionTypes.EDIT.value, self.user)
        self.user.has_employer_permission(PermissionName.MANAGE_REFERRAL_BONUSES.value, self.user.employer_id)
        
        set_object_attributes(employer, self.data, {
            'default_bonus_amount': None,
            'default_bonus_currency_id': AttributeCfg(form_name='default_bonus_currency'),
            'days_after_hire_payout': None
        })
        employer.save()
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Bonus rule defaults updated'
        })


class EmployerBonusRuleView(JobVyneAPIView):
    
    def get(self, request):
        if not (employer_id := self.query_params.get('employer_id')):
            return Response('An employer ID is required', status=status.HTTP_400_BAD_REQUEST)
        
        rules = self.get_employer_bonus_rules(self.user, employer_id=employer_id)
        return Response(
            status=status.HTTP_200_OK,
            data=[get_serialized_employer_bonus_rule(rule) for rule in rules]
        )
    
    @atomic
    def post(self, request):
        rule = EmployerReferralBonusRule(employer_id=self.data['employer_id'])
        self.update_bonus_rule(self.user, rule, self.data)
        return Response(
            status=status.HTTP_200_OK,
            data={
                SUCCESS_MESSAGE_KEY: 'Referral bonus rule added'
            }
        )
    
    @atomic
    def put(self, request, rule_id):
        rule = self.get_employer_bonus_rules(self.user, rule_id=rule_id)
        self.update_bonus_rule(self.user, rule, self.data)
        return Response(
            status=status.HTTP_200_OK,
            data={
                SUCCESS_MESSAGE_KEY: 'Referral bonus rule updated'
            }
        )
    
    def delete(self, request, rule_id):
        rule = self.get_employer_bonus_rules(self.user, rule_id=rule_id)
        rule.jv_check_permission(PermissionTypes.DELETE.value, self.user)
        rule.delete()
        return Response(
            status=status.HTTP_200_OK,
            data={
                SUCCESS_MESSAGE_KEY: 'Referral bonus rule deleted'
            }
        )
    
    @staticmethod
    def get_employer_bonus_rules(user, rule_id=None, employer_id=None, is_use_permissions=True):
        filter = Q()
        if rule_id:
            filter &= Q(id=rule_id)
        elif employer_id:
            filter &= Q(employer_id=employer_id)
        
        rules = EmployerReferralBonusRule.objects \
            .select_related('bonus_currency') \
            .prefetch_related(
                'include_departments',
                'exclude_departments',
                'include_cities',
                'exclude_cities',
                'include_states',
                'exclude_states',
                'include_countries',
                'exclude_countries',
                'modifier'
            ) \
            .filter(filter)
        
        if is_use_permissions:
            rules = EmployerReferralBonusRule.jv_filter_perm(user, rules)
        
        if rule_id:
            if not rules:
                raise EmployerReferralBonusRule.DoesNotExist
            return rules[0]
        
        return rules
    
    @staticmethod
    @atomic
    def update_bonus_rule(user, bonus_rule, data):
        data['include_job_titles_regex'] = data['inclusion_criteria'].get('job_titles_regex')
        data['exclude_job_titles_regex'] = data['exclusion_criteria'].get('job_titles_regex')
        set_object_attributes(bonus_rule, data, {
            'order_idx': None,
            'include_job_titles_regex': None,
            'exclude_job_titles_regex': None,
            'base_bonus_amount': None,
            'bonus_currency_id': AttributeCfg(form_name='bonus_currency'),
            'days_after_hire_payout': None
        })
        
        permission_type = PermissionTypes.EDIT.value if bonus_rule.id else PermissionTypes.CREATE.value
        bonus_rule.jv_check_permission(permission_type, user)
        bonus_rule.save()
        
        # Clear existing criteria
        for field in [
            'include_departments', 'exclude_departments',
            'include_cities', 'exclude_cities',
            'include_states', 'exclude_states',
            'include_countries', 'exclude_countries'
        ]:
            bonus_rule_field = getattr(bonus_rule, field)
            bonus_rule_field.clear()
        
        for dataKey, prepend_text in (('inclusion_criteria', 'include_'), ('exclusion_criteria', 'exclude_')):
            for criteriaKey, criteriaVals in data[dataKey].items():
                if criteriaKey == 'job_titles_regex':
                    continue
                rule_key = f'{prepend_text}{criteriaKey}'
                bonus_rule_field = getattr(bonus_rule, rule_key)
                for val in criteriaVals:
                    bonus_rule_field.add(val['id'])
        
        bonus_rule.modifier.all().delete()
        modifiers_to_save = []
        for modifier in data.get('modifiers'):
            modifiers_to_save.append(EmployerReferralBonusRuleModifier(
                referral_bonus_rule=bonus_rule,
                type=modifier['type'],
                amount=modifier['amount'],
                start_days_after_post=modifier['start_days_after_post']
            ))
        
        if modifiers_to_save:
            EmployerReferralBonusRuleModifier.objects.bulk_create(modifiers_to_save)


class EmployerBonusRuleOrderView(JobVyneAPIView):
    
    @atomic
    def put(self, request):
        rules = {
            r.id: r for r in
            EmployerBonusRuleView.get_employer_bonus_rules(self.user, employer_id=self.data['employer_id'])
        }
        rule_ids = self.data['rule_ids']
        if len(rules.values()) != len(rule_ids):
            return Response(
                'The length of the new rules order is not equal to the existing number of rules',
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not rules:
            return Response(status=status.HTTP_200_OK)
        
        for order_idx, rule_id in enumerate(rule_ids):
            if not (rule := rules.get(rule_id)):
                return Response(
                    f'Rule with ID = {rule_id} does not exist for this employer'
                )
            
            if order_idx == 0:
                rule.jv_check_permission(PermissionTypes.EDIT.value, self.user)
            
            rule.order_idx = order_idx
        
        EmployerReferralBonusRule.objects.bulk_update(list(rules.values()), ['order_idx'])
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: 'Referral bonus rules order updated'
        })


class EmployerAuthGroupView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerPermission]
    IGNORED_AUTH_GROUPS = [
        JobVyneUser.USER_TYPE_ADMIN, JobVyneUser.USER_TYPE_CANDIDATE,
    ]
    
    def get(self, request):
        employer_id = self.query_params['employer_id']
        auth_groups = self.get_auth_groups(self.user, employer_id)
        all_permissions = EmployerPermission.objects.all()
        return Response(
            status=status.HTTP_200_OK,
            data=[get_serialized_auth_group(ag, all_permissions, auth_groups, self.user) for ag in auth_groups]
        )
    
    @atomic
    def post(self, request):
        auth_group = EmployerAuthGroup(
            name=self.data['name'],
            user_type_bit=self.data['user_type_bit'],
            employer_id=self.data['employer_id']
        )
        auth_group.jv_check_permission(PermissionTypes.CREATE.value, self.user)
        auth_group.save()
        return Response(
            status=status.HTTP_200_OK,
            data={
                'auth_group_id': auth_group.id,
                SUCCESS_MESSAGE_KEY: f'{auth_group.name} group saved'
            }
        )
    
    @atomic
    def put(self, request, auth_group_id):
        auth_group = EmployerAuthGroup.objects.get(id=auth_group_id)
        set_object_attributes(auth_group, self.data, {
            'name': None,
            'user_type_bit': None,
            'is_default': None
        })
        auth_group.jv_check_permission(PermissionTypes.EDIT.value, self.user)
        auth_group.save()
        
        if permissions := self.data.get('permissions'):
            auth_group.jv_check_can_update_permissions(self.user)
            auth_group.permissions.clear()
            for permission in permissions:
                if permission['is_permitted']:
                    auth_group.permissions.add(permission['id'])
        
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'{auth_group.name} group saved'
        })
    
    @atomic
    def delete(self, request, auth_group_id):
        auth_group = EmployerAuthGroup.objects.get(id=auth_group_id)
        auth_group.jv_check_permission(PermissionTypes.DELETE.value, self.user)
        auth_group.delete()
        
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'{auth_group.name} group deleted'
        })
    
    @staticmethod
    def get_auth_groups(user, employer_id):
        auth_group_filter = (
            Q(employer_id=employer_id) | Q(employer_id__isnull=True)
            & ~Q(user_type_bit__in=EmployerAuthGroupView.IGNORED_AUTH_GROUPS)
        )
        auth_groups = EmployerAuthGroup.objects.prefetch_related('permissions').filter(auth_group_filter)
        return EmployerAuthGroup.jv_filter_perm(user, auth_groups)


class EmployerUserView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerPermission]
    
    @atomic
    def post(self, request):
        user, is_new = UserView.get_or_create_user(self.user, self.data)
        employer_id = self.data['employer_id']
        employer = Employer.objects.get(id=employer_id)
        if not user.employer_id:
            user.employer_id = employer_id
            user.save()
        elif user.employer_id != employer_id:
            return get_error_response('This user already exists and cannot be created')
        
        new_user_groups = []
        permission_group_ids = self.data['permission_group_ids']
        for group_id in permission_group_ids:
            new_user_groups.append(
                UserEmployerPermissionGroup(
                    user=user,
                    employer_id=employer_id,
                    permission_group_id=group_id,
                    is_employer_approved=True
                )
            )
        UserEmployerPermissionGroup.objects.bulk_create(new_user_groups)
        
        user_type_bits = reduce(
            lambda a, b: a | b,
            EmployerAuthGroup.objects.filter(id__in=permission_group_ids).values_list('user_type_bit', flat=True),
            0
        )
        user.user_type_bits = user_type_bits
        user.save()
        
        base_django_data = {
            'user': user,
            'employer': employer,
            'admin_user': self.user,
            'is_exclude_final_message': False,
            'reset_password_url': user.get_reset_password_link(),
            'is_employer_owner': user.is_employer_owner
        }
        if employer.organization_type == Employer.ORG_TYPE_EMPLOYER:
            referral_link = SocialLinkView.get_or_create_employee_referral_links([user], employer)[0]
            job_referral_url = referral_link.get_link_url()
            send_django_email(
                'Welcome to JobVyne!',
                'emails/employer_user_welcome_email.html',
                to_email=user.email,
                django_context={
                    'job_referral_url': job_referral_url,
                    **base_django_data
                },
                employer=employer,
                is_tracked=False
            )
        elif employer.organization_type == Employer.ORG_TYPE_GROUP:
            send_django_email(
                'Welcome to JobVyne!',
                'emails/group_user_welcome_email.html',
                to_email=user.email,
                django_context={
                    **base_django_data,
                    'job_board_url': employer.main_job_board_link
                },
                employer=employer,
                is_tracked=False
            )
        
        success_message = f'Account created for {user.full_name}' if is_new else f'Account already exists for {user.full_name}. Permissions were updated.'
        
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: success_message
        })
    
    @atomic
    def put(self, request):
        users = UserView.get_user(self.user, user_filter=Q(id__in=self.data['user_ids']))
        batchCount = 0
        
        def get_unique_permission_key(p):
            return p.user_id, p.employer_id, p.permission_group_id
        
        if len(users) == 1 and self.user.is_admin and (password := self.data.get('password')):
            user = users[0]
            try:
                validate_password(password, user=user)
            except ValidationError as e:
                return get_error_response(f'Password doesn\'t meet requirements: {e}')
            user.set_password(password)
            user.save()
        
        while batchCount < len(users):
            user_employer_permissions_to_delete_filters = []
            user_employer_permissions_to_add = []
            user_employer_permissions_to_update = []
            batched_users = users[batchCount:batchCount + BATCH_UPDATE_SIZE]
            for user in batched_users:
                set_object_attributes(user, self.data, {
                    'first_name': None,
                    'last_name': None,
                    'employer_id': None,
                    'profession_id': None
                })
                user.jv_check_permission(PermissionTypes.EDIT.value, self.user)
                current_user_permissions = {
                    get_unique_permission_key(p): p for p in user.employer_permission_group.all()
                }
                
                if permission_group_ids := self.data.get('permission_group_ids'):
                    user_employer_permissions_to_delete_filters.append(
                        Q(user_id=user.id) & Q(employer_id=self.data['employer_id']))
                    for group_id in permission_group_ids:
                        user_employer_permissions_to_add.append(UserEmployerPermissionGroup(
                            user=user,
                            employer_id=self.data['employer_id'],
                            permission_group_id=group_id,
                            is_employer_approved=True
                        ))
                
                if add_permission_group_ids := self.data.get('add_permission_group_ids'):
                    for group_id in add_permission_group_ids:
                        permission_group = UserEmployerPermissionGroup(
                            user=user,
                            employer_id=self.data['employer_id'],
                            permission_group_id=group_id,
                            is_employer_approved=True
                        )
                        if existing_permission := current_user_permissions.get(
                                get_unique_permission_key(permission_group)):
                            existing_permission.is_employer_approved = True
                            user_employer_permissions_to_update.append(existing_permission)
                        else:
                            user_employer_permissions_to_add.append(permission_group)
                
                if remove_permission_group_ids := self.data.get('remove_permission_group_ids'):
                    for group_id in remove_permission_group_ids:
                        user_employer_permissions_to_delete_filters.append(
                            Q(user_id=user.id) & Q(permission_group_id=group_id))
            
            JobVyneUser.objects.bulk_update(users, ['first_name', 'last_name', 'employer_id', 'profession_id'])
            if user_employer_permissions_to_delete_filters:
                def reduceFilters(allFilters, filter):
                    allFilters |= filter
                    return allFilters
                
                delete_filter = reduce(reduceFilters, user_employer_permissions_to_delete_filters)
                UserEmployerPermissionGroup.objects.filter(delete_filter).delete()
            UserEmployerPermissionGroup.objects.bulk_create(user_employer_permissions_to_add)
            UserEmployerPermissionGroup.objects.bulk_update(user_employer_permissions_to_update,
                                                            ['is_employer_approved'])
            
            # Update user types based on new permission groups
            users_to_update = []
            for user in UserView.get_user(self.user, user_filter=Q(id__in=[u.id for u in batched_users])):
                user_type_bits = reduce(
                    lambda a, b: a | b,
                    [pg.permission_group.user_type_bit for pg in user.employer_permission_group.all()],
                    0
                )
                # Don't remove any user type groups that are already set
                # Users can set their own user types prior to having the appropriate permission groups
                user.user_type_bits = user.user_type_bits | user_type_bits
                users_to_update.append(user)
            JobVyneUser.objects.bulk_update(users_to_update, ['user_type_bits'])
            
            batchCount += BATCH_UPDATE_SIZE
            
        if len(users) == 1 and self.user.is_admin and ('is_employer_owner' in self.data):
            user = users[0]
            user.is_employer_owner = self.data['is_employer_owner']
            user.save()
            
        user_count = len(users)
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'{user_count} {"user" if user_count == 1 else "users"} updated'
        })
    
    @atomic
    def delete(self, request):
        if not self.user.is_admin:
            return Response('You do not have permission to delete this user', status=status.HTTP_401_UNAUTHORIZED)
        
        users = JobVyneUser.objects.filter(id__in=self.data.get('user_ids'))
        user_count = len(users)
        users.delete()
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'{user_count} {"user" if user_count == 1 else "users"} deleted'
        })


class EmployerUserApproveView(JobVyneAPIView):
    
    @atomic
    def put(self, request):
        """Set unapproved permission groups to approved for selected users
        """
        users = UserView.get_user(self.user, user_filter=Q(id__in=self.data['user_ids']))
        groups_to_update = []
        for user in users:
            user.jv_check_permission(PermissionTypes.EDIT.value, self.user)
            for group in user.employer_permission_group.filter(is_employer_approved=False):
                group.is_employer_approved = True
                groups_to_update.append(group)
        
        UserEmployerPermissionGroup.objects.bulk_update(groups_to_update, ['is_employer_approved'])
        userCount = len(users)
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'Permissions approved for {userCount} {"user" if userCount == 1 else "users"}'
        })


class EmployerUserActivateView(JobVyneAPIView):
    
    @atomic
    def put(self, request):
        if not (employer_id := self.data.get('employer_id')):
            return Response('An employer ID is required', status=status.HTTP_400_BAD_REQUEST)
        is_deactivate = self.data.get('is_deactivate')
        # Assign employee a seat if explicitly set or employee is activated
        is_assign_seat = self.data.get('is_assign') if is_deactivate is None else not is_deactivate
        
        employer = Employer.objects.prefetch_related('subscription').get(id=employer_id)
        users = UserView.get_user(self.user, user_filter=Q(id__in=self.data['user_ids']))
        subscription = EmployerSubscriptionView.get_subscription(employer)
        active_users_count = EmployerSubscriptionView.get_active_employees(employer)
        unassigned_users = 0
        for user in users:
            user.jv_check_permission(PermissionTypes.EDIT.value, self.user)
            if is_deactivate is not None:
                user.is_employer_deactivated = is_deactivate
            
            is_add_seat = is_assign_seat and (active_users_count < subscription.employee_seats)
            user.has_employee_seat = is_add_seat
            if is_add_seat:
                active_users_count += 1
            elif is_assign_seat:
                # If the employer has run out of employee seats we need to warn them
                unassigned_users += 1
        
        update_values = ['has_employee_seat']
        if is_deactivate is not None:
            update_values.append('is_employer_deactivated')
        JobVyneUser.objects.bulk_update(users, update_values)
        userCount = len(users)
        msg = f'{userCount} {"user" if userCount == 1 else "users"}'
        if is_deactivate is not None:
            msg += f' {"deactivated" if is_deactivate else "activated"}'
        else:
            msg += f' {"assigned" if is_assign_seat else "un-assigned"} a seat'
        
        data = {SUCCESS_MESSAGE_KEY: msg}
        if unassigned_users:
            data[WARNING_MESSAGES_KEY] = [
                f'{unassigned_users} {"user" if unassigned_users == 1 else "users"} was unable to be assigned a seat because you have reached the number of seats allowed by your subscription']
        
        return Response(status=status.HTTP_200_OK, data=data)


class EmployerUserUploadView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerPermission]

    @atomic
    def post(self, request):
        if not (employer_id := self.data.get('employer_id')):
            return get_error_response('An employer ID is required')
        
        employer = EmployerView.get_employers(employer_id=employer_id)
        raw_file = self.files['user_file'][0]
        csv_text = raw_file.read().decode('utf-8')
        with StringIO(csv_text) as csv_file:
            csv.bulk_load_users(csv_file, employer)
        return Response(status=status.HTTP_200_OK)


class EmployerFileView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerOrReadOnlyPermission]
    
    def get(self, request):
        if not (employer_id := self.query_params.get('employer_id')):
            return Response('An employer ID is required', status=status.HTTP_400_BAD_REQUEST)
        
        files = self.get_employer_files(employer_id=employer_id)
        return Response(status=status.HTTP_200_OK, data=[get_serialized_employer_file(f) for f in files])
    
    @atomic
    def post(self, request):
        employer_file = EmployerFile()
        file = self.files['file'][0] if self.files.get('file') else None
        self.update_employer_file(employer_file, self.data, self.user, file=file)
        return Response(status=status.HTTP_200_OK, data={
            'id': employer_file.id,
            SUCCESS_MESSAGE_KEY: f'Created a new file titled {employer_file.title}'
        })
    
    @atomic
    def put(self, request, file_id):
        employer_file = self.get_employer_files(file_id=file_id)
        self.update_employer_file(employer_file, self.data, self.user)
        return Response(status=status.HTTP_200_OK, data={
            'id': employer_file.id,
            SUCCESS_MESSAGE_KEY: f'Updated file titled {employer_file.title}'
        })
    
    @staticmethod
    @atomic
    def update_employer_file(employer_file, data, user, file=None):
        set_object_attributes(employer_file, data, {
            'employer_id': None,
            'title': None
        })
        
        if file:
            employer_file.file = file
        
        employer_file.title = (
                employer_file.title
                or getattr(file, 'name', None)
                or employer_file.file.name.split('/')[-1]
        )
        
        permission_type = PermissionTypes.EDIT.value if employer_file.id else PermissionTypes.CREATE.value
        employer_file.jv_check_permission(permission_type, user)
        employer_file.save()
        
        employer_file.tags.clear()
        for tag in data.get('tags') or []:
            if isinstance(tag, str):
                tag = EmployerFileTagView.get_or_create_tag(tag, data['employer_id'])
                employer_file.tags.add(tag)
            else:
                employer_file.tags.add(tag['id'])
    
    @staticmethod
    def get_employer_files(file_id=None, employer_id=None, file_filter=None):
        file_filter = file_filter or Q()
        if file_id:
            file_filter &= Q(id=file_id)
        if employer_id:
            file_filter &= Q(employer_id=employer_id)
        
        files = EmployerFile.objects.prefetch_related('tags').filter(file_filter)
        if file_id:
            if not files:
                raise EmployerFile.DoesNotExist
            return files[0]
        
        return files


class EmployerFileTagView(JobVyneAPIView):
    
    def get(self, request):
        if not (employer_id := self.query_params['employer_id']):
            return Response('An employer ID is required', status=status.HTTP_400_BAD_REQUEST)
        
        tags = self.get_employer_file_tags(employer_id)
        return Response(status=status.HTTP_200_OK, data=[get_serialized_employer_file_tag(t) for t in tags])
    
    @atomic
    def delete(self, request, tag_id):
        tag = EmployerFileTag.objects.get(id=tag_id)
        tag.jv_check_permission(PermissionTypes.DELETE.value, self.user)
        tag.delete()
        return Response(status=status.HTTP_200_OK, data={
            SUCCESS_MESSAGE_KEY: f'{tag.name} tag was deleted'
        })
    
    @staticmethod
    @atomic
    def get_or_create_tag(tag_name, employer_id):
        try:
            return EmployerFileTag.objects.get(name=tag_name, employer_id=employer_id)
        except EmployerFileTag.DoesNotExist:
            tag = EmployerFileTag(name=tag_name, employer_id=employer_id)
            tag.save()
            return tag
    
    @staticmethod
    def get_employer_file_tags(employer_id):
        return EmployerFileTag.objects.filter(employer_id=employer_id)


class EmployerFromDomainView(JobVyneAPIView):
    
    def get(self, request):
        if not (email := self.query_params.get('email')):
            return get_error_response('An email address is required')
        
        try:
            employers = self.get_employers_from_email(email)
        except ValueError:
            return get_error_response(f'Could not parse email domain for {email}')
        
        return Response(status=status.HTTP_200_OK, data=employers)
    
    @staticmethod
    def get_employers_from_email(email):
        if not (email_domain := get_domain_from_email(email)):
            raise ValueError()
        
        employers = [(e.email_domains, e) for e in Employer.objects.all()]
        matched_employers = []
        for domains, employer in employers:
            if not domains:
                continue
            if email_domain in domains:
                matched_employers.append({'id': employer.id, 'name': employer.employer_name})
        
        return matched_employers
        


class EmployerJobDepartmentView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerOrReadOnlyPermission]
    
    def get(self, request):
        if not (employer_id := self.query_params.get('employer_id')):
            return Response('An employer ID is required', status=status.HTTP_400_BAD_REQUEST)
        
        job_subscriptions = JobSubscriptionView.get_job_subscriptions(employer_id=employer_id)
        job_subscription_filter = JobSubscriptionView.get_combined_job_subscription_filter(job_subscriptions)
        # Include both subscribed jobs and those owned by the employer
        if job_subscription_filter:
            job_subscription_filter |= Q(employer_id=employer_id)
        else:
            job_subscription_filter = Q(employer_id=employer_id)
        departments = EmployerJobView \
            .get_employer_jobs(employer_job_filter=job_subscription_filter) \
            .values('job_department_id', 'job_department__name') \
            .distinct()  # This will only get distinct jobs. Adding distinct value criteria is not supported by MySQL
        
        # Get unique departments
        departments = {(d['job_department_id'], d['job_department__name']): d for d in departments}.values()
        
        return Response(status=status.HTTP_200_OK, data=[{
            'id': dept['job_department_id'],
            'name': dept['job_department__name']
        } for dept in departments])


class EmployerJobLocationView(JobVyneAPIView):
    permission_classes = [IsAdminOrEmployerOrReadOnlyPermission]
    
    def get(self, request):
        if not (employer_id := self.query_params.get('employer_id')):
            return Response('An employer ID is required', status=status.HTTP_400_BAD_REQUEST)
        
        job_subscriptions = JobSubscriptionView.get_job_subscriptions(employer_id=employer_id)
        job_subscription_filter = JobSubscriptionView.get_combined_job_subscription_filter(job_subscriptions)
        # Include both subscribed jobs and those owned by the employer
        if job_subscription_filter:
            job_subscription_filter |= Q(employer_id=employer_id)
        else:
            job_subscription_filter = Q(employer_id=employer_id)
        jobs = (
            EmployerJob.objects
            .prefetch_related('locations', 'locations__city', 'locations__state', 'locations__country')
            .filter(job_subscription_filter)
            .only('id')
        )
        locations = []
        for job in jobs:
            for location in job.locations.all():
                locations.append(location)
        
        return Response(status=status.HTTP_200_OK, data=LocationView.get_serialized_locations(locations))


class EmployerInfoView(JobVyneAPIView):
    CONCURRENT_REQUESTS = 10
    MIN_AI_YEAR = 2020
    DESCRIPTION_PROMPT = (
        'You are helping describe a company. The user will provide a company name and a list of website domains used by the company. You should provide a'
        '(description) which is limited to 1 sentence and a (description_long) which is limited to 5 sentences.\n'
        'Both the (description) and (description_long) should begin with "(name of company) is a" and proceed with the description.\n'
        'Your response should be RFC8259 compliant JSON in the format:\n'
        '{"name": "(name of company)", "description": "(description)", "description_long": "(description_long)"}'
    )

    @staticmethod
    def fill_employer_description(limit=None, employer_filter=None):
        employer_filter = employer_filter or (
            (Q(description__isnull=True) | Q(description_long__isnull=True))
            & Q(organization_type=Employer.ORG_TYPE_EMPLOYER)
        )
        employer_filter &= Q(email_domains__isnull=False)
        undescribed_employers = Employer.objects.filter(employer_filter)
        if limit:
            undescribed_employers = undescribed_employers[:limit]
        employer_idx = 0
        while employer_idx < len(undescribed_employers):
            employers_to_process = undescribed_employers[employer_idx:employer_idx + EmployerInfoView.CONCURRENT_REQUESTS]
            asyncio.run(
                EmployerInfoView.process_employers(employers_to_process)
            )
    
            Employer.objects.bulk_update(
                employers_to_process, ['description', 'description_long']
            )
            employer_idx += EmployerInfoView.CONCURRENT_REQUESTS

    @staticmethod
    async def summarize_employer(queue):
        while True:
            employer = await queue.get()
            logger.info(f'Running employer description for {employer.employer_name}')
            try:
                website_domains = employer.email_domains.split(',')[0] if employer.email_domains else None
                resp, tracker = await ai.ask([
                    {'role': 'system', 'content': EmployerInfoView.DESCRIPTION_PROMPT},
                    {'role': 'user', 'content': f'Company: {employer.employer_name}\nWebsite domains: {website_domains}'}
                ], is_test=False)
                employer.description = resp.get('description')
                employer.description_long = resp.get('description_long')
            except (PromptError, RateLimitError):
                pass
        
            queue.task_done()

    @staticmethod
    async def process_employers(employers):
        queue = asyncio.Queue()
        workers = [
            asyncio.create_task(EmployerInfoView.summarize_employer(queue))
            for _ in range(EmployerInfoView.CONCURRENT_REQUESTS)
        ]
    
        for employer in employers:
            await queue.put(employer)
    
        # Wait until the queue is fully processed
        if not queue.empty():
            logger.info(f'Waiting for job queue to finish - Currently {queue.qsize()}')
            done, _ = await asyncio.wait([queue.join(), *workers], return_when=asyncio.FIRST_COMPLETED)
            consumers_raised = set(done) & set(workers)
            if consumers_raised:
                logger.info(f'Found {len(consumers_raised)} consumers that raised exceptions')
                await consumers_raised.pop()  # propagate the exception
    
        for worker in workers:
            worker.cancel()
    
        # Wait until all workers are cancelled
        await asyncio.gather(*workers, return_exceptions=True)
        logger.info(f'Completed summarizing {len(employers)} employers')
