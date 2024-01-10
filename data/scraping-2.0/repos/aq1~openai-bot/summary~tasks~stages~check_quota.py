from django.db import models
from django.db.models import functions
from django.utils import timezone
from django.utils.translation import gettext as _, activate

from .stage import Stage
from .exceptions import StopPipeline

from ...models import OpenAICall


class CheckQuota(Stage):
    def __init__(self, user_id: int, quota: int):
        self.user_id = user_id
        self.quota = quota

    async def __call__(self, data):
        tokens = (await OpenAICall.objects.filter(
            user_id=self.user_id,
            created_at__gte=timezone.now().replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            ),
        ).aaggregate(
            s=functions.Coalesce(
                models.Sum('tokens'),
                models.Value(0),
            )
        ))['s']

        if tokens >= self.quota:
            raise StopPipeline(_('😢 You exceeded 24-hour quota. Try again later.'))

        return data
