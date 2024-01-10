import logging
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Optional

import sentry_sdk
from bs_config import Env
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from rate_limiter import (
    RateLimiter,
    RateLimitingPolicy,
    RateLimitingRepo,
    Usage,
    policy,
    repo,
)
from zoneinfo import ZoneInfo

from horoscopebot.bot import Bot
from horoscopebot.config import (
    Config,
    EventPublisherConfig,
    HoroscopeConfig,
    HoroscopeMode,
    RateLimitConfig,
)
from horoscopebot.event.publisher import EventPublisher
from horoscopebot.event.pubsub import PubSubEventPublisher
from horoscopebot.event.stub import StubEventPublisher
from horoscopebot.horoscope.horoscope import Horoscope
from horoscopebot.horoscope.openai_chat import OpenAiChatHoroscope
from horoscopebot.horoscope.steffen import SteffenHoroscope
from horoscopebot.tracing import setup_tracing

_LOG = logging.getLogger(__package__)


def _setup_logging():
    LoggingInstrumentor().instrument(set_logging_format=True)
    _LOG.level = logging.INFO


def _setup_sentry(dsn: Optional[str], release: str):
    if not dsn:
        _LOG.warning("Sentry DSN not found")
        return

    sentry_sdk.init(
        dsn=dsn,
        release=release,
    )


def _load_horoscope(config: HoroscopeConfig) -> Horoscope:
    if config.mode == HoroscopeMode.Steffen:
        return SteffenHoroscope()
    elif config.mode == HoroscopeMode.OpenAiChat:
        return OpenAiChatHoroscope(config.openai)  # type:ignore
    else:
        raise ValueError()


def _load_event_publisher(config: EventPublisherConfig) -> EventPublisher:
    if config.mode == "stub":
        _LOG.warning("Using stub event publisher")
        return StubEventPublisher()
    elif config.mode == "pubsub":
        return PubSubEventPublisher(config)
    else:
        raise ValueError(f"Unknown mode {config.mode}")


class _StubRateLimitPolicy(RateLimitingPolicy):
    @property
    def requested_history(self) -> int:
        return 0

    def get_offending_usage(
        self,
        at_time: datetime,
        last_usages: list[Usage],
    ) -> Usage | None:
        return None


def _load_rate_limiter(timezone: tzinfo, config: RateLimitConfig) -> RateLimiter:
    match config.rate_limiter_type:
        case "stub":
            return RateLimiter(
                policy=_StubRateLimitPolicy(),
                repo=repo.InMemoryRateLimitingRepo(),
            )

    rate_limit_file = config.rate_limit_file
    repository: RateLimitingRepo

    if rate_limit_file is None:
        _LOG.warning("Using in-memory rate limiting repo")
        repository = repo.InMemoryRateLimitingRepo()
    else:
        repository = repo.SqliteRateLimitingRepo.connect(Path(rate_limit_file))

    return RateLimiter(
        policy=policy.DailyLimitRateLimitingPolicy(limit=1),
        repo=repository,
        timezone=timezone,
    )


def main():
    _setup_logging()

    config = Config.from_env(Env.load(include_default_dotenv=True))
    _setup_sentry(config.sentry_dsn, release=config.app_version)
    setup_tracing(config)

    timezone = ZoneInfo("Europe/Berlin")

    horoscope = _load_horoscope(config.horoscope)
    event_publisher = _load_event_publisher(config.event_publisher)
    rate_limiter = _load_rate_limiter(timezone, config.rate_limit)

    bot = Bot(
        config.telegram,
        horoscope=horoscope,
        event_publisher=event_publisher,
        rate_limiter=rate_limiter,
        timezone=timezone,
    )
    _LOG.info("Launching bot...")
    bot.run()


if __name__ == "__main__":
    main()
