#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os import environ

import openai
import pytest
from github import Event, Github

from gm import GH_EVENTS, event_dict, github_event, hey_gpt

openai.api_key = environ["OPENAI_KEY"]


@pytest.fixture(scope="session")
def events(pytestconfig):
    token = pytestconfig.getoption("token")
    g = Github(token)
    u = g.get_user()
    e = [e for e in u.get_events().get_page(0) if e.type in GH_EVENTS]
    return e


def test_event_is_event_event_type(events):
    assert isinstance(events[0], Event.Event)


def test_event_has_repo(events):
    assert events[0].repo is not None


def test_event_has_actor(events):
    assert events[0].actor is not None


def test_event_has_payload(events):
    assert events[0].payload is not None


def test_event_has_created_at(events):
    assert events[0].created_at is not None


def test_event_has_id(events):
    assert events[0].id is not None


def test_gh_events_allowed():
    assert len(GH_EVENTS) == 6


def test_event_dict(events):
    for e in events:
        e_dict_keys = event_dict(e).keys()
        assert len(e_dict_keys) == 3
        assert list(e_dict_keys) == ["repo", "type", "created_at"]


def test_github_event_serial(events):
    for e in events:
        github_event(e)


@pytest.mark.slow
def test_hey_gpt(events):
    e_log = []
    for e in events:
        e_log.append(github_event(e))
    ai_out = hey_gpt(e_log)
    assert isinstance(ai_out, str)
    assert len(ai_out) > 0
