# Copyright (c) 2022, 2023, Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl.

from decimal import Decimal
from typing import Any, AsyncGenerator, List, cast

import pytest
import pytest_asyncio

import tests
from coherence import Aggregators, Filters, NamedCache, Session
from coherence.aggregator import (
    EntryAggregator,
    PriorityAggregator,
    RecordType,
    ReducerResult,
    Schedule,
    ScriptAggregator,
    Timeout,
    TopAggregator,
)
from coherence.serialization import JSONSerializer
from tests.person import Person


@pytest_asyncio.fixture
async def setup_and_teardown() -> AsyncGenerator[NamedCache[Any, Any], None]:
    session: Session = await tests.get_session()
    cache: NamedCache[Any, Any] = await session.get_cache("test")

    await Person.populate_named_map(cache)

    yield cache

    await cache.clear()
    await cache.destroy()
    await session.close()


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_max(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    ag: EntryAggregator[int] = Aggregators.max("age")
    r: int = await cache.aggregate(ag)
    assert r == Person.pat().age


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_min(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    ag: EntryAggregator[int] = Aggregators.min("age")
    r: int = await cache.aggregate(ag)
    assert r == Person.alice().age


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_sum(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    ag = Aggregators.sum("age")
    r = await cache.aggregate(ag)
    assert r == (
        Person.andy().age
        + Person.alice().age
        + Person.pat().age
        + Person.paula().age
        + Person.fred().age
        + Person.fiona().age
        + Person.jim().age
    )


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_average(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    ag = Aggregators.average("age")
    r: Decimal = await cache.aggregate(ag)
    assert float(r) == round(
        (
            Person.andy().age
            + Person.alice().age
            + Person.pat().age
            + Person.paula().age
            + Person.fred().age
            + Person.fiona().age
            + Person.jim().age
        )
        / 7,
        8,
    )


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_count(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    ag = Aggregators.count()
    r: int = await cache.aggregate(ag)
    assert r == 7


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_distinct_values(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    ag: EntryAggregator[List[str]] = Aggregators.distinct("gender")
    r: list[str] = await cache.aggregate(ag)
    assert sorted(r) == ["Female", "Male"]


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_top(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    ag: TopAggregator[int, Person] = Aggregators.top(2).order_by("age").ascending
    r: list[Person] = await cache.aggregate(ag)
    assert r == [Person.alice(), Person.andy()]

    ag = Aggregators.top(2).order_by("age").ascending
    r = await cache.aggregate(ag, None, Filters.between("age", 30, 40))
    assert r == [Person.paula(), Person.jim()]

    ag = Aggregators.top(2).order_by("age").descending
    r = await cache.aggregate(ag)
    assert r == [Person.pat(), Person.fred()]

    ag = Aggregators.top(2).order_by("age").descending
    r = await cache.aggregate(ag, None, Filters.between("age", 20, 30))
    assert r == [Person.fiona(), Person.andy()]


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_group(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    ag: EntryAggregator[dict[str, int]] = Aggregators.group_by("gender", Aggregators.min("age"), Filters.always())

    r: dict[str, int] = await cache.aggregate(ag)
    print("\n" + str(r))
    assert r == {"Male": 25, "Female": 22}

    f = Filters.between("age", 20, 24)
    r = await cache.aggregate(ag, None, f)
    print("\n" + str(r))
    assert r == {"Female": 22}

    r = await cache.aggregate(ag, {"Pat", "Paula", "Fred"})
    print("\n" + str(r))
    assert r == {"Male": 58, "Female": 35}


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_priority(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    agg: EntryAggregator[Decimal] = Aggregators.priority(Aggregators.sum("age"))
    agg_actual: PriorityAggregator[Decimal] = cast(PriorityAggregator[Decimal], agg)
    assert agg_actual.execution_timeout_in_millis == Timeout.DEFAULT
    assert agg_actual.request_timeout_in_millis == Timeout.DEFAULT
    assert agg_actual.scheduling_priority == Schedule.STANDARD

    r = await cache.aggregate(agg)
    assert r == (
        Person.andy().age
        + Person.alice().age
        + Person.pat().age
        + Person.paula().age
        + Person.fred().age
        + Person.fiona().age
        + Person.jim().age
    )

    agg2: EntryAggregator[Decimal] = Aggregators.priority(
        Aggregators.sum("age"),
        execution_timeout=Timeout.NONE,
        request_timeout=Timeout.NONE,
        scheduling_priority=Schedule.IMMEDIATE,
    )
    agg2_actual: PriorityAggregator[Decimal] = cast(PriorityAggregator[Decimal], agg2)
    assert agg2_actual.execution_timeout_in_millis == Timeout.NONE
    assert agg2_actual.request_timeout_in_millis == Timeout.NONE
    assert agg2_actual.scheduling_priority == Schedule.IMMEDIATE

    eq_filter = Filters.equals("gender", "Male")
    r = await cache.aggregate(agg, None, eq_filter)
    assert r == (Person.andy().age + Person.pat().age + Person.fred().age + Person.jim().age)

    r = await cache.aggregate(agg, {"Alice", "Paula", "Fiona"})
    assert r == (Person.alice().age + Person.paula().age + Person.fiona().age)


# noinspection PyShadowingNames
def test_script() -> None:
    agg: EntryAggregator[Any] = Aggregators.script("py", "test_script.py", 0, "abc", 2, 4.0)
    serializer = JSONSerializer()
    j = serializer.serialize(agg)

    script_aggregator: ScriptAggregator[Any] = serializer.deserialize(j)
    assert script_aggregator.name == "test_script.py"
    assert script_aggregator.language == "py"
    assert script_aggregator.args == ["abc", 2, 4.0]
    assert script_aggregator.characteristics == 0


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_query_recorder(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    agg = Aggregators.record()
    f = Filters.between("age", 20, 30)
    my_result: dict[str, Any | list[Any]] = await cache.aggregate(agg, None, f)
    assert my_result.get("results") is not None
    my_list: Any | list[Any] = my_result.get("results")
    assert len(my_list) == 1
    assert my_list[0].get("partitionSet") is not None
    assert my_list[0].get("steps") is not None

    agg = Aggregators.record(RecordType.TRACE)
    f = Filters.between("age", 20, 30)
    my_result = await cache.aggregate(agg, None, f)
    assert my_result.get("results") is not None
    my_list = my_result.get("results")
    assert len(my_list) == 1  # type: ignore
    assert my_list[0].get("partitionSet") is not None  # type: ignore
    assert my_list[0].get("steps") is not None  # type: ignore


# noinspection PyShadowingNames
@pytest.mark.asyncio
async def test_reducer(setup_and_teardown: NamedCache[Any, Any]) -> None:
    cache: NamedCache[str, Person] = setup_and_teardown

    agg: EntryAggregator[ReducerResult[str]] = Aggregators.reduce("age")
    f = Filters.between("age", 20, 30)
    my_result: ReducerResult[str] = await cache.aggregate(agg, None, f)
    print("\n" + str(my_result))
    assert my_result == {"Andy": 25, "Fiona": 29, "Alice": 22}

    my_result = await cache.aggregate(agg, {"Andy", "Fiona", "Alice"})
    print("\n" + str(my_result))
    assert my_result == {"Andy": 25, "Alice": 22, "Fiona": 29}
