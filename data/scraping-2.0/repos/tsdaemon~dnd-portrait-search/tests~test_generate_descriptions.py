import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from portrait_search.data_sources.base import BaseDataSource
from portrait_search.dependencies import Container
from portrait_search.generate_descriptions import generate_descriptions
from portrait_search.open_ai.client import OpenAIClient
from portrait_search.open_ai.queries import PORTRAIT_DESCRIPTION_QUERY_V1
from portrait_search.portraits.entities import Portrait
from portrait_search.portraits.repository import PortraitRepository


@pytest.fixture
def portraits_repository_mock() -> Mock:
    return Mock(spec=PortraitRepository)


@pytest.fixture
def data_source_mock() -> Mock:
    return Mock(spec=BaseDataSource)


@pytest.fixture
def nexus_mods_fixture(fixtures_path: Path) -> Generator[Path, Any, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        dst = Path(temp_dir) / "nexus"
        shutil.copytree(fixtures_path / "nexus/test_mod", dst)
        yield dst


@pytest.fixture
def nexus_mods_all_portraits_fixture(nexus_mods_fixture: Path) -> list[Portrait]:
    root_folder = nexus_mods_fixture
    return [
        Portrait(
            fulllength_path=Path(root_folder / "some race/some type/AA-WI-MS-F107/Fulllength.png"),
            medium_path=Path(root_folder / "some race/some type/AA-WI-MS-F107/Medium.png"),
            small_path=Path(root_folder / "some race/some type/AA-WI-MS-F107/Small.png"),
            base_path=nexus_mods_fixture,
            tags=[
                "some race",
                "some type",
                "AA-WI-MS-F107",
            ],
            url="https://www.nexusmods.com/pathfinderkingmaker/mods/9",
        ),
        Portrait(
            fulllength_path=Path(root_folder / "some race/some type/AA-WI-MS-F107 copy/Fulllength.png"),
            medium_path=Path(root_folder / "some race/some type/AA-WI-MS-F107 copy/Medium.png"),
            small_path=Path(root_folder / "some race/some type/AA-WI-MS-F107 copy/Small.png"),
            base_path=nexus_mods_fixture,
            tags=[
                "some race",
                "some type",
                "AA-WI-MS-F107 copy",
            ],
            url="https://www.nexusmods.com/pathfinderkingmaker/mods/9",
        ),
        Portrait(
            fulllength_path=Path(root_folder / "another race/MX-IN-TL-F201/Fulllength.png"),
            medium_path=Path(root_folder / "another race/MX-IN-TL-F201/Medium.png"),
            small_path=Path(root_folder / "another race/MX-IN-TL-F201/Small.png"),
            base_path=nexus_mods_fixture,
            tags=[
                "another race",
                "MX-IN-TL-F201",
            ],
            url="https://www.nexusmods.com/pathfinderkingmaker/mods/9",
        ),
    ]


@pytest.fixture
def openai_client_mock() -> Mock:
    m = Mock(spec=OpenAIClient)
    m.make_image_query.return_value = "openai image description"
    return m


@pytest.fixture(autouse=True)
def container(
    container: Container, portraits_repository_mock: Mock, data_source_mock: Mock, openai_client_mock: Mock
) -> Generator[None, Any, Any]:
    with (
        container.portrait_repository.override(portraits_repository_mock),
        container.data_sources.override([data_source_mock]),
        container.openai_client.override(openai_client_mock),
    ):
        container.wire(modules=["portrait_search.generate_descriptions"])
        yield


@pytest.mark.asyncio
async def test_generate_descriptions__no_data(
    data_source_mock: Mock, portraits_repository_mock: Mock, openai_client_mock: Mock
) -> None:
    # GIVEN empty data source response
    data_source_mock.retrieve.return_value = []
    # WHEN generate_descriptions is called
    await generate_descriptions()
    # THEN openai is not called
    openai_client_mock.make_image_query.assert_not_called()
    # THEN no portraits are saved
    portraits_repository_mock.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_generate_descriptions__all_new_portraits(
    nexus_mods_all_portraits_fixture: list[Portrait],
    data_source_mock: Mock,
    portraits_repository_mock: Mock,
    openai_client_mock: Mock,
) -> None:
    # GIVEN data source with images, and some images has duplicates
    data_source_mock.retrieve.return_value = nexus_mods_all_portraits_fixture
    assert len(nexus_mods_all_portraits_fixture) == 3
    # GIVEN no existing portraits
    portraits_repository_mock.get_distinct_hashes.return_value = set()
    # WHEN generate_descriptions is called
    await generate_descriptions()
    # THEN openai is called to get a description for each image
    assert openai_client_mock.make_image_query.call_count == 2
    # THEN new portraits are saved
    portrait_record1 = nexus_mods_all_portraits_fixture[1].to_record()
    portrait_record1.description = "openai image description"
    portrait_record1.query = PORTRAIT_DESCRIPTION_QUERY_V1
    portraits_repository_mock.insert_one.assert_any_call(portrait_record1)
    portrait_record2 = nexus_mods_all_portraits_fixture[2].to_record()
    portrait_record2.description = "openai image description"
    portrait_record2.query = PORTRAIT_DESCRIPTION_QUERY_V1
    portraits_repository_mock.insert_one.assert_any_call(portrait_record1)
    assert portraits_repository_mock.insert_one.call_count == 2


@pytest.mark.asyncio
async def test_generate_descriptions__one_new_portrait(
    nexus_mods_all_portraits_fixture: list[Portrait],
    data_source_mock: Mock,
    portraits_repository_mock: Mock,
    openai_client_mock: Mock,
) -> None:
    # GIVEN data source with images, and some images has duplicates
    data_source_mock.retrieve.return_value = nexus_mods_all_portraits_fixture
    assert len(nexus_mods_all_portraits_fixture) == 3
    # GIVEN one existing portrait
    portraits_repository_mock.get_distinct_hashes.return_value = {"ff2f2f000060301c"}
    # WHEN generate_descriptions is called
    await generate_descriptions()
    # THEN openai is called to get a description for one image
    assert openai_client_mock.make_image_query.call_count == 1
    # THEN one new portrait is saved
    portrait_record1 = nexus_mods_all_portraits_fixture[1].to_record()
    portrait_record1.description = "openai image description"
    portrait_record1.query = PORTRAIT_DESCRIPTION_QUERY_V1
    portraits_repository_mock.insert_one.assert_any_call(portrait_record1)
    assert portraits_repository_mock.insert_one.call_count == 1


@pytest.mark.asyncio
async def test_generate_descriptions__no_new_portrait(
    nexus_mods_all_portraits_fixture: list[Portrait],
    data_source_mock: Mock,
    portraits_repository_mock: Mock,
    openai_client_mock: Mock,
) -> None:
    # GIVEN data source with images, and some images has duplicates
    data_source_mock.retrieve.return_value = nexus_mods_all_portraits_fixture
    assert len(nexus_mods_all_portraits_fixture) == 3
    # GIVEN all portraits already stored
    portraits_repository_mock.get_distinct_hashes.return_value = {
        "ff2f2f000060301c",
        "006f3f31002ffaf0",
    }
    # WHEN generate_descriptions is called
    await generate_descriptions()
    # THEN openai is not called
    assert openai_client_mock.make_image_query.call_count == 0
    # THEN no portraits are saved
    assert portraits_repository_mock.insert_one.call_count == 0
