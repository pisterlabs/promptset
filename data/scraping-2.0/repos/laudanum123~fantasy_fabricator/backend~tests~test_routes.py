"""Tests for app.py"""
import json
from unittest.mock import patch, MagicMock
import pytest
import main.util.utilities as utilities
from flask_sqlalchemy import SQLAlchemy
from main.routes.routes import routes

# from pytest_flask_sqlalchemy import db_session
from main import db
from main.models import Adventures, AdventureNPCs, AdventureLocations


@pytest.fixture()
def test_adventures(app):
    with app.app_context():
        adventure0 = Adventures(
            adventure_title="The Lost Temple of Zalazar",
            adventure_hook="The adventurers are hired by a mysterious patron to explore an ancient ruin in the jungle.",
            adventure_plot="The ruin is actually the temple of Zalazar, a powerful lich who ruled the land centuries ago. The patron is actually Zalazar's servant, who wants to use the adventurers to activate a hidden portal that will unleash Zalazar's army of undead.",
            adventure_climax="The adventurers face Zalazar himself in his throne room, where he tries to persuade them to join him or die. The portal is activated, and hordes of zombies and skeletons pour in.",
            adventure_resolution="The adventurers can either fight Zalazar and his minions, or try to escape through the portal. If they defeat Zalazar, they can loot his treasure and learn his secrets. If they escape, they can warn the nearby settlements of the impending danger.",
            adventure_npcs="Zalazar, the lich; Raxus, the patron/servant; Lila, the guide; Turok, the tribal chief; Zara, the priestess.",
        )
        adventure1 = Adventures(
            adventure_title="The Dragon's Lair",
            adventure_hook="The adventurers are hired by a local lord to slay a dragon that has been terrorizing the nearby villages.",
            adventure_plot="The dragon lives in a cave in the mountains, guarded by various traps and monsters. The adventurers have to overcome these obstacles and reach the dragon's lair, where they discover that the dragon is actually a young and scared creature that was driven from its home by a more powerful dragon.",
            adventure_climax="The adventurers can either fight the dragon or try to reason with it. If they fight, they have to deal with the dragon's breath weapon, claws, and tail. If they reason, they have to persuade the dragon to leave peacefully or find a new home for it.",
            adventure_resolution="If the adventurers kill the dragon, they can claim its hoard and return to the lord for their reward. If they spare the dragon, they can either lie to the lord or tell the truth. If they lie, they risk being exposed by the dragon or the villagers. If they tell the truth, they may face the lord's wrath or gratitude, depending on his personality.",
            adventure_npcs="Draco, the dragon; Lord Balder, the lord; Sir Reginald, the knight; Nala, the druid; Grog, the orc.",
        )
        db.session.add_all([adventure0, adventure1])
        db.session.commit()


@patch("openai.Completion.create")
def test_generate_adventure(mock_create, app, client):
    """Tests generate_adventure route"""
    # GIVEN a mocked OpenAI completion API and a Flask test client

    # set up mock response
    mock_response = {
        "id": "cmpl-1q2w3e4r5t6y7u8i9o0p",
        "object": "text_completion",
        "created": 1620000000,
        "model": "davinci003",
        "choices": [
            {
                "text": """{
        "AdventureTitle": "The Lost Temple",
        "AdventureHook":
        "The players must find a way to stop an ancient evil from being unleashed.",
        "AdventurePlot":
        "The players are hired to infiltrate a rival guild and steal a powerful artifact",
        "AdventureClimax":
        "The players must defeat the guild leader in a final battle to stop the ancient evil from being unleashed.",
        "AdventureResolution":
        "The players succeed in defeating the guild leader and stopping the ancient evil.",
        "AdventureNPCs": "Guild leader, ancient evil"}""",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 2000,
            "completion_tokens": 2000,
            "total_tokens": 4000,
        },
    }

    mock_create.return_value = mock_response

    # WHEN a request is made to the generate_adventure route with test data
    test_data = {
        "adventureTitle": "The Lost Temple",
        "adventureSetting": "Medieval Fantasy",
        "adventurePlot": "The players must find a way to stop an ancient evil from being unleashed.",
    }

    # send request
    response = client.post(
        "/generate_adventure",
        data=json.dumps(test_data),
        content_type="application/json",
    )

    data = json.loads(response.get_data())

    # THEN the response should have a 201 status code and the expected message
    expected_keys = [
        "AdventureTitle",
        "AdventureHook",
        "AdventurePlot",
        "AdventureClimax",
        "AdventureResolution",
        "AdventureNPCs",
    ]

    # assert response
    assert response.status_code == 201
    assert data["message"] == utilities.clean_gpt_response(
        mock_response["choices"][0]["text"], expected_keys=expected_keys
    )


def test_get_adventures_from_db(test_adventures, client):
    """tests get_adventures_from_db route

    Args:
        client (_type_): _description_
    """
    # Given two test adventures in the database

    # When the get_adventures_from_db endpoint is accessed
    response = client.get("/get_adventures_from_db")
    # Then the response should have a status code of 200 and return two adventures
    assert response.status_code == 200
    assert len(json.loads(response.data)) == 2


def test_get_specific_adventures_from_db(test_adventures, client):
    """tests get_adventures_from_db route

    Args:
        client (_type_): _description_
    """
    # Given an adventure with id 1 in the database

    # When the get_adventures_from_db endpoint is accessed with id=1
    response = client.get("/get_adventures_from_db?id=1")

    # Then the response should have a status code of 200
    response = client.get("/get_adventures_from_db?id=1")
    assert response.status_code == 200
    assert len(json.loads(response.data)) > 0


# Test the extract_entities function with a valid id
@patch(
    "main.util.utilities.extract_entities_from_adventure",
    return_value=([["temple", "jungle"]], [["temple", "jungle"]]),
)
def test_extract_entities_valid_id(extract_entity, test_adventures, client):
    # Given an adventure with id 1 in the database

    # And the extract_entities_from_adventure utility function has been patched
    # to return the expected NPC and location names

    # When a POST request is sent to the /extract_entities/1 endpoint
    response = client.post("/extract_entities/1")

    # Then the response should have a status code of 201
    assert response.status_code == 201

    # And the response data should be a JSON object
    assert response.is_json

    # And the response data should have a "status" key with the value "success"
    data = json.loads(response.data)
    assert data["status"] == "success"

    # And the response data should have a "message" key containing a list of two lists
    assert isinstance(data["message"], list)
    assert len(data["message"]) == 2

    # And the first list in the "message" key should contain the expected NPC names
    assert data["message"][0] == [["temple", "jungle"]]

    # And the second list in the "message" key should contain the expected location names
    assert data["message"][1] == [["temple", "jungle"]]

    # Check the database has the expected NPC and location records


def test_delete_adventures_from_db(client):
    # Given an existing adventure
    adventure = Adventures(
        adventure_title="Test Adventure",
        adventure_hook="Test Hook",
        adventure_plot="Test Plot",
        adventure_climax="Test Climax",
        adventure_resolution="Test Resolution",
        adventure_npcs="Test NPCs",
    )
    db.session.add(adventure)
    db.session.commit()

    # When the adventure is deleted
    response = client.delete("/delete_adventures_from_db", json={"ids": [adventure.id]})

    # Then the response status code should be 204
    assert response.status_code == 204

    # And the adventure should no longer be in the database
    assert Adventures.query.filter_by(id=adventure.id).first() is None


def test_get_NPCs_from_db(client):
    # GIVEN an adventure with NPCs
    adventure = Adventures(
        adventure_title="Test Adventure",
        adventure_hook="Test Hook",
        adventure_plot="Test Plot",
        adventure_climax="Test Climax",
        adventure_resolution="Test Resolution",
        adventure_npcs="Test NPCs",
    )
    db.session.add(adventure)
    db.session.commit()

    npc1 = AdventureNPCs(adventure_id=adventure.id, npc_name="Test NPC 1")
    npc2 = AdventureNPCs(adventure_id=adventure.id, npc_name="Test NPC 2")
    db.session.add(npc1)
    db.session.add(npc2)
    db.session.commit()

    # WHEN the client requests the NPCs for the adventure
    response = client.get(f"/get_NPCs_from_db?id={adventure.id}")

    # THEN the response should contain the correct NPCs
    assert response.status_code == 200
    npc_data = response.get_json()
    assert len(npc_data) == 2
    assert "Test NPC 1" in npc_data
    assert "Test NPC 2" in npc_data



def test_get_locations_from_db(client):
    # GIVEN an adventure with two locations
    adventure = Adventures(
        adventure_title="Test Adventure",
        adventure_hook="Test Hook",
        adventure_plot="Test Plot",
        adventure_climax="Test Climax",
        adventure_resolution="Test Resolution",
        adventure_npcs="Test NPCs",
    )
    db.session.add(adventure)
    db.session.commit()

    location1 = AdventureLocations(
        adventure_id=adventure.id, location_name="Test Location 1"
    )
    location2 = AdventureLocations(
        adventure_id=adventure.id, location_name="Test Location 2"
    )
    db.session.add(location1)
    db.session.add(location2)
    db.session.commit()

    # WHEN the get_locations_from_db route is called for the adventure
    response = client.get(f"/get_locations_from_db?id={adventure.id}")

    # THEN the response should have a 200 status code
    assert response.status_code == 200

    # AND the response should contain the correct locations
    location_data = response.get_json()
    assert len(location_data) == 2
    assert "Test Location 1" in location_data
    assert "Test Location 2" in location_data


@patch("openai.Completion.create")
def test_generate_npc(mock_create, client, app):
    # Given an adventure and a mocked response from OpenAI
    mock_response = {
        "choices": [
            {"text": '{"NPCBackground": "Background text", "NPCStats": "Stats text"}'}
        ]
    }
    mock_create.return_value = mock_response


    adventure = Adventures(
        adventure_title="Test Adventure",
        adventure_hook="Test Hook",
        adventure_plot="Test Plot",
        adventure_climax="Test Climax",
        adventure_resolution="Test Resolution",
        adventure_npcs="[]",
    )
    db.session.add(adventure)
    db.session.commit()

    # When a request is sent to generate an NPC
    response = client.post(
        "/generate_npc",
        json={
            "adventureId": adventure.id,
            "characterName": "Test NPC",
            "selectedSystem": "D&D 5e",
            "selectedSystemVersion": "Core",
            "custom_system": "",
        },
    )

    # Then the response should be valid and the NPC should be added to the database
    assert response.status_code == 201
    assert response.json == {
        "status": "success",
        "message": {"NPCBackground": "Background text", "NPCStats": "Stats text"},
    }
    assert AdventureNPCs.query.count() == 1
    npc = AdventureNPCs.query.first()
    assert npc.npc_name == "Test NPC"
