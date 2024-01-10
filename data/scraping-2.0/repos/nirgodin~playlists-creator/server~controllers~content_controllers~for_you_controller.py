from genie_common.openai import OpenAIClient
from spotipyio import SpotifyClient
from spotipyio.logic.collectors.top_items_collectors.items_type import ItemsType
from spotipyio.logic.collectors.top_items_collectors.time_range import TimeRange

from server.consts.data_consts import ITEMS
from server.controllers.content_controllers.base_content_controller import BaseContentController
from server.data.playlist_resources import PlaylistResources
from server.logic.data_collection.spotify_playlist_details_collector import PlaylistDetailsCollector
from server.logic.playlist_imitation.playlist_imitator import PlaylistImitator
from server.logic.playlists_creator import PlaylistsCreator
from server.tools.authenticator import Authenticator
from server.tools.spotify_session_creator import SpotifySessionCreator


class ForYouController(BaseContentController):
    def __init__(self,
                 authenticator: Authenticator,
                 playlists_creator: PlaylistsCreator,
                 openai_client: OpenAIClient,
                 session_creator: SpotifySessionCreator,
                 playlists_imitator: PlaylistImitator,
                 playlist_details_collector: PlaylistDetailsCollector = PlaylistDetailsCollector()):
        super().__init__(authenticator, playlists_creator, openai_client, session_creator)
        self._playlist_details_collector = playlist_details_collector
        self._playlists_imitator = playlists_imitator

    async def _generate_playlist_resources(self,
                                           request_body: dict,
                                           dir_path: str,
                                           spotify_client: SpotifyClient) -> PlaylistResources:
        response = await spotify_client.current_user.top_items.run(
            items_type=ItemsType.TRACKS,
            time_range=TimeRange.MEDIUM_TERM,
            limit=50
        )
        tracks = response[ITEMS]
        playlist_details = await self._playlist_details_collector.collect_playlist(tracks, spotify_client)

        return await self._playlists_imitator.imitate_playlist(playlist_details, dir_path)

    async def _generate_playlist_cover(self, request_body: dict, image_path: str) -> None:  # TODO: Implement
        raise NotImplementedError
