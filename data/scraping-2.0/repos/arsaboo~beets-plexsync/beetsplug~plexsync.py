"""Update and sync Plex music library.

Plex users enter the Plex Token to enable updating.
Put something like the following in your config.yaml to configure:
    plex:
        host: localhost
        token: token
"""

import asyncio
import difflib
import os
import re
import time
from datetime import datetime

import confuse
import dateutil.parser
import requests
import spotipy
from beets import config, ui
from beets.dbcore import types
from beets.dbcore.query import MatchQuery
from beets.library import DateType
from beets.plugins import BeetsPlugin
from beets.ui import input_, print_
from bs4 import BeautifulSoup
from jiosaavn import JioSaavn
from plexapi import exceptions
from plexapi.server import PlexServer
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth


class PlexSync(BeetsPlugin):
    """Define plexsync class."""

    data_source = "Plex"

    item_types = {
        "plex_guid": types.STRING,
        "plex_ratingkey": types.INTEGER,
        "plex_userrating": types.FLOAT,
        "plex_skipcount": types.INTEGER,
        "plex_viewcount": types.INTEGER,
        "plex_lastviewedat": DateType(),
        "plex_lastratedat": DateType(),
        "plex_updated": DateType(),
    }

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""

        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def __init__(self):
        """Initialize plexsync plugin."""
        super().__init__()

        self.config_dir = config.config_dir()
        self.google = None
        self.openai = None

        # Call the setup methods
        try:
            self.setup_google_ai()
        except Exception as e:
            print(f"Failed to set up Google AI: {e}")
            self.google = None

        try:
            self.setup_openai_api()
        except Exception as e:
            print(f"Failed to set up OpenAI API: {e}")
            self.openai = None

        # Adding defaults.
        config["plex"].add(
            {
                "host": "localhost",
                "port": 32400,
                "token": "",
                "library_name": "Music",
                "secure": False,
                "ignore_cert_errors": False,
            }
        )

        config["plexsync"].add(
            {"tokenfile": "spotify_plexsync.json", "manual_search": False}
        )
        self.plexsync_token = config["plexsync"]["tokenfile"].get(
            confuse.Filename(in_app_dir=True)
        )

        # add OpenAI defaults
        config["openai"].add(
            {
                "api_key": "",
                "model": "gpt-3.5-turbo",
            }
        )

        config["openai"]["api_key"].redact = True
        config["google"]["api_key"].redact = True

        config["plex"]["token"].redact = True
        baseurl = (
            "http://"
            + config["plex"]["host"].get()
            + ":"
            + str(config["plex"]["port"].get())
        )
        try:
            self.plex = PlexServer(baseurl, config["plex"]["token"].get())
        except exceptions.Unauthorized:
            raise ui.UserError("Plex authorization failed")
        try:
            self.music = self.plex.library.section(config["plex"]["library_name"].get())
        except exceptions.NotFound:
            raise ui.UserError(
                f"{config['plex']['library_name']} \
                library not found"
            )
        self.register_listener("database_change", self.listen_for_db_change)

    def authenticate_spotify_old(self):
        ID = config["spotify"]["client_id"].get()
        SECRET = config["spotify"]["client_secret"].get()
        redirect_uri = "http://localhost/"
        scope = (
            "user-read-private user-read-email playlist-modify-public "
            "playlist-modify-private playlist-read-private"
        )
        # Create a SpotifyOAuth object with your credentials and scope
        self.auth_manager = SpotifyOAuth(
            client_id=ID,
            client_secret=SECRET,
            redirect_uri=redirect_uri,
            scope=scope,
            open_browser=False,
            cache_path=self.plexsync_token,
        )
        # Create a Spotify object with the auth_manager
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)

    def authenticate_spotify(self):
        ID = config["spotify"]["client_id"].get()
        SECRET = config["spotify"]["client_secret"].get()
        redirect_uri = "http://localhost/"
        scope = (
            "user-read-private user-read-email playlist-modify-public "
            "playlist-modify-private playlist-read-private"
        )

        # Create a SpotifyOAuth object with your credentials and scope
        self.auth_manager = SpotifyOAuth(
            client_id=ID,
            client_secret=SECRET,
            redirect_uri=redirect_uri,
            scope=scope,
            open_browser=False,
            cache_path=self.plexsync_token,
        )
        self.token_info = self.auth_manager.get_cached_token()
        if self.token_info is None:
            self.auth_manager.get_access_token(as_dict=True)
        need_token = self.auth_manager.is_token_expired(self.token_info)
        if need_token:
            new_token = self.auth_manager.refresh_access_token(
                self.token_info["refresh_token"]
            )
            self.token_info = new_token
        # Create a Spotify object with the auth_manager
        self.sp = spotipy.Spotify(auth=self.token_info.get("access_token"))

    def import_spotify_playlist(self, playlist_id):
        """This function returns a list of tracks in a Spotify playlist."""
        self.authenticate_spotify()
        songs = self.get_playlist_tracks(playlist_id)
        song_list = []
        for song in songs:
            # Find and store the song title
            if ('From "' in song["track"]["name"]) or (
                "From &quot" in song["track"]["name"]
            ):
                title_orig = song["track"]["name"].replace("&quot;", '"')
                title, album = self.parse_title(title_orig)
            else:
                title = song["track"]["name"]
                album = self.clean_album_name(song["track"]["album"]["name"])
            try:
                year = dateutil.parser.parse(
                    song["track"]["album"]["release_date"], ignoretz=True
                )
            except ValueError:
                year = None
            # Find and store the song artist
            artist = song["track"]["artists"][0]["name"]
            # Create a dictionary with the song information
            song_dict = {
                "title": title.strip(),
                "album": album.strip(),
                "artist": artist.strip(),
                "year": year,
            }
            # Append the dictionary to the list of songs
            song_list.append(song_dict)
        return song_list

    def get_playlist_id(self, url):
        # split the url by "/"
        parts = url.split("/")
        # find the index of "playlist"
        index = parts.index("playlist")
        # get the next part as the playlist id
        playlist_id = parts[index + 1]
        # return the playlist id
        return playlist_id

    def get_playlist_tracks(self, playlist_id):
        """This function returns a list of tracks in a Spotify playlist.

        Args:
            playlist_id (string): Spotify playlist ID

        Returns:
            list: tracks in a Spotify playlist
        """

        tracks_response = self.sp.playlist_tracks(playlist_id)
        tracks = tracks_response["items"]
        while tracks_response["next"]:
            tracks_response = self.sp.next(tracks_response)
            tracks.extend(tracks_response["items"])
        return tracks

    def listen_for_db_change(self, lib, model):
        """Listens for beets db change and register the update for the end."""
        self.register_listener("cli_exit", self._plexupdate)

    def commands(self):
        """Add beet UI commands to interact with Plex."""
        plexupdate_cmd = ui.Subcommand(
            "plexupdate", help=f"Update {self.data_source} library"
        )

        def func(lib, opts, args):
            self._plexupdate()

        plexupdate_cmd.func = func

        # plexsync command
        sync_cmd = ui.Subcommand("plexsync", help="fetch track attributes from Plex")
        sync_cmd.parser.add_option(
            "-f",
            "--force",
            dest="force_refetch",
            action="store_true",
            default=False,
            help="re-sync Plex data when already present",
        )

        def func_sync(lib, opts, args):
            items = lib.items(ui.decargs(args))
            self._fetch_plex_info(items, ui.should_write(), opts.force_refetch)

        sync_cmd.func = func_sync

        # plexplaylistadd command
        playlistadd_cmd = ui.Subcommand(
            "plexplaylistadd", help="add tracks to Plex playlist"
        )
        playlistadd_cmd.parser.add_option(
            "-m", "--playlist", default="Beets", help="add playlist to Plex"
        )

        def func_playlist_add(lib, opts, args):
            items = lib.items(ui.decargs(args))
            self._plex_add_playlist_item(items, opts.playlist)

        playlistadd_cmd.func = func_playlist_add

        # plexplaylistremove command
        playlistrem_cmd = ui.Subcommand(
            "plexplaylistremove", help="Plex playlist to edit"
        )
        playlistrem_cmd.parser.add_option(
            "-m", "--playlist", default="Beets", help="Plex playlist to edit"
        )

        def func_playlist_rem(lib, opts, args):
            items = lib.items(ui.decargs(args))
            self._plex_remove_playlist_item(items, opts.playlist)

        playlistrem_cmd.func = func_playlist_rem

        # plexsyncrecent command - instead of using the plexsync command which
        # can be slow, we can use the plexsyncrecent command to update info
        # for tracks played in the last X days.
        syncrecent_cmd = ui.Subcommand(
            "plexsyncrecent", help="Sync recently played tracks"
        )
        syncrecent_cmd.parser.add_option(
            "--days", default=7, help="Number of days to be synced"
        )

        def func_sync_recent(lib, opts, args):
            self._update_recently_played(lib, opts.days)

        syncrecent_cmd.func = func_sync_recent

        # plexplaylistimport command
        playlistimport_cmd = ui.Subcommand(
            "plexplaylistimport", help="import playlist in to Plex"
        )

        playlistimport_cmd.parser.add_option(
            "-m",
            "--playlist",
            default="Beets",
            help="name of the playlist to be added in Plex",
        )
        playlistimport_cmd.parser.add_option(
            "-u",
            "--url",
            default="",
            help="playlist URL to be imported in Plex",
        )
        playlistimport_cmd.parser.add_option(
            "-l",
            "--listenbrainz",
            action="store_true",
            help="use ListenBrainz as input option",
        )

        def func_playlist_import(lib, opts, args):
            self._plex_import_playlist(opts.playlist, opts.url, opts.listenbrainz)

        playlistimport_cmd.func = func_playlist_import

        # plexplaylist2collection command
        plexplaylist2collection_cmd = ui.Subcommand("plexplaylist2collection")

        plexplaylist2collection_cmd.parser.add_option(
            "-m",
            "--playlist",
            default="Beets",
            help="name of the playlist to be converted",
        )

        def func_playlist2collection(lib, opts, args):
            self._plex_playlist_to_collection(opts.playlist)

        plexplaylist2collection_cmd.func = func_playlist2collection

        # plexsearchimport command
        searchimport_cmd = ui.Subcommand(
            "plexsearchimport",
            help="import playlist in to Plex based on Youtube search",
        )
        searchimport_cmd.parser.add_option(
            "-m",
            "--playlist",
            default="Beets",
            help="name of the playlist to be added in Plex",
        )
        searchimport_cmd.parser.add_option(
            "-s",
            "--search",
            default="",
            help="Create playlist based on Youtube search in Plex",
        )
        searchimport_cmd.parser.add_option(
            "-l", "--limit", default=10, help="Number of tracks"
        )

        def func_search_import(lib, opts, args):
            self._plex_import_search(opts.playlist, opts.search, opts.limit)

        searchimport_cmd.func = func_search_import

        # plexplaylistclear command
        playlistclear_cmd = ui.Subcommand(
            "plexplaylistclear", help="clear Plex playlist"
        )

        playlistclear_cmd.parser.add_option(
            "-m",
            "--playlist",
            default="",
            help="name of the Plex playlist to be cleared",
        )

        def func_playlist_clear(lib, opts, args):
            self._plex_clear_playlist(opts.playlist)

        playlistclear_cmd.func = func_playlist_clear

        # plexcollage command
        collage_cmd = ui.Subcommand(
            "plexcollage", help="create album collage based on Plex history"
        )

        collage_cmd.parser.add_option(
            "-i", "--interval", default=7, help="days to look back for history"
        )
        collage_cmd.parser.add_option(
            "-g", "--grid", default=3, help="dimension of the collage grid"
        )

        def func_collage(lib, opts, args):
            self._plex_collage(opts.interval, opts.grid)

        collage_cmd.func = func_collage

        # plexsonic command
        sonicsage_cmd = ui.Subcommand(
            "plexsonic", help="create ChatGPT-based playlists"
        )

        sonicsage_cmd.parser.add_option(
            "-n", "--number", default=10, help="number of song recommendations"
        )
        sonicsage_cmd.parser.add_option(
            "-p", "--prompt", default="", help="describe what you want to hear"
        )
        sonicsage_cmd.parser.add_option(
            "-m",
            "--playlist",
            default="SonicSage",
            help="name of the playlist to be added in Plex",
        )
        sonicsage_cmd.parser.add_option(
            "-c",
            "--clear",
            dest="clear",
            default=False,
            help="Clear playlist if not empty",
        )

        def func_sonic(lib, opts, args):
            self._plex_sonicsage(opts.number, opts.prompt, opts.playlist, opts.clear)

        sonicsage_cmd.func = func_sonic

        # plex2spotify command
        plex2spotify_cmd = ui.Subcommand(
            "plex2spotify", help="Transfer Plex playlist to Spotify"
        )

        plex2spotify_cmd.parser.add_option(
            "-m",
            "--playlist",
            default="beets",
            help="name of the playlist to be added in Spotify",
        )

        def func_plex2spotify(lib, opts, args):
            self._plex2spotify(lib, opts.playlist)

        plex2spotify_cmd.func = func_plex2spotify

        return [
            plexupdate_cmd,
            sync_cmd,
            playlistadd_cmd,
            playlistrem_cmd,
            syncrecent_cmd,
            playlistimport_cmd,
            playlistclear_cmd,
            collage_cmd,
            sonicsage_cmd,
            searchimport_cmd,
            plexplaylist2collection_cmd,
            plex2spotify_cmd,
        ]

    def parse_title(self, title_orig):
        if '(From "' in title_orig:
            title = re.sub(r"\(From.*\)", "", title_orig)
            album = re.sub(r'^[^"]+"|(?<!^)"[^"]+"|"[^"]+$', "", title_orig)
        elif '[From "' in title_orig:
            title = re.sub(r"\[From.*\]", "", title_orig)
            album = re.sub(r'^[^"]+"|(?<!^)"[^"]+"|"[^"]+$', "", title_orig)
        else:
            title = title_orig
            album = ""
        return title.strip(), album.strip()

    def clean_album_name(self, album_orig):
        album_orig = (
            album_orig.replace("(Original Motion Picture Soundtrack)", "")
            .replace("- Hindi", "")
            .strip()
        )
        if '(From "' in album_orig:
            album = re.sub(r'^[^"]+"|(?<!^)"[^"]+"|"[^"]+$', "", album_orig)
        elif '[From "' in album_orig:
            album = re.sub(r'^[^"]+"|(?<!^)"[^"]+"|"[^"]+$', "", album_orig)
        else:
            album = album_orig
        return album

    saavn = JioSaavn()

    # Define a function to get playlist songs by id
    async def get_playlist_songs(playlist_url):
        # Use the async method from saavn
        songs = await saavn.get_playlist_songs(playlist_url)
        # Return a list of songs with details
        return songs

    def import_jiosaavn_playlist(self, playlist_url):
        data = asyncio.run(
            self.saavn.get_playlist_songs(playlist_url, page=1, limit=100)
        )
        songs = data["data"]["list"]
        song_list = []
        for song in songs:
            # Find and store the song title
            if ('From "' in song["title"]) or ("From &quot" in song["title"]):
                title_orig = song["title"].replace("&quot;", '"')
                title, album = self.parse_title(title_orig)
            else:
                title = song["title"]
                album = self.clean_album_name(song["more_info"]["album"])
            year = song["year"]
            # Find and store the song artist
            try:
                artist = song["more_info"]["artistMap"]["primary_artists"][0]["name"]
            except KeyError:
                continue
            # Find and store the song duration
            # duration = song.find("div", class_="songs-list-row__length").text.strip()
            # Create a dictionary with the song information
            song_dict = {
                "title": title.strip(),
                "album": album.strip(),
                "artist": artist.strip(),
                "year": year,
            }
            # Append the dictionary to the list of songs
            song_list.append(song_dict)
        return song_list

    # Define a function that takes a title string and a list of tuples as input
    def find_closest_match(self, title, lst):
        # Initialize an empty list to store the matches and their scores
        matches = []
        # Loop through each tuple in the list
        for t in lst:
            # Use the SequenceMatcher class to compare the title with the
            # first element of the tuple
            # The ratio method returns a score between 0 and 1 indicating how
            # similar the two strings are based on the Levenshtein distance
            score = difflib.SequenceMatcher(None, title, t.title).ratio()
            # Append the tuple and the score to the matches list
            matches.append((t, score))
        # Sort the matches list by the score in descending order
        matches.sort(key=lambda x: x[1], reverse=True)
        # Return only the first element of each tuple in the matches
        # list as a new list
        return [m[0] for m in matches]

    def import_gaana_playlist_old(self, playlist_url):
        # Make a GET request to the playlist url
        response = requests.get(playlist_url)
        # Check if the response is successful
        if response.status_code == 200:
            # Parse the html data from the response
            soup = BeautifulSoup(response.text, "html.parser")
            # Find all the div elements with class "s_c"
            result = soup.find_all("ul", {"class": "_row list_data"})
            # Create an empty list to store the tracks
            tracks = []
            # Loop through each div element
            for div in result:
                div_art = div.find("div", {"class": "_art"})
                artist = div_art.text.strip()
                div_alb = div.find("div", {"class": "_alb"})
                album = div_alb.text.strip()
                span = div.find("span", {"class": "t_over"})
                # Get the text content of the span element
                title_tmp = span.text.strip()
                title_orig = re.sub("^Premium  ", "", title_tmp)
                if '(From "' in title_orig or '[From "' in title_orig:
                    title, album = self.parse_title(title_orig)
                else:
                    title = title_orig.strip()
                song_dict = {
                    "title": title.strip(),
                    "album": self.clean_album_name(album.strip()),
                    "artist": artist,
                }
                # Append the title to the tracks list
                tracks.append(song_dict)
            # Return the tracks as a list of strings
            return tracks
        else:
            # Raise an exception if the response is not successful
            raise Exception(
                f"Gaana website returned status code \
                            {response.status_code}"
            )

    def import_apple_playlist(self, url):
        import json

        # Send a GET request to the URL and get the HTML content
        response = requests.get(url)
        content = response.text

        # Create a BeautifulSoup object with the HTML content
        soup = BeautifulSoup(content, "html.parser")
        try:
            data = soup.find("script", id="serialized-server-data").text
        except:
            self._log.debug("Error parsing Apple Music playlist")
            return None
        # load the data as a JSON object
        data = json.loads(data)
        songs = data[0]["data"]["sections"][1]["items"]

        # Create an empty list to store the songs
        song_list = []
        # Loop through each song element
        for song in songs:
            # Find and store the song title
            title = song["title"].strip()
            album = song["tertiaryLinks"][0]["title"]
            # Find and store the song artist
            artist = song["subtitleLinks"][0]["title"]
            # Create a dictionary with the song information
            song_dict = {
                "title": title.strip(),
                "album": album.strip(),
                "artist": artist.strip(),
            }
            # Append the dictionary to the list of songs
            song_list.append(song_dict)
        return song_list

    def _plexupdate(self):
        """Update Plex music library."""
        try:
            self.music.update()
            self._log.info("Update started.")
        except exceptions.PlexApiException:
            self._log.warning("{} Update failed", self.config["plex"]["library_name"])

    def _fetch_plex_info(self, items, write, force):
        """Obtain track information from Plex."""
        for index, item in enumerate(items, start=1):
            self._log.info("Processing {}/{} tracks - {} ", index, len(items), item)
            # If we're not forcing re-downloading for all tracks, check
            # whether the popularity data is already present
            if not force:
                if "plex_userrating" in item:
                    self._log.debug("Plex rating already present for: {}", item)
                    continue
            plex_track = self.search_plex_track(item)
            if plex_track is None:
                self._log.info("No track found for: {}", item)
                continue
            item.plex_guid = plex_track.guid
            item.plex_ratingkey = plex_track.ratingKey
            item.plex_userrating = plex_track.userRating
            item.plex_skipcount = plex_track.skipCount
            item.plex_viewcount = plex_track.viewCount
            item.plex_lastviewedat = plex_track.lastViewedAt
            item.plex_lastratedat = plex_track.lastRatedAt
            item.plex_updated = time.time()
            item.store()
            if write:
                item.try_write()

    def search_plex_track(self, item):
        """Fetch the Plex track key."""
        tracks = self.music.searchTracks(
            **{"album.title": item.album, "track.title": item.title}
        )
        if len(tracks) == 1:
            return tracks[0]
        elif len(tracks) > 1:
            for track in tracks:
                if track.parentTitle == item.album and track.title == item.title:
                    return track
        else:
            self._log.debug("Track {} not found in Plex library", item)
            return None

    def sort_plex_playlist(self, playlist_name, sort_field):
        """Sort a Plex playlist by a given field."""

        # Get the playlist
        playlist = self.plex.playlist(playlist_name)

        # Get the items in the playlist
        items = playlist.items()

        # Sort the items based on the sort_field
        sorted_items = sorted(
            items,
            key=lambda x: getattr(x, sort_field)
            if getattr(x, sort_field) is not None
            else datetime(1900, 1, 1),
        )

        # Remove all items from the playlist
        playlist.removeItems(items)

        # Add the sorted items back to the playlist
        for item in sorted_items:
            playlist.addItems(item)

    def _plex_add_playlist_item(self, items, playlist):
        """Add items to Plex playlist."""
        plex_set = set()
        try:
            plst = self.plex.playlist(playlist)
            playlist_set = set(plst.items())
        except exceptions.NotFound:
            plst = None
            playlist_set = set()
        for item in items:
            try:
                plex_set.add(self.plex.fetchItem(item.plex_ratingkey))
            except (exceptions.NotFound, AttributeError) as e:
                self._log.warning("{} not found in Plex library. Error: {}", item, e)
                continue
        to_add = plex_set - playlist_set
        self._log.info("Adding {} tracks to {} playlist", len(to_add), playlist)
        if plst is None:
            self._log.info("{} playlist will be created", playlist)
            self.plex.createPlaylist(playlist, items=list(to_add))
        else:
            try:
                plst.addItems(items=list(to_add))
            except exceptions.BadRequest as e:
                self._log.error(
                    "Error adding items {} to {} playlist. Error: {}",
                    items,
                    playlist,
                    e,
                )
        self.sort_plex_playlist(playlist, "lastViewedAt")

    def _plex_playlist_to_collection(self, playlist):
        """Convert a Plex playlist to a Plex collection."""
        try:
            plst = self.music.playlist(playlist)
            playlist_set = set(plst.items())
        except exceptions.NotFound:
            self._log.error("{} playlist not found", playlist)
            return
        try:
            col = self.music.collection(playlist)
            collection_set = set(col.items())
        except exceptions.NotFound:
            col = None
            collection_set = set()
        to_add = playlist_set - collection_set
        self._log.info("Adding {} tracks to {} collection", len(to_add), playlist)
        if col is None:
            self._log.info("{} collection will be created", playlist)
            self.music.createCollection(playlist, items=list(to_add))
        else:
            try:
                col.addItems(items=list(to_add))
            except exceptions.BadRequest as e:
                self._log.error(
                    "Error adding items {} to {} collection. Error: {}",
                    items,
                    playlist,
                    e,
                )

    def _plex_remove_playlist_item(self, items, playlist):
        """Remove items from Plex playlist."""
        plex_set = set()
        try:
            plst = self.plex.playlist(playlist)
            playlist_set = set(plst.items())
        except exceptions.NotFound:
            self._log.error("{} playlist not found", playlist)
            return
        for item in items:
            try:
                plex_set.add(self.plex.fetchItem(item.plex_ratingkey))
            except exceptions.NotFound as e:
                self._log.warning("{} not found in Plex library. Error: {}", item, e)
                continue
        to_remove = plex_set.intersection(playlist_set)
        self._log.info("Removing {} tracks from {} playlist", len(to_remove), playlist)
        plst.removeItems(items=list(to_remove))

    def _update_recently_played(self, lib, days=7):
        """Fetch the Plex track key."""
        tracks = self.music.search(
            filters={"track.lastViewedAt>>": f"{days}d"}, libtype="track"
        )
        self._log.info("Updating information for {} tracks", len(tracks))
        with lib.transaction():
            for track in tracks:
                query = MatchQuery("plex_ratingkey", track.ratingKey, fast=False)
                items = lib.items(query)
                if not items:
                    self._log.debug("{} | track not found", query)
                    continue
                elif len(items) == 1:
                    self._log.info("Updating information for {} ", items[0])
                    try:
                        items[0].plex_userrating = track.userRating
                        items[0].plex_skipcount = track.skipCount
                        items[0].plex_viewcount = track.viewCount
                        items[0].plex_lastviewedat = track.lastViewedAt
                        items[0].plex_lastratedat = track.lastRatedAt
                        items[0].plex_updated = time.time()
                        items[0].store()
                        items[0].try_write()
                    except exceptions.NotFound:
                        self._log.debug("{} | track not found", items[0])
                        continue
                else:
                    self._log.debug("Please sync Plex library again")
                    continue

    def search_plex_song(self, song, manual_search=False):
        """Fetch the Plex track key."""

        if 'From "' in song["title"] or '[From "' in song["title"]:
            song["title"], song["album"] = self.parse_title(song["title"])

        try:
            if song["album"] is None:
                tracks = self.music.searchTracks(**{"track.title": song["title"]})
            else:
                tracks = self.music.searchTracks(
                    **{"album.title": song["album"], "track.title": song["title"]}
                )
                if len(tracks) == 0:
                    tracks = self.music.searchTracks(**{"track.title": song["title"]})
        except exceptions as e:
            self._log.debug(
                "Error searching for {} - {}. Error: {}",
                song["album"],
                song["title"],
                e,
            )
            return None
        artist = song["artist"].split(",")[0]
        if len(tracks) == 1:
            return tracks[0]
        elif len(tracks) > 1:
            sorted_tracks = self.find_closest_match(song["title"], tracks)
            self._log.debug("Found {} tracks for {}", len(sorted_tracks), song["title"])
            if manual_search and len(sorted_tracks) > 0:
                print_(f'Choose candidates for {song["album"]} - ' f'{song["title"]}:')
                for i, track in enumerate(sorted_tracks, start=1):
                    print_(
                        f"{i}. {track.parentTitle} - {track.title} - "
                        f"{track.artist().title}"
                    )
                sel = ui.input_options(
                    ("aBort", "Skip"), numrange=(1, len(sorted_tracks)), default=1
                )
                if sel in ("b", "B", "s", "S"):
                    return None
                return sorted_tracks[sel - 1] if sel > 0 else None
            for track in sorted_tracks:
                if track.originalTitle is not None:
                    plex_artist = track.originalTitle
                else:
                    plex_artist = track.artist().title
                if artist in plex_artist:
                    return track
        else:
            if config["plexsync"]["manual_search"] and not manual_search:
                self._log.info(
                    "Track {} - {} not found in Plex", song["album"], song["title"]
                )
                if ui.input_yn("Search manually? (Y/n)"):
                    self.manual_track_search()
            else:
                self._log.info(
                    "Track {} - {} not found in Plex", song["album"], song["title"]
                )
            return None

    def manual_track_search(self):
        """Manually search for a track in the Plex library.

        Prompts the user to enter the title, album, and artist of the track
        they want to search for.
        Calls the `search_plex_song` method with the provided information and
        sets the `manual_search` flag to True.
        """
        song_dict = {}
        title = input_("Title:").strip()
        album = input_("Album:").strip()
        artist = input_("Artist:").strip()
        song_dict = {
            "title": title.strip(),
            "album": album.strip(),
            "artist": artist.strip(),
        }
        self.search_plex_song(song_dict, manual_search=True)

    def _plex_import_playlist(self, playlist, playlist_url=None, listenbrainz=False):
        """Import playlist into Plex."""
        if listenbrainz:
            try:
                from beetsplug.listenbrainz import ListenBrainzPlugin
            except ModuleNotFoundError:
                self._log.error("ListenBrainz plugin not installed")
                return
            try:
                lb = ListenBrainzPlugin()
            except Exception as e:
                self._log.error(
                    "Unable to initialize ListenBrainz plugin. Error: {}", e
                )
                return
            # there are 2 playlists to be imported. 1. Weekly jams 2. Weekly exploration
            # get the weekly jams playlist
            self._log.info("Importing weekly jams playlist")
            weekly_jams = lb.get_weekly_jams()
            self._log.info("Importing {} songs from Weekly Jams", len(weekly_jams))
            self.add_songs_to_plex("Weekly Jams", weekly_jams)

            self._log.info("Importing weekly exploration playlist")
            weekly_exploration = lb.get_weekly_exploration()
            self._log.info(
                "Importing {} songs from Weekly Exploration", len(weekly_exploration)
            )
            self.add_songs_to_plex("Weekly Exploration", weekly_exploration)
        else:
            if playlist_url is None or (
                "http://" not in playlist_url and "https://" not in playlist_url
            ):
                raise ui.UserError("Playlist URL not provided")
            if "apple" in playlist_url:
                songs = self.import_apple_playlist(playlist_url)
            elif "jiosaavn" in playlist_url:
                songs = self.import_jiosaavn_playlist(playlist_url)
            elif "gaana.com" in playlist_url:
                songs = self.import_gaana_playlist(playlist_url)
            elif "spotify" in playlist_url:
                songs = self.import_spotify_playlist(self.get_playlist_id(playlist_url))
            elif "youtube" in playlist_url:
                songs = self.import_yt_playlist(playlist_url)
            elif "tidal" in playlist_url:
                songs = self.import_tidal_playlist(playlist_url)
            else:
                songs = []
                self._log.error("Playlist URL not supported")
            self._log.info("Importing {} songs from {}", len(songs), playlist_url)
            self.add_songs_to_plex(playlist, songs)

    def add_songs_to_plex(self, playlist, songs):
        song_list = []
        if songs:
            for song in songs:
                if self.search_plex_song(song) is not None:
                    found = self.search_plex_song(song)
                    song_dict = {
                        "title": found.title,
                        "album": found.parentTitle,
                        "plex_ratingkey": found.ratingKey,
                    }
                    song_list.append(self.dotdict(song_dict))
        self._plex_add_playlist_item(song_list, playlist)

    def _plex_import_search(self, playlist, search, limit=10):
        """Import search results into Plex."""
        self._log.info("Searching for {}", search)
        songs = self.import_yt_search(search, limit)
        song_list = []
        if songs:
            for song in songs:
                if self.search_plex_song(song) is not None:
                    found = self.search_plex_song(song)
                    song_dict = {
                        "title": found.title,
                        "album": found.parentTitle,
                        "plex_ratingkey": found.ratingKey,
                    }
                    song_list.append(self.dotdict(song_dict))
        self._plex_add_playlist_item(song_list, playlist)

    def _plex_clear_playlist(self, playlist):
        """Clear Plex playlist."""
        # Get the playlist
        plist = self.plex.playlist(playlist)
        # Get a list of all the tracks in the playlist
        tracks = plist.items()
        # Loop through each track
        for track in tracks:
            # Remove the track from the playlist
            plist.removeItems(track)

    def _plex_collage(self, interval, grid):
        """Create a collage of most played albums."""
        self._log.info(
            "Creating collage of most played albums in the last {} " "days", interval
        )
        interval2 = str(interval) + "d"
        tracks = self.music.search(
            filters={"track.lastViewedAt>>": interval2},
            sort="viewCount:desc",
            libtype="track",
        )
        sorted = self._plex_most_played_albums(tracks, int(interval))
        # Create a list of album art
        album_art = []
        for album in sorted:
            album_art.append(album.thumbUrl)
        collage = self.create_collage(album_art, int(grid))
        try:
            collage.save(os.path.join(self.config_dir, "collage.png"))
        except Exception as e:
            self._log.error("Unable to save collage. Error: {}", e)
            return

    def _plex_most_played_albums(self, tracks, interval):
        from datetime import datetime, timedelta

        now = datetime.now()
        frm_dt = now - timedelta(days=interval)
        album = []
        # save album object, parenttitle, thumburl, viewcount, and last played date in album list
        for track in tracks:
            history = track.history(mindate=frm_dt)
            count = len(history)
            try:
                last_played_date = (
                    max(
                        (h for h in history if h.lastViewedAt is not None),
                        key=lambda x: x.lastViewedAt,
                    ).lastViewedAt
                    if history
                    else None
                )
            except ValueError:
                last_played_date = None
            if track.parentTitle not in [a[1] for a in album]:
                album.append(
                    [track.album(), track.parentTitle, count, last_played_date]
                )
            else:
                for i in album:
                    if i[1] == track.parentTitle:
                        i[2] += count
                        i[3] = max(i[3], last_played_date) if i[3] else last_played_date
        # sort album list by viewcount and then by last played date
        sorted_albums = sorted(
            album, key=lambda x: (-x[2], -x[3] if x[3] is not None else float("inf"))
        )
        # only return the album objects and add count and last played date to the album object
        for album in sorted_albums:
            album[0].count = album[2]
            album[0].last_played_date = album[3]
        # sort album objects by viewcount and then by last played date
        sorted_albums = [i[0] for i in sorted_albums]
        for album in sorted_albums:
            self._log.debug(
                "{} played {} times, last played on {}",
                album.title,
                album.count,
                album.last_played_date,
            )
        return sorted_albums

    def create_collage(self, list_image_urls, dimension):
        """Create a collage from a list of image urls."""
        import math
        from io import BytesIO

        from PIL import Image

        thumbnail_size = 300
        images = []
        for url in list_image_urls:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img = img.resize((thumbnail_size, thumbnail_size))
                images.append(img)
            except Exception:
                self._log.debug("Unable to fetch image from {}", url)
                continue
        # Calculate the size of the grid
        grid_size = thumbnail_size * dimension
        # Create the new image
        grid = Image.new("RGB", size=(grid_size, grid_size))
        # Paste the images into the grid
        for index, image in enumerate(images):
            x = thumbnail_size * (index % dimension)
            y = thumbnail_size * math.floor(index / dimension)
            grid.paste(image, box=(x, y))
        return grid

    def _plex_sonicsage(self, number, prompt, playlist, clear):
        """
        Generate song recommendations using OpenAI's GPT-3 model based on a
        given prompt, and add the recommended songs to a Plex playlist.

        Args:
            number (int): The number of song recommendations to generate.
            prompt (str): The prompt to use for generating song recommendations.
            playlist (str): The name of the Plex playlist to add the recommended songs to.
            clear (bool): Whether to clear the playlist before adding the recommended songs.

        Returns:
            None
        """
        if self.google is None and self.openai is None:
            self._log.error("No LLMs configured correctly")
            return
        if prompt == "":
            self._log.error("Prompt not provided")
            return
        if self.google:
            songs = self.google_ai_song_rec(number, prompt)
        elif self.openai:
            songs = self.chat_gpt_song_rec(number, prompt)
        song_list = []
        if songs is None:
            return
        for song in songs["songs"]:
            title = song["title"]
            album = song["album"]
            artist = song["artist"]
            year = song["year"]
            song_dict = {
                "title": title.strip(),
                "album": album.strip(),
                "artist": artist.strip(),
                "year": int(year),
            }
            song_list.append(song_dict)
        self._log.debug(
            "{} songs to be added in Plex library: {}", len(song_list), song_list
        )
        matched_songs = []
        for song in song_list:
            if self.search_plex_song(song) is not None:
                found = self.search_plex_song(song)
                match_dict = {
                    "title": found.title,
                    "album": found.parentTitle,
                    "plex_ratingkey": found.ratingKey,
                }
                self._log.debug("Song matched in Plex library: {}", match_dict)
                matched_songs.append(self.dotdict(match_dict))
        self._log.debug("Songs matched in Plex library: {}", matched_songs)
        if clear:
            try:
                self._plex_clear_playlist(playlist)
            except exceptions.NotFound:
                self._log.debug(f"Unable to clear playlist {playlist}")
        try:
            self._plex_add_playlist_item(matched_songs, playlist)
        except Exception as e:
            self._log.error("Unable to add songs to playlist. Error: {}", e)

    def setup_openai_api(self):
        import openai

        openai.api_key = config["openai"]["api_key"].get()
        try:
            openai.api_base = config["openai"]["api_base"].get()
        except Exception:
            pass
        self.openai = openai

    def setup_google_ai(self):
        import google.generativeai as genai

        key = config["google"]["api_key"].get()
        model = config["google"]["model"].get("gemini-pro")
        service_json = config["google"]["service_json"].get()
        self._log.debug("JSON key: {}", service_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_json
        genai.configure(api_key=key)
        try:
            gen_ai = genai.GenerativeModel(model_name=model)
            self.google = gen_ai
        except Exception:
            self.google = None
            pass

    def google_ai_song_rec(self, number, prompt):
        num_songs = int(number)
        sys_prompt = f"""
        You are a music recommendation system. You will reply with
        {num_songs} song recommendations in a JSON format. Only
        reply with the JSON object, no need to send anything else.
        Include title, artist, album, and year in the JSON response.
        Don't make up things. Use the JSON format:
        {{
            "songs": [
                {{
                    "title": "Title of song 1",
                    "artist": "Artist of Song 1",
                    "album": "Album of Song 1",
                    "year": "Year of release"
                }}
            ]
        }}
        Now, {prompt}
        """
        try:
            self._log.info("Sending request to Google AI")
            chat = self.google.generate_content(sys_prompt)
        except Exception as e:
            self._log.error("Unable to connect to Google AI. Error: {}", e)
            return
        reply = chat.text
        self._log.debug("Google AI replied: {}", reply)
        return self.extract_json(reply)

    def chat_gpt_song_rec(self, number, prompt):
        model = config["openai"]["model"].get()
        num_songs = int(number)
        sys_prompt = f"""
        You are a music recommender. You will reply with {num_songs} song
        recommendations in a JSON format. Only reply with the JSON object,
        no need to send anything else. Include title, artist, album, and
        year in the JSON response. Use the JSON format:
        {{
            "songs": [
                {{
                    "title": "Title of song 1",
                    "artist": "Artist of Song 1",
                    "album": "Album of Song 1",
                    "year": "Year of release"
                }}
            ]
        }}
        """
        messages = [{"role": "system", "content": sys_prompt}]
        messages.append({"role": "user", "content": prompt})
        try:
            self._log.info("Sending request to OpenAI")
            chat = self.openai.ChatCompletion.create(
                model=model, messages=messages, temperature=0.7
            )
        except Exception as e:
            self._log.error("Unable to connect to OpenAI. Error: {}", e)
            return
        reply = chat.choices[0].message.content
        tokens = chat.usage.total_tokens
        self._log.debug("OpenAI used {} tokens and replied: {}", tokens, reply)
        return self.extract_json(reply)

    def extract_json(self, jsonString):
        import json

        startIndex = jsonString.index("{")
        endIndex = jsonString.rindex("}")
        jsonSubstring = jsonString[startIndex : endIndex + 1]
        try:
            return json.loads(jsonSubstring)
        except Exception as e:
            self._log.error("Unable to parse JSON. Error: {}", e)
            return

    def import_yt_playlist(self, url):
        try:
            from beetsplug.youtube import YouTubePlugin
        except ModuleNotFoundError:
            self._log.error("YouTube plugin not installed")
            return
        try:
            ytp = YouTubePlugin()
        except Exception as e:
            self._log.error("Unable to initialize YouTube plugin. Error: {}", e)
            return
        return ytp.import_youtube_playlist(url)

    def import_yt_search(self, query, limit):
        try:
            from beetsplug.youtube import YouTubePlugin
        except ModuleNotFoundError:
            self._log.error("YouTube plugin not installed")
            return
        try:
            ytp = YouTubePlugin()
        except Exception as e:
            self._log.error("Unable to initialize YouTube plugin. Error: {}", e)
            return
        return ytp.import_youtube_search(query, limit)

    def import_tidal_playlist(self, url):
        try:
            from beetsplug.tidal import TidalPlugin
        except ModuleNotFoundError:
            self._log.error("Tidal plugin not installed")
            return
        try:
            tidal = TidalPlugin()
        except Exception as e:
            self._log.error("Unable to initialize Tidal plugin. Error: {}", e)
            return
        return tidal.import_tidal_playlist(url)

    def import_gaana_playlist(self, url):
        try:
            from beetsplug.gaana import GaanaPlugin
        except ModuleNotFoundError:
            self._log.error(
                "Gaana plugin not installed. \
                            See https://github.com/arsaboo/beets-gaana"
            )
            return
        try:
            gaana = GaanaPlugin()
        except Exception as e:
            self._log.error("Unable to initialize Gaana plugin. Error: {}", e)
            return
        return gaana.import_gaana_playlist(url)

    def _plex2spotify(self, lib, playlist):
        self.authenticate_spotify()
        plex_playlist = self.plex.playlist(playlist)
        plex_playlist_items = plex_playlist.items()
        self._log.debug(f"Plex playlist items: {plex_playlist_items}")
        spotify_tracks = []
        for item in plex_playlist_items:
            self._log.debug(f"Processing {item.ratingKey}")
            with lib.transaction():
                query = MatchQuery("plex_ratingkey", item.ratingKey, fast=False)
                items = lib.items(query)
                if not items:
                    self._log.debug(
                        f"Item not found in Beets "
                        f"{item.ratingKey}: {item.parentTitle} - "
                        f"{item.title}"
                    )
                    continue
                beets_item = items[0]
                self._log.debug(f"Beets item: {beets_item}")
                try:
                    spotify_track_id = beets_item.spotify_track_id
                    self._log.debug(
                        f"Spotify track id in beets: " f"{spotify_track_id}"
                    )
                except Exception:
                    spotify_track_id = None
                    self._log.debug("Spotify track_id not found in beets")
                if not spotify_track_id:
                    self._log.debug(
                        f"Searching for {beets_item.title} "
                        f"{beets_item.album} in Spotify"
                    )
                    spotify_search_results = self.sp.search(
                        q=f"track:{beets_item.title} album:{beets_item.album}",
                        limit=1,
                        type="track",
                    )
                    if not spotify_search_results["tracks"]["items"]:
                        self._log.info(f"Spotify match not found for " f"{beets_item}")
                        continue
                    spotify_track_id = spotify_search_results["tracks"]["items"][0][
                        "id"
                    ]
                spotify_tracks.append(spotify_track_id)
        self.add_tracks_to_spotify_playlist(playlist, spotify_tracks)

    def add_tracks_to_spotify_playlist_old(self, playlist_name, track_uris):
        user_id = self.sp.current_user()["id"]
        playlists = self.sp.user_playlists(user_id)
        playlist_exists = False
        for playlist in playlists["items"]:
            self._log.debug(
                f'Processing Playlist {playlist["name"]} ' f'with id {playlist["id"]}'
            )
            if playlist["name"].lower() == playlist_name.lower():
                playlist_id = playlist["id"]
                playlist_exists = True
                self._log.debug(
                    f"Playlist {playlist_name} exists " f"with id {playlist_id}"
                )
                # get the tracks in the playlist
                playlist_tracks = self.get_playlist_tracks(playlist_id)
                # get the track uris in the playlist
                uris = [
                    track["track"]["uri"].replace("spotify:track:", "")
                    for track in playlist_tracks["items"]
                ]
                self._log.debug(f"Playlist track uris: {uris}")
                # remove the tracks that are already in the playlist
                track_uris = list(set(track_uris) - set(uris))
                self._log.debug(f"Tracks to be added: {track_uris}")
                break
        if not playlist_exists:
            playlist = self.sp.user_playlist_create(
                user_id, playlist_name, public=False
            )
            playlist_id = playlist["id"]
            self._log.debug(
                f"Playlist {playlist_name} created " f"with id {playlist_id}"
            )
        self._log.debug(f"Adding tracks to playlist {playlist_id}")
        # add the tracks to the playlist
        if len(track_uris) > 0:
            for i in range(0, len(track_uris), 100):
                chunk = track_uris[i : i + 100]
                self.sp.user_playlist_add_tracks(user_id, playlist_id, chunk)
        else:
            self._log.debug("No tracks to add to playlist")

    def add_tracks_to_spotify_playlist(self, playlist_name, track_uris):
        user_id = self.sp.current_user()["id"]
        playlists = self.sp.user_playlists(user_id)
        playlist_id = None
        for playlist in playlists["items"]:
            if playlist["name"].lower() == playlist_name.lower():
                playlist_id = playlist["id"]
                break
        if not playlist_id:
            playlist = self.sp.user_playlist_create(
                user_id, playlist_name, public=False
            )
            playlist_id = playlist["id"]
            self._log.debug(
                f"Playlist {playlist_name} created with id " f"{playlist_id}"
            )
        playlist_tracks = self.get_playlist_tracks(playlist_id)
        # get the tracks in the playlist
        uris = [
            track["track"]["uri"].replace("spotify:track:", "")
            for track in playlist_tracks
        ]
        track_uris = list(set(track_uris) - set(uris))
        self._log.debug(f"Tracks to be added: {track_uris}")
        if len(track_uris) > 0:
            for i in range(0, len(track_uris), 100):
                chunk = track_uris[i : i + 100]
                self.sp.user_playlist_add_tracks(user_id, playlist_id, chunk)
            self._log.debug(
                f"Added {len(track_uris)} tracks to playlist " f"{playlist_id}"
            )
        else:
            self._log.debug("No tracks to add to playlist")
