from __future__ import print_function, unicode_literals

import re
import itertools
from HTMLParser import HTMLParser

from coherence import log
from coherence.upnp.core import DIDLLite
from coherence.upnp.core.utils import ReverseProxyUriResource
from coherence.upnp.core.DIDLLite import (
    Resource,
)
from coherence.backend import (
    BackendItem, Container, AbstractBackendStore,
)

import api
import settings

_htmlparser = HTMLParser()


class MoeFmProxyStream(ReverseProxyUriResource, log.Loggable):
    logCategory = 'moefm_stream'

    def __init__(self, uri, parent):
        self.parent = parent
        ReverseProxyUriResource.__init__(self, uri.encode("utf-8"))

    def log_playing(self):
        if self.parent.store.last_played_item is self:
            obj_id = self.parent.sub_id
            d = api.moefm.get(
                "/ajax/log?log_obj_type=sub&log_type=listen&obj_type=song&api=json", # noqa
                {"obj_id": obj_id}
            )
            d.addCallback(lambda res: self.debug(
                "Logged %s: %r", obj_id, res,
            ))
            d.addErrback(lambda res: self.warning(
                "Unable to log %s: %r", obj_id, res,
            ))

    def render(self, request):
        self.debug("render %r", self.parent.item_data)
        self.parent.container.on_item_play(self.parent)
        self.parent.store.last_played_item = self
        reactor.callLater(self.parent.duration_seconds / 2, self.log_playing)
        return ReverseProxyUriResource.render(self, request)


class MoeFmTrack(BackendItem):
    logCategory = "moefm"
    next_sn = 0

    def __init__(self, item_data, container):
        BackendItem.__init__(self)
        self.item_data = item_data
        self.container = container
        self.sub_id = item_data["sub_id"]
        self.storage_id = "track-%s$%s" % (self.sub_id, container.get_id())
        self.__class__.next_sn += 1
        self.sort_key = self.__class__.next_sn

        track_number = None
        m = re.match(
            r"^song\.(\d+)\s+.*$",
            _htmlparser.unescape(item_data["title"]),
            re.I,
        )
        if m:
            track_number, = m.groups()

        title = _htmlparser.unescape(item_data["sub_title"])
        self.name = title
        self.title = title
        self.originalTrackNumber = track_number
        self.artist = _htmlparser.unescape(item_data["artist"])
        self.album = _htmlparser.unescape(item_data["wiki_title"])
        self.cover = item_data["cover"]["large"]
        self.duration = item_data["stream_time"]
        self.duration_seconds = int(item_data["stream_length"])
        if not re.match(r"^\d{2}:\d{2}:\d{2}(?:\.\d+)?", self.duration):
            self.duration = "0:" + self.duration  # Add hour part

        self.mimetype = "audio/mpeg"

        self.item = None

    def get_id(self):
        return self.storage_id

    def get_item(self):
        if self.item is None:
            upnp_id = self.get_id()
            upnp_parent_id = self.parent.get_id()
            self.debug("get_item %s %s %s", upnp_id, upnp_parent_id, self.name)
            item = DIDLLite.MusicTrack(upnp_id, upnp_parent_id, self.name)
            item.restricted = True
            item.name = self.name
            item.originalTrackNumber = self.originalTrackNumber
            item.title = self.title
            item.artist = self.artist
            item.album = self.album
            item.albumArtURI = self.cover
            item.duration = self.duration

            proxied_url = "%s%s" % (self.store.urlbase, self.get_id())
            proxied_url = proxied_url.encode("utf-8")
            self.url = proxied_url
            self.location = MoeFmProxyStream(self.item_data["url"], self)

            protocol = "http-get"

            res = Resource(
                proxied_url,
                ("%s:*:%s:*" % (protocol, self.mimetype)).encode("utf-8")
            )
            res.size = self.item_data["file_size"] * 1024
            res.duration = self.duration
            item.res.append(res)

            self.item = item

        return self.item

    def get_url(self):
        return self.url


class MoeFmTrackContainer(Container):
    logCategory = "moefm_track_container"
    ContainerClass = DIDLLite.PlaylistContainer
    preferred_id = None

    def __init__(self, store, parent, title, api_params=None):
        super(MoeFmTrackContainer, self).__init__(parent, title)
        self.sorting_method = lambda x, y: cmp(x.sort_key, y.sort_key)
        self.store = store
        self.api_params = api_params if api_params is not None else {}
        self.loaded = False
        if self.preferred_id:
            self.storage_id = self.preferred_id
            if store.get_by_id(self.storage_id):
                self.storage_id += "$" + parent.get_id()

    def get_item(self):
        if not self.loaded:
            return self.load_tracks().addCallback(lambda _: self.get_item())

        if self.item is None:
            self.item = self.ContainerClass(
                self.storage_id, self.parent_id, self.name
            )

        self.item.childCount = self.get_child_count()
        return self.item

    def get_children(self, *args, **kwargs):
        if not self.loaded:
            return self.load_tracks().addCallback(
                lambda _: self.get_children(*args, **kwargs)
            )

        return super(MoeFmTrackContainer, self).get_children(*args, **kwargs)

    def get_api_params(self):
        return self.api_params

    def on_got_response(self, resp_container):
        self.info("Got response")
        resp = resp_container["response"]
        self.debug("Information: %r", resp["information"])
        if resp["information"]["has_error"]:
            self.error("Got error response: %s" % resp)
            return

        items = []
        existing_ids = set(x.sub_id for x in self.children)
        for item_data in resp["playlist"]:
            item = MoeFmTrack(item_data, self)
            if item.sub_id in existing_ids:
                continue

            items.append(item)
            self.add_child(item)

        self.on_update_completed()
        self.loaded = True
        return items

    def on_got_error(self, error):
        self.warning("Unable to retrieve tracks: %s", error)
        return error

    def load_tracks(self):
        params = {"perpage": settings.get("tracks_per_request", 30)}
        params.update(self.get_api_params())
        d = api.moefm.get("/listen/playlist?api=json", params)
        return d.addCallbacks(self.on_got_response, self.on_got_error)

    def on_update_completed(self):
        self.update_id += 1
        self.store.on_update_completed(self)

    def on_item_play(self, item):
        pass


class MoeFmMultiPageTrackContainer(MoeFmTrackContainer):
    def __init__(self, *args, **kwargs):
        super(MoeFmMultiPageTrackContainer, self).__init__(*args, **kwargs)
        self.current_page = 1

    @property
    def should_load_next_page(self):
        return True

    def load_tracks(self):
        def on_completed(items):
            if items:
                self.current_page += 1

            if items and self.should_load_next_page:
                return self.load_tracks().addCallback(
                    lambda x: itertools.chain(x, items)
                )
            else:
                return items

        d = super(MoeFmMultiPageTrackContainer, self).load_tracks()
        return d.addCallback(on_completed)

    def get_api_params(self):
        params = super(MoeFmMultiPageTrackContainer, self).get_api_params()
        params["page"] = self.current_page
        return params


class MoeFmRandomPlaylist(MoeFmMultiPageTrackContainer):
    preferred_id = "magic"

    def __init__(self, store, parent):
        super(MoeFmRandomPlaylist, self).__init__(store, parent, "Magic")

    def remove_child(self, child, external_id=None, update=True):
        try:
            self.children.remove(child)
            # We'd like the item to be accessible even after removing it
            # self.store.remove_item(child)
        except ValueError:
            pass
        else:
            if update:
                self.update_id += 1

    @property
    def need_more_tracks(self):
        current_count = self.get_child_count()
        return current_count < settings.get("min_tracks_in_playlist", 120)

    @property
    def should_load_next_page(self):
        return self.need_more_tracks

    @property
    def loaded(self):
        return not self.need_more_tracks

    loaded = loaded.setter(lambda self, value: None)

    def on_item_play(self, item):
        self.remove_child(item)
        self.on_update_completed()


class MoeFmPlaylistStore(AbstractBackendStore):
    logCategory = "moefm"
    name = "Moe FM"
    implements = ["MediaServer"]
    wmc_mapping = {"16": 1000}

    def __init__(self, server, **kwargs):
        AbstractBackendStore.__init__(self, server, **kwargs)

        self.init_completed()

    def __repr__(self):
        return self.__class__.__name__

    def append_item(self, item, storage_id=None):
        if storage_id is None:
            storage_id = item.get_id()

        if storage_id is None:
            storage_id = self.getnextID()

        storage_id = str(storage_id)
        return super(MoeFmPlaylistStore, self).append_item(item, storage_id)

    def get_by_id(self, id):
        self.info("get_by_id: %r", id)
        if isinstance(id, basestring):
            id = id.split("@", 1)
            id = id[0].split(".")[0]

        return self.store.get(str(id))

    def upnp_init(self):
        self.current_connection_id = None
        self.server.connection_manager_server.set_variable(
            0,
            "SourceProtocolInfo",
            ["http-get:*:audio/mpeg:*"],
            default=True,
        )

        root_item = Container(None, "Moe FM")
        self.root_item = root_item
        self.set_root_item(root_item)
        root_item.add_child(MoeFmRandomPlaylist(self, root_item))

        fav_tracks = MoeFmMultiPageTrackContainer(
            self, root_item, "Favorite tracks", api_params={"fav": "song"}
        )
        fav_tracks.storage_id = "fav-tracks"
        root_item.add_child(fav_tracks)

    def on_update_completed(self, container):
        self.update_id += 1
        try:
            self.server.content_directory_server.set_variable(
                0, "SystemUpdateID", self.update_id,
            )
            value = (container.get_id(), container.get_update_id())
            self.info("on_update_completed %s %s", self.update_id, value)
            self.server.content_directory_server.set_variable(
                0, "ContainerUpdateIDs", value,
            )
        except Exception as e:
            self.warning("%r", e)


if __name__ == '__main__':
    from twisted.internet import reactor

    def main():
        from coherence.base import Coherence, Plugins
        Plugins().set("MoeFmPlaylistStore", MoeFmPlaylistStore)
        conf = dict(settings.get("coherence_config", {}))
        conf.update({
            "plugin": [{"backend": "MoeFmPlaylistStore"}]
        })
        Coherence(conf)

    reactor.callWhenRunning(main)
    reactor.run()
