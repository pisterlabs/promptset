#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the MIT license
# http://opensource.org/licenses/mit-license.php

# Copyright 2010 - Philippe Normand <phil@base-art.net>

import os
import gettext

from mirabeau.maemo.constants import *
from mirabeau.maemo.main import MainWindow

from coherence.base import Coherence
from coherence.extern.simple_config import Config, XmlDictObject

_ = gettext.gettext

class MirabeauController(object):
    coherence_started_cb = lambda: None

    def start(self):
        # TODO: select correct window class depending on platform
        main_window = MainWindow(self)
        main_window.show_all()

    def load_config(self):
        self.first_run = False
        if not os.path.exists(BASEDIR):
            os.makedirs(BASEDIR)
        if not os.path.exists(CONFIG_PATH):
            self.first_run = True

            default_account = ''
            vars = locals()
            vars["MR_UUID"] = MR_UUID
            cfg = DEFAULT_CONFIG % vars
            fd = open(CONFIG_PATH, "w")
            fd.write(cfg)
            fd.close()

        self.config = Config(CONFIG_PATH, root='config', element2attr_mappings={'active':'active'})

    def reload_config(self):
        self.config.save()
        self.load_config()

    def enable_mirabeau(self):
        self.config.set("enable_mirabeau", "yes")
        self.reload_config()

    def disable_mirabeau(self):
        self.config.set("enable_mirabeau", "no")
        self.reload_config()

    def platform_media_directories(self):
        candidates = ["~/MyDocs/.images", "~/MyDocs/.sounds", "~/MyDocs/.videos",
                      "~/MyDocs/DCIM", "~/MyDocs/Music", "~/MyDocs/Videos",
                      ]
        expanded = [os.path.expanduser(c) for c in candidates]
        dirs = [c for c in expanded if os.path.isdir(c)]
        return dirs

    def enable_media_server(self, nickname=None):
        nickname = nickname or "N900"
        name_template = _("%(nickname)s Media Files")

        def generate_cfg(nickname):
            directories = self.platform_media_directories()
            name = name_template % locals()
            opts = dict(uuid=MS_UUID, name=name, content=",".join(directories),
                        backend="FSStore", active="yes")
            return XmlDictObject(initdict=opts)

        plugins = self.config.get("plugin")
        if not plugins:
            self.config.set("plugin", generate_cfg())
        else:
            if isinstance(plugins, XmlDictObject):
                plugins = [plugins,]
            already_in_config = False
            for plugin in plugins:
                if plugin.get("uuid") == MS_UUID:
                    plugin.active = "yes"
                    plugin.name = name_template % locals()
                    already_in_config = True
                    break
            if not already_in_config:
                plugins.append(generate_cfg(nickname))
            self.config.set("plugin", plugins)
        self.reload_config()

    def disable_media_server(self):
        plugins = self.config.get("plugin")
        if plugins:
            if isinstance(plugins, XmlDictObject):
                plugins = [plugins,]
            for plugin in plugins:
                if plugin.get("uuid") == MS_UUID:
                    plugin.active = "no"
                    break
            self.config.set("plugin", plugins)
            self.reload_config()

    def media_server_enabled(self):
        plugins = self.config.get("plugin")
        if plugins:
            if isinstance(plugins, XmlDictObject):
                plugins = [plugins,]
            for plugin in plugins:
                if plugin.get("uuid") == MS_UUID and \
                   plugin.active == "yes":
                    return True
        return False

    def set_media_renderer_name(self, nickname=None):
        nickname = nickname or "N900"
        name_template = _("%(nickname)s Media Renderer")
        plugins = self.config.get("plugin")
        if plugins:
            if isinstance(plugins, XmlDictObject):
                plugins = [plugins,]
            for plugin in plugins:
                if plugin.get("uuid") == MR_UUID:
                    plugin.name = name_template % locals()
                    break
            self.config.set("plugin", plugins)
            self.reload_config()

    def start_coherence(self, restart=False):
        def start():
            if self.config.get("mirabeau").get("account"):
                self.enable_mirabeau()
            else:
                self.disable_mirabeau()
            self.coherence_instance = Coherence(self.config.config)

        if restart:
            if self.coherence_instance:
                dfr = self.stop_coherence()
                dfr.addCallback(lambda result: start())
                return dfr
            else:
               start()
        else:
            start()
        if self.coherence_instance:
            self.coherence_started_cb()

    def stop_coherence(self):
        def stopped(result):
            if self.coherence_instance:
                self.coherence_instance.clear()
                self.coherence_instance = None

        dfr = self.coherence_instance.shutdown(force=True)
        dfr.addBoth(stopped)
        return dfr

    def toggle_coherence(self):
        if self.coherence_instance:
            self.stop_coherence()
        else:
            self.start_coherence()

    def update_settings(self, chatroom, conf_server, account, account_nickname,
                        media_server_enabled):
        mirabeau_section = self.config.get("mirabeau")
        mirabeau_section.set("chatroom", chatroom)
        mirabeau_section.set("conference-server", conf_server)
        mirabeau_section.set("account", account)
        self.config.set("mirabeau", mirabeau_section)
        self.reload_config()

        nickname = account_nickname
        self.set_media_renderer_name(nickname)
        if media_server_enabled:
            self.enable_media_server(nickname=nickname)
        else:
            self.disable_media_server()
        self.start_coherence(restart=True)
