import os

from baserow.core.registries import plugin_registry
from django.apps import AppConfig


def install_argos_translate_package(from_code, to_code):
    import argostranslate.package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages)
    )
    argostranslate.package.install_from_path(package_to_install.download())        

class BaserowTranslatePluginDjangoAppConfig(AppConfig):
    name = "baserow_translate_plugin"

    def ready(self):
        # install argostranslate language packs. they need to be installed by the user id running baserow,
        # as their data will be stored in $HOME/.local/share/argos-translate/
        install_argos_translate_package('en', 'fr')
        install_argos_translate_package('fr', 'en')

        # configure OpenAI
        openai_api_key = os.environ.get('OPENAI_API_KEY', '')
        if openai_api_key:
            import openai
            openai.api_key = openai_api_key

        from .plugins import BaserowTranslatePlugin

        plugin_registry.register(BaserowTranslatePlugin())

        # register our new field type
        from baserow.contrib.database.fields.registries import field_type_registry
        from .field_types import TranslationFieldType, ChatGPTFieldType

        field_type_registry.register(TranslationFieldType())
        field_type_registry.register(ChatGPTFieldType())
