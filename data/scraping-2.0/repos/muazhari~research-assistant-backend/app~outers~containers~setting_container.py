from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer

from app.outers.settings.datastore_one_setting import DatastoreOneSetting
from app.outers.settings.datastore_two_setting import DatastoreTwoSetting
from app.outers.settings.openai_setting import OpenAiSetting
from app.outers.settings.temp_persistence_setting import TempPersistenceSetting


class SettingContainer(DeclarativeContainer):
    datastore_one: DatastoreOneSetting = providers.Singleton(DatastoreOneSetting)
    datastore_two: DatastoreTwoSetting = providers.Singleton(DatastoreTwoSetting)
    openai: OpenAiSetting = providers.Singleton(OpenAiSetting)
    temp_persistence: TempPersistenceSetting = providers.Singleton(TempPersistenceSetting)
