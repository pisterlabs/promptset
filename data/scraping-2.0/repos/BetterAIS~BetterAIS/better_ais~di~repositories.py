from functools import cached_property
from better_ais.config.core import CoreSettings
from better_ais.config.openai import OpenAiSettings
from better_ais.config.ldap import LdapSettings
from better_ais.repositories.http.ais.repo import AISRepository
from better_ais.repositories.http.ais.source import FakeAISClient
from better_ais.repositories.postgres.users.repo import UserRepository
from better_ais.repositories.postgres.mails.repo import MailRepository
from better_ais.repositories.postgres.documents.repo import DocumentRepository

from better_ais.repositories.ldap.stu import LdapRepository
from better_ais.repositories.http.accommodation import AccommodationRepository

class RepositoriesContainer:
    def __init__(self, core: CoreSettings, openai: OpenAiSettings, ldap_settings: LdapSettings):
        self._core = core
        self._openai = openai
        self._ldap_settings = ldap_settings

    @cached_property
    def ais_repository(self) -> AISRepository:
        return AISRepository(FakeAISClient(self._openai))

    @cached_property
    def pg_users_repository(self) -> UserRepository:
        return UserRepository()

    @cached_property
    def pg_mails_repository(self) -> MailRepository:
        return MailRepository()

    @cached_property
    def pg_documents_repository(self) -> DocumentRepository:
        return DocumentRepository()
    
    @cached_property
    def ldap_repository(self) -> LdapRepository:
        return LdapRepository(self._ldap_settings)

    @cached_property
    def accommodation_repository(self) -> AccommodationRepository:
        return AccommodationRepository()

    