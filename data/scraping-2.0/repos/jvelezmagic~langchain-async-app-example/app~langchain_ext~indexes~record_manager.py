from __future__ import annotations

import contextlib
import decimal
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Union

# from langchain.indexes import SQLRecordManager, index
from sqlalchemy import (
    URL,
    Column,
    Float,
    Index,
    String,
    UniqueConstraint,
    and_,
    delete,
    select,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.ext.declarative import declarative_base

NAMESPACE_UUID = uuid.UUID(int=1984)


class RecordManagerAsync(ABC):
    """An abstract base class representing the interface for an async record manager."""

    def __init__(
        self,
        namespace: str,
    ) -> None:
        """Initialize the record manager.

        Args:
            namespace (str): The namespace for the record manager.
        """
        self.namespace = namespace

    @abstractmethod
    async def create_schema(self) -> None:
        """Create the database schema for the record manager."""

    @abstractmethod
    async def get_time(self) -> float:
        """Get the current server time as a high resolution timestamp!

        It's important to get this from the server to ensure a monotonic clock,
        otherwise there may be data loss when cleaning up old documents!

        Returns:
            The current server time as a float timestamp.
        """

    @abstractmethod
    async def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
            time_at_least: if provided, updates should only happen if the
              updated_at field is at least this time.

        Raises:
            ValueError: If the length of keys doesn't match the length of group_ids.
        """

    @abstractmethod
    async def exists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        """

    @abstractmethod
    async def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """List records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
            after: Filter to list records updated after this time.
            group_ids: Filter to list records with specific group IDs.

        Returns:
            A list of keys for the matching records.
        """

    @abstractmethod
    async def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        """


Base = declarative_base()


class UpsertionRecord(Base):  # type: ignore[valid-type,misc]
    """Table used to keep track of when a key was last updated."""

    # ATTENTION:
    # Prior to modifying this table, please determine whether
    # we should create migrations for this table to make sure
    # users do not experience data loss.
    __tablename__ = "upsertion_record"

    uuid = Column(
        String,
        index=True,
        default=lambda: str(uuid.uuid4()),
        primary_key=True,
        nullable=False,
    )
    key = Column(String, index=True)
    # Using a non-normalized representation to handle `namespace` attribute.
    # If the need arises, this attribute can be pulled into a separate Collection
    # table at some time later.
    namespace = Column(String, index=True, nullable=False)
    group_id = Column(String, index=True, nullable=True)

    # The timestamp associated with the last record upsertion.
    updated_at = Column(Float, index=True)

    __table_args__ = (
        UniqueConstraint("key", "namespace", name="uix_key_namespace"),
        Index("ix_key_namespace", "key", "namespace"),
    )


class SQLRecordManagerAsync(RecordManagerAsync):
    """A SQL Alchemy based implementation of the async record manager."""

    def __init__(
        self,
        namespace: str,
        *,
        engine: Optional[AsyncEngine] = None,
        db_url: Union[None, str, URL] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the SQLRecordManagerAsync.

        This class serves as a manager persistence layer that uses an SQL
        backend to track upserted records. You should specify either a db_url
        to create an engine or provide an existing engine.

        Args:
            namespace: The namespace associated with this record manager.
            engine: An already existing SQL Alchemy engine.
                Default is None.
            db_url: A database connection string used to create
                an SQL Alchemy engine. Default is None.
            engine_kwargs: Additional keyword arguments
                to be passed when creating the engine. Default is an empty dictionary.

        Raises:
            ValueError: If both db_url and engine are provided or neither.
            AssertionError: If something unexpected happens during engine configuration.
        """
        super().__init__(namespace=namespace)
        if db_url is None and engine is None:
            raise ValueError("Must specify either db_url or engine")
        if db_url is not None and engine is not None:
            raise ValueError("Must specify either db_url or engine, not both")

        if db_url:
            _kwargs = engine_kwargs or {}
            _engine = create_async_engine(db_url, **_kwargs)
        elif engine:
            _engine = engine
        else:
            raise AssertionError("Something went wrong with configuration of engine.")

        self.engine = _engine
        self.dialect = _engine.dialect.name
        self.session_factory = async_sessionmaker(bind=self.engine)

    async def create_schema(self) -> None:
        """Create the database schema."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @contextlib.asynccontextmanager
    async def _make_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Create a session and close it after use."""
        async with self.session_factory() as session:
            try:
                yield session
            finally:
                await session.close()

    async def get_time(self) -> float:
        """Get the current server time as a timestamp.

        Please note it's critical that time is obtained from the server since
        we want a monotonic clock.
        """
        async with self._make_session() as session:
            # * SQLite specific implementation, can be changed based on dialect.
            # * For SQLite, unlike unixepoch it will work with older versions of SQLite.
            # ----
            # julianday('now'): Julian day number for the current date and time.
            # The Julian day is a continuous count of days, starting from a
            # reference date (Julian day number 0).
            # 2440587.5 - constant represents the Julian day number for January 1, 1970
            # 86400.0 - constant represents the number of seconds
            # in a day (24 hours * 60 minutes * 60 seconds)
            if self.dialect == "sqlite":
                query = text("SELECT (julianday('now') - 2440587.5) * 86400.0;")
            elif self.dialect == "postgresql":
                query = text("SELECT EXTRACT (EPOCH FROM CURRENT_TIMESTAMP);")
            else:
                raise NotImplementedError(f"Not implemented for dialect {self.dialect}")

            dt = await session.execute(query)
            dt = dt.scalar()
            if isinstance(dt, decimal.Decimal):
                dt = float(dt)
            if not isinstance(dt, float):
                raise AssertionError(f"Unexpected type for datetime: {type(dt)}")
            return dt

    async def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert records into the SQLite database."""
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            raise ValueError(
                f"Number of keys ({len(keys)}) does not match number of "
                f"group_ids ({len(group_ids)})"
            )

        # Get the current time from the server.
        # This makes an extra round trip to the server, should not be a big deal

        # if the batch size is large enough.
        # Getting the time here helps us compare it against the time_at_least
        # and raise an error if there is a time sync issue.
        # Here, we're just being extra careful to minimize the chance of
        # data loss due to incorrectly deleting records.
        update_time = await self.get_time()

        if time_at_least and update_time < time_at_least:
            # Safeguard against time sync issues
            raise AssertionError(f"Time sync issue: {update_time} < {time_at_least}")

        records_to_upsert = [
            {
                "key": key,
                "namespace": self.namespace,
                "updated_at": update_time,
                "group_id": group_id,
            }
            for key, group_id in zip(keys, group_ids)
        ]

        async with self._make_session() as session:
            if self.dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                # Note: uses SQLite insert to make on_conflict_do_update work.
                # This code needs to be generalized a bit to work with more dialects.
                insert_stmt = sqlite_insert(UpsertionRecord).values(records_to_upsert)
                stmt = insert_stmt.on_conflict_do_update(  # type: ignore[attr-defined]
                    [UpsertionRecord.key, UpsertionRecord.namespace],
                    set_=dict(
                        # attr-defined type ignore
                        updated_at=insert_stmt.excluded.updated_at,  # type: ignore
                        group_id=insert_stmt.excluded.group_id,  # type: ignore
                    ),
                )
            elif self.dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                # Note: uses SQLite insert to make on_conflict_do_update work.
                # This code needs to be generalized a bit to work with more dialects.
                insert_stmt = pg_insert(UpsertionRecord).values(records_to_upsert)
                stmt = insert_stmt.on_conflict_do_update(  # type: ignore[attr-defined]
                    "uix_key_namespace",  # Name of constraint
                    set_=dict(
                        # attr-defined type ignore
                        updated_at=insert_stmt.excluded.updated_at,  # type: ignore
                        group_id=insert_stmt.excluded.group_id,  # type: ignore
                    ),
                )
            else:
                raise NotImplementedError(f"Unsupported dialect {self.dialect}")

            await session.execute(stmt)
            await session.commit()

    async def exists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the given keys exist in the SQLite database."""
        async with self._make_session() as session:
            records = (
                await session.execute(
                    select(UpsertionRecord.key).where(
                        and_(
                            UpsertionRecord.key.in_(keys),
                            UpsertionRecord.namespace == self.namespace,
                        )
                    )
                )
            ).all()
        found_keys = set(r.key for r in records)
        return [k in found_keys for k in keys]

    async def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """List records in the SQLite database based on the provided date range."""
        async with self._make_session() as session:
            query = select(UpsertionRecord).filter(
                UpsertionRecord.namespace == self.namespace
            )

            # mypy does not recognize .all() or .filter()
            if after:
                query = query.filter(  # type: ignore[attr-defined]
                    UpsertionRecord.updated_at > after
                )
            if before:
                query = query.filter(  # type: ignore[attr-defined]
                    UpsertionRecord.updated_at < before
                )
            if group_ids:
                query = query.filter(  # type: ignore[attr-defined]
                    UpsertionRecord.group_id.in_(group_ids)
                )
            records = (await session.execute(query)).all()  # type: ignore[attr-defined]
            return [r[0].key for r in records]

    async def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete records from the SQLite database."""
        async with self._make_session() as session:
            await session.execute(
                delete(UpsertionRecord).filter(
                    and_(
                        UpsertionRecord.key.in_(keys),
                        UpsertionRecord.namespace == self.namespace,
                    )
                )
            )
            await session.commit()
