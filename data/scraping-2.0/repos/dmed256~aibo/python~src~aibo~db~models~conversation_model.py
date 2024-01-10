import datetime as dt
from typing import Optional, Self
from uuid import UUID, uuid4

import sqlalchemy as sa
import sqlalchemy.orm as orm

from aibo.common.chat.message_source import OpenAIModelSource
from aibo.common.time import now_utc
from aibo.db.client import get_session
from aibo.db.models.base_db_model import BaseDBModel
from aibo.db.models.custom_types import (
    OpenAIModelSourceColumn,
    StrListColumn,
    UUIDColumn,
)
from aibo.db.models.message_model import MessageModel


class ConversationModel(BaseDBModel):
    __tablename__ = "conversations"
    __table_args__ = (
        sa.Index("conversations_idx_trace_id", "trace_id"),
        sa.Index("conversations_idx_created_at", "created_at"),
    )

    id: orm.Mapped[UUID] = orm.mapped_column(
        UUIDColumn, primary_key=True, default=uuid4
    )
    trace_id: orm.Mapped[UUID] = orm.mapped_column(UUIDColumn)
    title: orm.Mapped[str]
    openai_model_source: orm.Mapped[OpenAIModelSource] = orm.mapped_column(
        OpenAIModelSourceColumn
    )
    enabled_package_names: orm.Mapped[list[str]] = orm.mapped_column(StrListColumn)

    # The columns below are only optional since we need to first create the
    # conversation, and then the messages below
    #
    # In practice, we make both objects at the same time though
    root_message_id: orm.Mapped[Optional[UUID]] = orm.mapped_column(
        UUIDColumn,
        sa.ForeignKey("messages.id"),
        nullable=True,
    )
    current_message_id: orm.Mapped[Optional[UUID]] = orm.mapped_column(
        UUIDColumn,
        sa.ForeignKey("messages.id"),
        nullable=True,
    )

    # If the conversation spun off another conversation, track
    # which conversation and message in it
    origin_message_id: orm.Mapped[Optional[UUID]] = orm.mapped_column(
        UUIDColumn,
        sa.ForeignKey("messages.id"),
        nullable=True,
    )
    conversation_depth: orm.Mapped[int] = orm.mapped_column(default=0)

    created_at: orm.Mapped[dt.datetime] = orm.mapped_column(default=now_utc)
    deleted_at: orm.Mapped[Optional[dt.datetime]]

    async def soft_delete(self) -> Self:
        if self.deleted_at is None:
            self.deleted_at = now_utc()
            async with get_session() as session:
                session.add(self)
                await session.commit()

        return self

    @staticmethod
    async def get_uncached_messages(
        id: UUID, *, include_deletions: bool = False
    ) -> list[MessageModel]:
        async with get_session() as session:
            query = sa.select(MessageModel).where(MessageModel.conversation_id == id)
            if not include_deletions:
                query = query.where(MessageModel.deleted_at == None)

            query_result = await session.execute(query)

        return list(query_result.scalars().all())
