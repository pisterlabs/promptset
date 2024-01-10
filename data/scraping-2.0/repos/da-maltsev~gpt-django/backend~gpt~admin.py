from django.contrib import admin
from gpt.models import OpenAiProfile, Reply


@admin.register(Reply)
class ReplyAdmin(admin.ModelAdmin):
    list_display = ("get_question", "author", "status")
    list_filter = ("status", "author")
    search_fields = ("question", "answer")
    ordering = ("-created",)
    fieldsets = ((None, {"fields": ("question", "answer", "previous_reply", "author", "status")}),)
    readonly_fields = ("previous_reply",)

    def get_question(self, obj: Reply) -> str:
        offset = 30
        return obj.question if len(obj.question) <= offset else f"{obj.question[:offset]}..."

    get_question.short_description = "question"  # type: ignore


@admin.register(OpenAiProfile)
class OpenAiProfileAdmin(admin.ModelAdmin):
    list_display = ("uuid", "token", "usage_count", "status")
    list_filter = ("status",)
    ordering = ("-created",)
    search_fields = ("uuid", "token", "comment")
