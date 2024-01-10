from django.contrib import admin
from django.db.models import QuerySet, Q
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.qdrant import Qdrant

from src.account.models import Platform
from src.config import config
from src.qa.models import HotQuestion


class HotQuestionAdmin(admin.ModelAdmin):
    search_fields = ['region', 'tags', 'standard_question']
    list_display = ('id', 'platform', 'region', 'category_1', 'category_2', 'tags', 'standard_question')
    list_display_links = ('id', 'standard_question')
    list_filter = ('platform', 'region', 'category_1', 'category_2', 'tags')


class PlatformAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'host', 'wx_appid', 'qdrant_collection_name', 'as_default')
    list_display_links = ('id', 'title')

    actions = ['创建qdrant集合']

    def 创建qdrant集合(self, request, queryset: QuerySet[Platform]):
        for platform in queryset:
            questions = HotQuestion.objects.filter(Q(platform__isnull=True) | Q(platform=platform)).all()
            self.message_user(request, f"found {len(questions)} questions")
            if len(questions) == 0:
                continue
            docs = []
            for question in questions:
                docs.append(Document(page_content=question.standard_question, metadata={
                    '地域': question.region,
                    '大类': question.category_1,
                    '小类': question.category_2,
                    '标签': question.tag,
                    'type': 'question',
                    'question_type': 1,
                    'answer': question.standard_answer,
                }))
                if type(question.similar_questions) == str and len(question.similar_questions) > 10:
                    for similar_question in question.similar_questions.split('？ '):
                        docs.append(Document(page_content=similar_question + '?', metadata={
                            '地域': question.region,
                            '大类': question.category_1,
                            '小类': question.category_2,
                            '标签': question.tag,
                            'type': 'question',
                            'question_type': 2,
                            'answer': question.standard_answer,
                        }))
                docs.append(Document(page_content=question.standard_answer, metadata={
                    '地域': question.region,
                    '大类': question.category_1,
                    '小类': question.category_2,
                    '标签': question.tag,
                    'type': 'answer',
                    'question': question.standard_question,
                }))

            self.message_user(request, f"create {len(docs)} docs")
            embeddings = OpenAIEmbeddings()
            qdrant = Qdrant.from_documents(
                docs,
                embeddings,
                force_recreate=True,
                url=config['qdrant']['url'],
                collection_name=platform.qdrant_collection_name,
            )

            self.message_user(request, f"{len(docs)} records imported to {platform.qdrant_collection_name} successfully.")


admin.site.register(HotQuestion, HotQuestionAdmin)
admin.site.register(Platform, PlatformAdmin)
