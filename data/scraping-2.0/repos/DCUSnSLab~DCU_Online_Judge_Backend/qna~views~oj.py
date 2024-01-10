from account.decorators import ensure_qna_access
from lecture.models import Lecture, ta_admin_class
from utils.api import APIView
from ..models import Post, Comment
from problem.models import Problem
from django.db.models import Q
from contest.models import Contest
from submission.models import Submission
from ..serializers import PostListSerializer, PostDetailSerializer, CommentSerializer, PostListPushSerializer
import openai
import os

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
openai.api_key=OPENAI_API_KEY
'''
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    date_posted = models.DateTimeField(default=timezone.now)
    content = models.TextField()
    author = models.CharField(max_length=200)
'''
class CommentAPI(APIView):
    def get(self, request):
        questionID = request.GET.get("questionID")
        offset = int(request.GET.get("offset", "0"))

        if questionID:
            question = Post.objects.get(id=questionID)
            comment = Comment.objects.filter(post=question).order_by("date_posted")
            ensure_qna_access(question, request.user)
            offset = int(request.GET.get("offset", "0"))
            limit = int(request.GET.get("limit", "10"))
            if offset == -1:
                import math
                page = math.ceil(comment.count() / limit)
                return self.success(page)
            return self.success(self.paginate_data(request, comment, CommentSerializer))

    def post(self, request):
        data = request.data
        questionID = data['questionID']
        comment = data['comment']

        if questionID:
            question = Post.objects.get(id=questionID)
            #ensure_qna_access(question, request.user)
            comment = Comment.objects.create(post=question, content=comment, author=request.user)
            if request.user == question.author:
                question.proceeding = True
            else:
                question.proceeding = False
            question.save()
            return self.success(CommentSerializer(comment).data)

    def delete(self, request):
        comment_id = request.GET.get("id")
        if comment_id:
            comment = Comment.objects.get(id=comment_id)
            if comment.author == request.user or request.user.is_super_admin():
                comment.delete()
            elif comment.post.contest is not None:
                if comment.post.contest.lecture.created_by == request.user:
                    comment.delete()
                else:
                    return self.error("작성 본인 또는 관리자만 삭제할 수 있습니다.")
            else:
                return self.error("작성 본인 또는 관리자만 삭제할 수 있습니다.")
        return self.success()



class QnAPostDetailAPI(APIView):
    def post(self, request):
        data = request.data
        questionID = data['questionID']

        if questionID:
            question = Post.objects.get(id=questionID)
            ensure_qna_access(question, request.user)
            question.solved = not question.solved
            question.save()
            return self.success(question.solved)

    def put(self, request):
        questionID = request.GET.get("questionID")
        OpenQnA = request.GET.get("OpenQnA")
        if questionID:
            question = Post.objects.get(id=questionID)
            if request.user.is_admin() or request.user.is_super_admin():
                if OpenQnA == 'true':
                    question.private = False
                else:
                    question.private = True
                question.save()
                return self.success()
        return self.error()

    def get(self, request):
        questionID = request.GET.get("questionID")

        if questionID:
            question = Post.objects.get(id=questionID)
            #ensure_qna_access(question, request.user)
            return self.success(PostDetailSerializer(question).data)

    def delete(self, request):
        questionID = request.GET.get("questionID")

        if questionID:
            question = Post.objects.get(id=questionID)
            if question.author == request.user or request.user.is_super_admin():
                question.delete()
            elif question.contest is not None:
                if question.contest.lecture.created_by == request.user:
                    question.delete()
                else:
                    return self.error("작성 본인 또는 관리자만 삭제할 수 있습니다.")
            else:
                return self.error("작성 본인 또는 관리자만 삭제할 수 있습니다.")

        return self.success()

class QnAPostAPI(APIView):
    def put(self, request):
        """
        edit announcement
        """
        data = request.data
        PostList = Post.objects.filter(proceeding=True).exclude(solved=True)

        if request.user.is_super_admin():
            return self.success(self.post_paginate_data(request, PostList, PostListPushSerializer))
        elif request.user.is_semi_admin():

            taAdmin = ta_admin_class.objects.filter(user=request.user)
            Plist = ''

            for ta in taAdmin:
                if Plist == '':
                    Plist = PostList.filter(contest__lecture=ta.lecture)
                else:
                    Plist = Plist.union(PostList.filter(contest__lecture=ta.lecture))

            PostList = Plist.filter(proceeding=True).exclude(solved=True)

        elif request.user.is_admin():
            PostList = Post.objects.filter(contest__lecture__created_by=request.user, proceeding=True).exclude(solved=True)

        elif request.user.is_student():
            PostList = Post.objects.filter(author=request.user, proceeding=False).exclude(solved=True)

        return self.success(self.post_paginate_data(request, PostList, PostListPushSerializer))

    def post(self, request):
        data = request.data
        try:
            contest = Contest.objects.get(id=data['contestID'])
            problem = ''
            try:
                problem = Problem.objects.get(id=data['problemID'])
            except:
                print(data['problemID'])
                problem = Problem.objects.get(contest=contest, _id=data['problemID'])
            submission = Submission.objects.get(id=data['id'])
            qna = Post.objects.create(title=data['content']['title'], content=data['content']['content'], author=request.user, submission=submission, problem=problem, contest=contest)
        except:
            private = data['private']
            if not private:
                qna = Post.objects.create(title=data['content']['title'], content=data['content']['content'], author=request.user, private=False)
            else:
                qna = Post.objects.create(title=data['content']['title'], content=data['content']['content'], author=request.user)

        return self.success(PostListSerializer(qna).data)

    def get(self, request):
        lectureID = request.GET.get("LectureID")
        allQuestion = request.GET.get("all")
        problemID = request.GET.get("problemID")

        if allQuestion == 'all':
            visible = False if (request.GET.get("visible") == 'false') else True
            PostList = Post.objects.filter(solved=visible, contest=None, problem=None, private=False).order_by("-date_posted")

            return self.success(self.paginate_data(request, PostList, PostListSerializer))

        elif problemID:
            lecture = Lecture.objects.get(id=lectureID)
            problem = Problem.objects.get(id=problemID)
            PostList = Post.objects.filter(contest__lecture=lecture, problem=problem, private=False).order_by("-date_posted")

            return self.success(self.paginate_data(request, PostList, PostListSerializer))

        elif lectureID:
            lecture = Lecture.objects.get(id=lectureID)
            visible = False if (request.GET.get("visible") == 'false') else True
            PostList = Post.objects.filter(contest__lecture=lecture, solved=visible).order_by("-date_posted")
            if request.user.is_admin() or request.user.is_super_admin():
                return self.success(self.paginate_data(request, PostList, PostListSerializer))
            elif request.user.is_semi_admin():
                if ta_admin_class.objects.filter(Q(lecture=lecture, user=request.user)):
                    return self.success(self.paginate_data(request, PostList, PostListSerializer))

            PostList = PostList.filter(Q(author=request.user) | Q(private=False)).order_by("-date_posted")
            return self.success(self.paginate_data(request, PostList, PostListSerializer))

        else:
            if request.user.is_super_admin():
                lecture = Lecture.objects.get(id=lectureID)
                visible = False if (request.GET.get("visible") == 'false') else True
                PostList = Post.objects.filter(contest__lecture=lecture, solved=visible).order_by("-date_posted")
                return self.success(self.paginate_data(request, PostList, PostListSerializer))

class AIhelperAPI(APIView):
    def get(self, request):
        # get code form submission data
        # print('AIhelperAPI called')
        result = request.GET.get("result")
        if result == '0':
            code = request.GET.get("code") + "\n이 코드를 최적화하고 가독성을 높여줘."
        elif result == '8':
            code = request.GET.get("code") + "\n이 다음 코드의 동작을 분석해서 논리 오류를 알려줘."
        else:
            code = request.GET.get("code") + "\n이 코드에서 오류를 찾아서 알려줘."

        messages = [{"role": "user", "content": code}]
        # send chatGPT and get answer
        assistant_content = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        messages.append({"role": "assistant", "content": f"{assistant_content}"})
        # delete code in answer when not
        if result != '0':
            code_deleted_response=assistant_content.choices[0].message.content[:assistant_content.choices[0].message.content.find("```")]
            code_deleted_response=code_deleted_response[:code_deleted_response.find("corrected code")] + "정답 코드는 보이지 않습니다."
        else:
            code_deleted_response=assistant_content.choices[0].message.content
        return self.success(code_deleted_response)
