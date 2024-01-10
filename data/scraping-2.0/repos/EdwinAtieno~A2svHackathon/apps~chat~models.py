import uuid
from typing import Optional, List
from django.db import models
from django.utils import timezone
from apps.users.models import User
from apps.chat.openai_service import OpenAIService
import re
import logging


class ChatSession(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="chat_sessions"
    )
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    conversation_history = models.JSONField(default=list)
    created_at = models.DateTimeField(default=timezone.now)
    last_activity = models.DateTimeField(default=timezone.now)

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the conversation history."""
        message = {"role": role, "content": content}
        self.conversation_history.append(message)
        self.save()

    def generate_response(
        self, user_prompt: str, openai_service: OpenAIService
    ) -> Optional[str]:
        """Generates a response to the given user prompt using the GPT-3 model."""
        is_finance_related = self.is_finance_related(user_prompt)

        if not is_finance_related:
            return "Hello, I am your financial assistant. If you have any finance-related questions, feel free to ask."

        context = self.get_conversation_context()

        if "user_goals" not in context:
            context["user_goals"] = self.extract_goals_from_prompt(user_prompt)

        system_message = {
            "role": "system",
            "content": f"You are a proficient financial advisor with expertise in financial matters and risk assessment in Kenya. "
            f"Leverage your skills to respond to customer financial questions. "
            f"Generate tailored recommendations to enhance the customer's financial well-being. "
            f"Provide insightful advice and guidance based on your financial expertise "
            f"and understanding of the unique needs and goals of the customer. "
            f"Do not respond to any question outside finance and banking. "
            f"You are also proficient in data governance, AML, and compliance. "
            f"Listen and understand user-specific goals & priorities. You have mentioned goals related to {', '.join(context.get('user_goals', []))}."
            f"If you have any specific questions or goals in mind, feel free to share, and I'll provide personalized assistance.",
        }

        user_message = {"role": "user", "content": user_prompt}
        model_response = openai_service.generate_chat_response(
            [system_message, user_message]
        )

        # Add user and model messages to conversation history
        self.add_message(user_message["role"], user_message["content"])
        self.add_message("model", model_response)

        return model_response

    def is_finance_related(self, query: str) -> bool:
        finance_keywords = [
            "finance",
            "investment",
            "savings",
            "returns",
            "risk",
            "stocks",
            "bonds",
            "mutual funds",
            "real estate",
            "retirement",
            "income",
            "budgeting",
            "financial freedom",
            "wealth",
            "credit",
            "debt",
            "insurance",
            "tax planning",
            "crypto",
            "401(k)",
            "IRA",
            "net worth",
            "stock market",
            "passive income",
            "robo-advisor",
            "financial planner",
            "investment strategy",
            "mortgage",
            "loan",
            "credit card",
            "money management",
            "financial literacy",
        ]
        return any(keyword in query.lower() for keyword in finance_keywords)

    def extract_goals_from_prompt(self, prompt: str) -> List[str]:
        goals_keywords = [
            "save",
            "invest",
            "plan",
            "budget",
            "education",
            "mortgage",
            "financial freedom",
            "retirement",
            "wealth",
            "savings",
            "income",
            "assets",
            "expenses",
            "emergency fund",
            "debt",
            "real estate",
            "passive income",
            "insurance",
            "tax planning",
        ]
        prompt_tokens = re.findall(r"\b\w+\b", prompt.lower())  # tokenize prompt
        extracted_goals = [goal for goal in goals_keywords if goal in prompt_tokens]

        return extracted_goals

    def get_conversation_context(self) -> dict:
        """Get or initialize the conversation context."""
        if not self.conversation_history:
            return {}

        last_message = self.conversation_history[-1]

        if isinstance(last_message, dict) and "context" in last_message:
            return last_message["context"]
        else:
            return {}

    def save_chat_session(self) -> None:
        """Saves the chat session to the database."""
        self.last_activity = timezone.now()  # update last activity on save
        try:
            self.save()
        except models.Model.DoesNotExist:
            logging.error(f"ChatSession does not exist.")
        except Exception as e:
            logging.error(f"Database error: {e}")

    @classmethod
    def load_chat_session(cls, user: str) -> Optional["ChatSession"]:
        """Load the latest chat session for a user."""
        try:
            chat_sessions = cls.objects.filter(user=user).order_by("-created_at")
            return chat_sessions.first() if chat_sessions.exists() else None
        except cls.DoesNotExist:
            return None
        except Exception as e:
            logging.error(f"Database error: {e}")


class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ("user", "User"),
        ("system", "System"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="chat_messages"
    )
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
        null=True,
        blank=True,
    )
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    timestamp = models.DateTimeField(auto_now_add=True, editable=False)
    user_message = models.TextField()
    model_response = models.TextField(null=True, blank=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default="user")

    def __str__(self):
        return f"{self.get_role_display()} ({self.user}) at {self.timestamp}: {self.user_message}\nResponse: {self.model_response}"

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the conversation using the latest session."""
        user_sessions = ChatSession.objects.filter(user=self.user).order_by(
            "-created_at"
        )

        if user_sessions.exists():
            self.session = user_sessions.first()
            logging.info(f"Using existing session: {self.session.id}")
        else:
            self.session = ChatSession.objects.create(user=self.user)
            logging.info(f"Created new session: {self.session.id}")

        message = {
            "role": role,
            "content": content,
        }
        self.session.conversation_history.append(message)
        self.session.save()

    def save(self, *args, **kwargs):
        """Saves the ChatMessage using the latest session."""
        user_sessions = ChatSession.objects.filter(user=self.user).order_by(
            "-created_at"
        )

        if user_sessions.exists():
            self.session = user_sessions.first()
            logging.info(f"Using existing session: {self.session.id}")
        else:
            self.session = ChatSession.objects.create(user=self.user)
            logging.info(f"Created new session: {self.session.id}")

        super().save(*args, **kwargs)
