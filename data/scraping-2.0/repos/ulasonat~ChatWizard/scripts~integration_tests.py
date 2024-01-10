import discord
from unittest.mock import MagicMock, patch
from discord_bot import DiscordBot
from openai_handler import OpenAIHandler


class TestIntegration:
    def setup(self):
        self.grammar_prompt_path = "../prompts/grammar.txt"
        self.mock_openai_handler = MagicMock(spec=OpenAIHandler)
        self.intents = discord.Intents.default()
        self.intents.members = True
        self.intents.message_content = True

        self.bot = DiscordBot(
            intents=self.intents,
            openai_handler=self.mock_openai_handler,
            log_file_path="test_log.txt",
            user_scores_path="test_user_scores.json",
        )
        self.test_server = MagicMock(spec=discord.Guild)
        self.test_user = MagicMock(spec=discord.Member)

    @patch("discord.Message")
    async def test_on_message(self, mock_discord_message):
        mock_discord_message.content = "Hello there!"
        mock_discord_message.author = self.test_user
        self.mock_openai_handler.get_message_score.return_value = {"grammar": 10}

        await self.bot.on_message(mock_discord_message)

        assert self.bot.last_message is not None
        self.mock_openai_handler.get_message_score.assert_called_once_with(mock_discord_message.content)

    async def test_update_scores(self):
        self.bot.user_scores = {}
        self.mock_openai_handler.generate_default_scores.return_value = {"grammar": 50}

        self.bot.update_scores(self.test_user.id)

        assert self.bot.user_scores[self.test_user.id] == {"grammar": 50}
        self.mock_openai_handler.generate_default_scores.assert_called_once()

    def test_load_user_scores(self):
        test_user_scores = {"test_user_id": {"grammar": 30}}
        with patch("json.load", return_value=test_user_scores):
            self.bot.user_scores = self.bot.load_user_scores()

        assert self.bot.user_scores == test_user_scores

    def test_save_user_scores(self):
        test_user_scores = {"test_user_id": {"grammar": 30}}
        with patch("json.dump") as mock_dump:
            self.bot.user_scores = test_user_scores
            self.bot.save_user_scores()

        mock_dump.assert_called_once_with(test_user_scores, MagicMock())

    def test_update_log_file(self):
        test_nickname = "test_user"
        test_content = "test message"

        with patch("os.path.exists", return_value=False), patch("builtins.open", create=True), patch(
            "builtins.print"
        ) as mock_print:
            self.bot.update_log_file(test_nickname, test_content)

        mock_print.assert_called_once_with("Log file created.\n")

        with patch("os.path.exists", return_value=True), patch("builtins.open", create=True), patch(
            "builtins.print"
        ) as mock_print:
            self.bot.update_log_file(test_nickname, test_content)

        mock_print.assert_not_called()
