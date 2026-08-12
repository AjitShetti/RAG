"""Telegram Bot Service for Agentic RAG system."""

import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ...config import settings
from ...schemas.rag import AgenticAskResponse, AskRequest
from ..agents.agentic_rag import run_agentic_rag

logger = logging.getLogger(__name__)


def format_agentic_response(response: AgenticAskResponse) -> str:
    """Format AgenticAskResponse into a clean Markdown message for Telegram."""
    lines: list[str] = []

    # Answer section
    lines.append(f"*Answer:*\n{response.answer}\n")

    # Reasoning Steps section
    if response.reasoning_steps:
        lines.append("*Reasoning Steps:*")
        for step in response.reasoning_steps:
            lines.append(f"• `[{step.node.upper()}]` *{step.decision}*: {step.detail}")
        lines.append("")

    # Sources section
    if response.sources:
        distinct_papers = len(set(s.paper_id for s in response.sources))
        lines.append(f"*Sources ({response.used_chunk_count} chunks from {distinct_papers} papers):*")
        for idx, src in enumerate(response.sources, 1):
            lines.append(
                f"{idx}. [{src.title}]({src.pdf_url})\n"
                f"   _Section:_ {src.section_name} (Relevance: {src.relevance_score:.2f})\n"
                f"   _{src.snippet}_"
            )
        lines.append("")

    # Performance metadata
    lines.append(
        f"_Stats: {response.took_ms:.0f}ms | Chunks: {response.used_chunk_count}/"
        f"{response.retrieved_chunk_count} | Rewrites: {response.rewrite_count} | "
        f"Cached: {response.cached}_"
    )

    return "\n".join(lines)


def split_message(text: str, max_length: int = 4000) -> list[str]:
    """Split response string if > max_length to adhere to Telegram limits."""
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        split_idx = text.rfind("\n", 0, max_length)
        if split_idx == -1 or split_idx < max_length // 2:
            split_idx = max_length
        chunks.append(text[:split_idx].strip())
        text = text[split_idx:].strip()

    return chunks


class TelegramBot:
    """Telegram Bot wrapper utilizing python-telegram-bot ApplicationBuilder."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.application: Application | None = None

    def build(self) -> Application:
        """Build and configure telegram Application with handlers and error handling."""
        app = ApplicationBuilder().token(self.token).build()

        app.add_handler(CommandHandler("start", self._start_handler))
        app.add_handler(CommandHandler("help", self._help_handler))
        app.add_handler(CommandHandler("ask", self._ask_handler))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self._text_handler))

        app.add_error_handler(self._error_handler)
        self.application = app
        return app

    async def _start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command with welcome message and usage guide."""
        welcome_msg = (
            "👋 *Welcome to the arXiv Paper Agentic RAG Bot!*\n\n"
            "I can answer your questions about scientific arXiv papers using an agentic RAG "
            "workflow with guardrails, iterative retrieval, grading, and query rewriting.\n\n"
            "*Commands:*\n"
            "• `/ask <question>` — Ask a specific question\n"
            "• `/help` — Display detailed usage guide\n\n"
            "Or simply send me any plain text question!"
        )
        if update.message:
            await update.message.reply_text(welcome_msg, parse_mode="Markdown")

    async def _help_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command with detailed usage information."""
        help_msg = (
            "ℹ️ *arXiv Paper Agentic RAG Bot Help*\n\n"
            "*Usage:*\n"
            "1. Type `/ask <your question>` or send a message directly.\n"
            "2. The bot will run an Agentic RAG pipeline:\n"
            "   - Safety Guardrail check\n"
            "   - Hybrid Search & Retrieval\n"
            "   - Relevance Grading\n"
            "   - Iterative Query Rewriting (if initial results are poor)\n"
            "   - Grounded LLM Answer Generation\n\n"
            "3. You will receive the final answer along with step-by-step reasoning and "
            "source paper attributions."
        )
        if update.message:
            await update.message.reply_text(help_msg, parse_mode="Markdown")

    async def _ask_question_internal(self, update: Update, query: str) -> None:
        """Internal helper to process a question and send formatted response."""
        if not update.message:
            return

        status_msg = await update.message.reply_text(
            "🤔 *Processing your question with Agentic RAG...*", parse_mode="Markdown"
        )

        try:
            req = AskRequest(query=query)
            response: AgenticAskResponse = await asyncio.to_thread(run_agentic_rag, req)

            formatted_text = format_agentic_response(response)
            message_chunks = split_message(formatted_text, max_length=4000)

            for idx, chunk in enumerate(message_chunks):
                if idx == 0 and status_msg:
                    try:
                        await status_msg.edit_text(
                            chunk, parse_mode="Markdown", disable_web_page_preview=True
                        )
                    except Exception:
                        await status_msg.edit_text(chunk, disable_web_page_preview=True)
                else:
                    try:
                        await update.message.reply_text(
                            chunk, parse_mode="Markdown", disable_web_page_preview=True
                        )
                    except Exception:
                        await update.message.reply_text(chunk, disable_web_page_preview=True)

        except Exception as exc:
            logger.error("Error processing Telegram question: %s", exc, exc_info=True)
            err_text = f"❌ An error occurred while processing your request: {exc}"
            if status_msg:
                try:
                    await status_msg.edit_text(err_text)
                except Exception:
                    await update.message.reply_text(err_text)

    async def _ask_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ask <question> command."""
        if not context.args:
            if update.message:
                await update.message.reply_text(
                    "⚠️ Please provide a question. Example: `/ask What is attention mechanism?`",
                    parse_mode="Markdown",
                )
            return

        query = " ".join(context.args)
        await self._ask_question_internal(update, query)

    async def _text_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle plain text messages as questions."""
        if update.message and update.message.text:
            query = update.message.text.strip()
            if query:
                await self._ask_question_internal(update, query)

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Global error handler catching all exceptions to prevent bot crashes."""
        logger.error(
            "Telegram bot error encountered: %s", context.error, exc_info=context.error
        )
        if isinstance(update, Update) and update.message:
            try:
                await update.message.reply_text(
                    "❌ An unexpected error occurred in the bot handler. Please try again later."
                )
            except Exception as e:
                logger.warning("Failed to send error notification to Telegram user: %s", e)


async def start_telegram_bot() -> Application | None:
    """Async starter for Telegram bot polling if token is configured."""
    token = settings.telegram_bot_token.strip()
    if not token:
        logger.info("TELEGRAM_BOT_TOKEN not provided — Telegram Bot polling disabled.")
        return None

    logger.info("Initializing Telegram Bot...")
    bot = TelegramBot(token=token)
    app = bot.build()
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram Bot started polling successfully.")
    return app
