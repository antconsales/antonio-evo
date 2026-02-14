"""
Telegram Bot Channel for Antonio Evo (v6.0).

Receives messages from Telegram, processes them through the pipeline,
and sends back responses. Supports text and image attachments.

Requires: pip install python-telegram-bot

Configuration: config/channels.json
"""

import asyncio
import base64
import logging
import threading
from typing import Dict, Any, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..main import Orchestrator

# Try to import telegram bot library
try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, MessageHandler, filters
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.info("python-telegram-bot not installed. Telegram channel disabled.")


class TelegramChannel:
    """
    Telegram Bot channel for Antonio Evo.

    Gateway pattern: receives Telegram messages -> pipeline -> responds.
    Runs in a separate thread to not block the main server.
    """

    def __init__(self, config: Dict[str, Any], orchestrator: "Orchestrator"):
        self.config = config
        self.orchestrator = orchestrator
        self.token = config.get("token", "")
        self.allowed_users = config.get("allowed_users", [])
        self.max_message_length = config.get("max_message_length", 4096)
        self.send_thinking = config.get("send_thinking", True)
        self._app = None
        self._thread = None
        self._running = False

    @property
    def enabled(self) -> bool:
        return bool(self.token) and TELEGRAM_AVAILABLE

    def start(self) -> bool:
        """Start the Telegram bot in a background thread."""
        if not self.enabled:
            if not TELEGRAM_AVAILABLE:
                logger.warning("Telegram channel: python-telegram-bot not installed")
            elif not self.token:
                logger.warning("Telegram channel: no token configured")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._run_bot, daemon=True, name="telegram-bot")
        self._thread.start()
        logger.info("Telegram channel started")
        return True

    def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False
        if self._app:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._app.stop(), self._app.loop
                )
            except Exception:
                pass
        logger.info("Telegram channel stopped")

    def _run_bot(self) -> None:
        """Run the bot event loop in a separate thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._start_bot())
        except Exception as e:
            logger.error(f"Telegram bot error: {e}")
            self._running = False

    async def _start_bot(self) -> None:
        """Initialize and start the Telegram bot."""
        self._app = Application.builder().token(self.token).build()

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self._handle_message
        ))
        self._app.add_handler(MessageHandler(
            filters.PHOTO, self._handle_photo
        ))

        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram bot polling started")

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    def _is_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to use the bot."""
        if not self.allowed_users:
            return True  # No restrictions
        return user_id in self.allowed_users

    async def _cmd_start(self, update: Update, context) -> None:
        """Handle /start command."""
        if not self._is_allowed(update.effective_user.id):
            await update.message.reply_text("Not authorized.")
            return

        await update.message.reply_text(
            f"Ciao! Sono Antonio, il tuo assistente AI locale.\n\n"
            f"Scrivimi qualsiasi cosa e ti risponderò.\n"
            f"Puoi anche inviarmi foto da analizzare.\n\n"
            f"Comandi:\n"
            f"/help - Mostra aiuto\n"
            f"/status - Stato del sistema"
        )

    async def _cmd_help(self, update: Update, context) -> None:
        """Handle /help command."""
        if not self._is_allowed(update.effective_user.id):
            return

        await update.message.reply_text(
            "Posso:\n"
            "- Rispondere a domande\n"
            "- Cercare sul web\n"
            "- Leggere e scrivere file\n"
            "- Eseguire codice Python\n"
            "- Analizzare immagini\n\n"
            "Scrivimi in qualsiasi lingua!"
        )

    async def _cmd_status(self, update: Update, context) -> None:
        """Handle /status command."""
        if not self._is_allowed(update.effective_user.id):
            return

        try:
            health = self.orchestrator.health_check()
            status_text = (
                f"Antonio Evo v6.0\n"
                f"Status: {'OK' if health.get('ollama_available') else 'Degraded'}\n"
                f"Memory: {health.get('memory', {}).get('total_neurons', 0)} neurons\n"
                f"Profile: {health.get('profile', 'unknown')}"
            )
            await update.message.reply_text(status_text)
        except Exception as e:
            await update.message.reply_text(f"Status check failed: {e}")

    async def _handle_message(self, update: Update, context) -> None:
        """Handle incoming text messages."""
        if not self._is_allowed(update.effective_user.id):
            return

        text = update.message.text
        if not text:
            return

        # Send thinking indicator
        if self.send_thinking:
            thinking_msg = await update.message.reply_text("...")

        try:
            # Process through the pipeline (synchronous)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.orchestrator.process({"text": text, "source": "telegram"}),
            )

            # Extract response text
            response_text = self._extract_text(result)

            # Truncate if needed
            if len(response_text) > self.max_message_length:
                response_text = response_text[:self.max_message_length - 20] + "\n\n... (truncated)"

            # Delete thinking message and send response
            if self.send_thinking:
                try:
                    await thinking_msg.delete()
                except Exception:
                    pass

            await update.message.reply_text(response_text)

        except Exception as e:
            logger.error(f"Telegram message processing error: {e}")
            if self.send_thinking:
                try:
                    await thinking_msg.delete()
                except Exception:
                    pass
            await update.message.reply_text(f"Error: {e}")

    async def _handle_photo(self, update: Update, context) -> None:
        """Handle incoming photo messages."""
        if not self._is_allowed(update.effective_user.id):
            return

        # Get the largest photo
        photo = update.message.photo[-1]
        caption = update.message.caption or "Describe this image"

        if self.send_thinking:
            thinking_msg = await update.message.reply_text("Analyzing image...")

        try:
            # Download photo
            file = await context.bot.get_file(photo.file_id)
            photo_bytes = await file.download_as_bytearray()
            photo_b64 = base64.b64encode(photo_bytes).decode("utf-8")

            # Build input with attachment
            input_data = {
                "text": caption,
                "source": "telegram",
                "attachments": [{
                    "name": f"telegram_photo_{photo.file_id[:8]}.jpg",
                    "type": "image/jpeg",
                    "size": len(photo_bytes),
                    "data": f"data:image/jpeg;base64,{photo_b64}",
                }],
            }

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.orchestrator.process(input_data),
            )

            response_text = self._extract_text(result)

            if self.send_thinking:
                try:
                    await thinking_msg.delete()
                except Exception:
                    pass

            await update.message.reply_text(response_text)

        except Exception as e:
            logger.error(f"Telegram photo processing error: {e}")
            if self.send_thinking:
                try:
                    await thinking_msg.delete()
                except Exception:
                    pass
            await update.message.reply_text(f"Error processing image: {e}")

    def _extract_text(self, result: Dict[str, Any]) -> str:
        """Extract readable text from pipeline result."""
        if not result:
            return "No response"

        if result.get("success"):
            output = result.get("output") or result.get("text", "")
            if isinstance(output, dict):
                return output.get("content") or output.get("text") or output.get("answer") or str(output)
            return str(output) if output else "No response"
        else:
            return f"Error: {result.get('error', 'Unknown error')}"

    def get_status(self) -> Dict[str, Any]:
        """Return channel status for API."""
        return {
            "name": "telegram",
            "enabled": self.enabled,
            "running": self._running,
            "has_token": bool(self.token),
            "library_available": TELEGRAM_AVAILABLE,
            "allowed_users": len(self.allowed_users),
        }
