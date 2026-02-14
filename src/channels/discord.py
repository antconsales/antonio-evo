"""
Discord Bot Channel for Antonio Evo (v7.0).

Receives messages from Discord, processes them through the pipeline,
and sends back responses. Supports text and image attachments.
Uses rich embeds for formatted responses.

Requires: pip install discord.py

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

# Try to import discord.py library
try:
    import discord
    from discord import app_commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.info("discord.py not installed. Discord channel disabled.")


class DiscordChannel:
    """
    Discord Bot channel for Antonio Evo.

    Gateway pattern: receives Discord messages -> pipeline -> responds.
    Runs in a separate thread to not block the main server.
    """

    def __init__(self, config: Dict[str, Any], orchestrator: "Orchestrator"):
        self.config = config
        self.orchestrator = orchestrator
        self.token = config.get("token", "")
        self.allowed_users = config.get("allowed_users", [])
        self.allowed_channels = config.get("allowed_channels", [])
        self.max_message_length = 2000  # Discord limit
        self.use_embeds = config.get("use_embeds", True)
        self.send_thinking = config.get("send_thinking", True)
        self._client = None
        self._tree = None
        self._thread = None
        self._running = False
        self._loop = None

    @property
    def enabled(self) -> bool:
        return bool(self.token) and DISCORD_AVAILABLE

    def start(self) -> bool:
        """Start the Discord bot in a background thread."""
        if not self.enabled:
            if not DISCORD_AVAILABLE:
                logger.warning("Discord channel: discord.py not installed")
            elif not self.token:
                logger.warning("Discord channel: no token configured")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._run_bot, daemon=True, name="discord-bot")
        self._thread.start()
        logger.info("Discord channel started")
        return True

    def stop(self) -> None:
        """Stop the Discord bot."""
        self._running = False
        if self._client and self._loop:
            try:
                asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
            except Exception:
                pass
        logger.info("Discord channel stopped")

    def _run_bot(self) -> None:
        """Run the bot event loop in a separate thread."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_bot())
        except Exception as e:
            logger.error(f"Discord bot error: {e}")
            self._running = False

    async def _start_bot(self) -> None:
        """Initialize and start the Discord bot."""
        intents = discord.Intents.default()
        intents.message_content = True

        self._client = discord.Client(intents=intents)
        self._tree = app_commands.CommandTree(self._client)

        # Register event handlers
        @self._client.event
        async def on_ready():
            logger.info(f"Discord bot logged in as {self._client.user}")
            # Sync slash commands
            try:
                await self._tree.sync()
                logger.info("Discord slash commands synced")
            except Exception as e:
                logger.warning(f"Failed to sync slash commands: {e}")

        @self._client.event
        async def on_message(message):
            # Ignore own messages
            if message.author == self._client.user:
                return
            # Ignore bot messages
            if message.author.bot:
                return
            await self._handle_message(message)

        # Register slash commands
        @self._tree.command(name="ask", description="Ask Antonio a question")
        async def cmd_ask(interaction: discord.Interaction, question: str):
            if not self._is_allowed_user(interaction.user.id):
                await interaction.response.send_message("Not authorized.", ephemeral=True)
                return
            await interaction.response.defer()
            await self._process_and_reply_interaction(interaction, question)

        @self._tree.command(name="status", description="Show Antonio system status")
        async def cmd_status(interaction: discord.Interaction):
            if not self._is_allowed_user(interaction.user.id):
                await interaction.response.send_message("Not authorized.", ephemeral=True)
                return
            await interaction.response.defer()
            await self._cmd_status(interaction)

        @self._tree.command(name="help", description="Show Antonio help")
        async def cmd_help(interaction: discord.Interaction):
            await self._cmd_help(interaction)

        # Start the bot
        await self._client.start(self.token)

    def _is_allowed_user(self, user_id: int) -> bool:
        """Check if user is allowed to use the bot."""
        if not self.allowed_users:
            return True
        return user_id in self.allowed_users

    def _is_allowed_channel(self, channel_id: int) -> bool:
        """Check if channel is allowed."""
        if not self.allowed_channels:
            return True
        return channel_id in self.allowed_channels

    async def _handle_message(self, message) -> None:
        """Handle incoming text messages (and optional image attachments)."""
        if not self._is_allowed_user(message.author.id):
            return
        if not self._is_allowed_channel(message.channel.id):
            return

        text = message.content
        if not text and not message.attachments:
            return

        # Check for image attachments
        image_attachment = None
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/"):
                image_attachment = att
                break

        # Send thinking indicator
        thinking_msg = None
        if self.send_thinking:
            thinking_text = "Analyzing image..." if image_attachment else "..."
            thinking_msg = await message.reply(thinking_text, mention_author=False)

        try:
            if image_attachment:
                response_text, metadata = await self._process_with_image(
                    text or "Describe this image", image_attachment
                )
            else:
                response_text, metadata = await self._process_text(text)

            # Delete thinking message
            if thinking_msg:
                try:
                    await thinking_msg.delete()
                except Exception:
                    pass

            # Send response
            await self._send_response(message.channel, response_text, metadata, reply_to=message)

        except Exception as e:
            logger.error(f"Discord message processing error: {e}")
            if thinking_msg:
                try:
                    await thinking_msg.delete()
                except Exception:
                    pass
            await message.reply(f"Error: {e}", mention_author=False)

    async def _process_text(self, text: str) -> tuple:
        """Process text through the pipeline. Returns (response_text, metadata)."""
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.orchestrator.process({"text": text, "source": "discord"}),
        )
        return self._extract_text(result), self._extract_metadata(result)

    async def _process_with_image(self, text: str, attachment) -> tuple:
        """Process text + image through the pipeline."""
        # Download image
        image_bytes = await attachment.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        content_type = attachment.content_type or "image/png"
        input_data = {
            "text": text,
            "source": "discord",
            "attachments": [{
                "name": attachment.filename or "discord_image.png",
                "type": content_type,
                "size": len(image_bytes),
                "data": f"data:{content_type};base64,{image_b64}",
            }],
        }

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.orchestrator.process(input_data),
        )
        return self._extract_text(result), self._extract_metadata(result)

    async def _process_and_reply_interaction(self, interaction, question: str) -> None:
        """Process a slash command question and reply via interaction."""
        try:
            response_text, metadata = await self._process_text(question)
            if self.use_embeds:
                embed = self._build_embed(response_text, metadata)
                await interaction.followup.send(embed=embed)
            else:
                for chunk in self._split_message(response_text):
                    await interaction.followup.send(chunk)
        except Exception as e:
            logger.error(f"Discord slash command error: {e}")
            await interaction.followup.send(f"Error: {e}")

    async def _cmd_status(self, interaction) -> None:
        """Handle /status slash command."""
        try:
            health = self.orchestrator.health_check()
            if self.use_embeds:
                embed = discord.Embed(
                    title="Antonio Evo v7.0",
                    color=0x00ff88 if health.get("ollama_available") else 0xff4444,
                )
                embed.add_field(
                    name="Status",
                    value="Online" if health.get("ollama_available") else "Degraded",
                    inline=True,
                )
                embed.add_field(
                    name="Memory",
                    value=f"{health.get('memory', {}).get('total_neurons', 0)} neurons",
                    inline=True,
                )
                embed.add_field(
                    name="Profile",
                    value=health.get("profile", "unknown"),
                    inline=True,
                )
                await interaction.followup.send(embed=embed)
            else:
                status_text = (
                    f"**Antonio Evo v7.0**\n"
                    f"Status: {'Online' if health.get('ollama_available') else 'Degraded'}\n"
                    f"Memory: {health.get('memory', {}).get('total_neurons', 0)} neurons\n"
                    f"Profile: {health.get('profile', 'unknown')}"
                )
                await interaction.followup.send(status_text)
        except Exception as e:
            await interaction.followup.send(f"Status check failed: {e}")

    async def _cmd_help(self, interaction) -> None:
        """Handle /help slash command."""
        if self.use_embeds:
            embed = discord.Embed(
                title="Antonio Evo - Help",
                description="I'm your local AI assistant with evolutionary memory.",
                color=0x5865F2,
            )
            embed.add_field(
                name="What I can do",
                value=(
                    "- Answer questions\n"
                    "- Search the web\n"
                    "- Read and write files\n"
                    "- Execute Python code\n"
                    "- Analyze images\n"
                    "- Search knowledge base"
                ),
                inline=False,
            )
            embed.add_field(
                name="Slash Commands",
                value=(
                    "`/ask <question>` - Ask me anything\n"
                    "`/status` - System status\n"
                    "`/help` - This help message"
                ),
                inline=False,
            )
            embed.set_footer(text="You can also just type in the channel!")
            await interaction.response.send_message(embed=embed)
        else:
            await interaction.response.send_message(
                "**Antonio Evo** - Local AI assistant\n\n"
                "I can answer questions, search the web, analyze images, and more.\n\n"
                "Commands: `/ask`, `/status`, `/help`\n"
                "Or just type in the channel!"
            )

    async def _send_response(self, channel, text: str, metadata: dict, reply_to=None) -> None:
        """Send a response, using embeds if enabled."""
        if self.use_embeds and metadata:
            embed = self._build_embed(text, metadata)
            if reply_to:
                await reply_to.reply(embed=embed, mention_author=False)
            else:
                await channel.send(embed=embed)
        else:
            chunks = self._split_message(text)
            for i, chunk in enumerate(chunks):
                if i == 0 and reply_to:
                    await reply_to.reply(chunk, mention_author=False)
                else:
                    await channel.send(chunk)

    def _build_embed(self, text: str, metadata: dict) -> "discord.Embed":
        """Build a rich embed for the response."""
        # Truncate text for embed description (max 4096)
        if len(text) > 4000:
            text = text[:4000] + "\n\n... (truncated)"

        embed = discord.Embed(description=text, color=0x00ff88)

        # Add metadata footer
        parts = []
        if metadata.get("elapsed_ms"):
            parts.append(f"{metadata['elapsed_ms']}ms")
        if metadata.get("persona"):
            parts.append(metadata["persona"])
        if metadata.get("tools_used"):
            parts.append(f"tools: {', '.join(metadata['tools_used'])}")
        if metadata.get("neuron_stored"):
            parts.append("learned")

        if parts:
            embed.set_footer(text=" | ".join(parts))

        return embed

    def _split_message(self, text: str) -> list:
        """Split a message to fit Discord's 2000-char limit."""
        if len(text) <= self.max_message_length:
            return [text]

        chunks = []
        while text:
            if len(text) <= self.max_message_length:
                chunks.append(text)
                break

            # Try to split at a newline
            split_at = text.rfind("\n", 0, self.max_message_length)
            if split_at == -1 or split_at < self.max_message_length // 2:
                # Fall back to space
                split_at = text.rfind(" ", 0, self.max_message_length)
            if split_at == -1:
                split_at = self.max_message_length

            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()

        return chunks

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

    def _extract_metadata(self, result: Dict[str, Any]) -> dict:
        """Extract metadata from pipeline result for embed display."""
        if not result:
            return {}

        meta = result.get("_meta", {})
        decision = result.get("decision", {})
        return {
            "elapsed_ms": result.get("elapsed_ms", 0),
            "persona": decision.get("persona", ""),
            "tools_used": meta.get("tools_used", []),
            "neuron_stored": meta.get("neuron_stored", False),
        }

    def get_status(self) -> Dict[str, Any]:
        """Return channel status for API."""
        return {
            "name": "discord",
            "enabled": self.enabled,
            "running": self._running,
            "has_token": bool(self.token),
            "library_available": DISCORD_AVAILABLE,
            "allowed_users": len(self.allowed_users),
            "allowed_channels": len(self.allowed_channels),
            "use_embeds": self.use_embeds,
        }
