"""
Webhook Service - n8n workflow integration.

Manages webhook configurations and triggers n8n workflows
in response to Antonio events (post-response, memory, errors).

Webhooks are NON-BLOCKING - they never delay Antonio's response.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("data/webhooks.json")


def _default_config() -> Dict:
    return {"webhooks": []}


def _load_config() -> Dict:
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load webhook config: {e}")
    return _default_config()


def _save_config(config: Dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


class WebhookService:
    """
    Manages n8n webhook integrations.

    Webhooks are triggered asynchronously in background threads
    to avoid blocking the main response pipeline.
    """

    VALID_TRIGGERS = {"post_response", "on_memory", "on_error", "manual"}

    def __init__(self):
        self._config = _load_config()

    def reload(self):
        """Reload config from disk."""
        self._config = _load_config()

    @property
    def webhooks(self) -> List[Dict]:
        return self._config.get("webhooks", [])

    def get_webhook(self, webhook_id: str) -> Optional[Dict]:
        for wh in self.webhooks:
            if wh["id"] == webhook_id:
                return wh
        return None

    def add_webhook(self, name: str, url: str, trigger: str = "post_response") -> Dict:
        """Add a new webhook configuration."""
        if trigger not in self.VALID_TRIGGERS:
            raise ValueError(f"Invalid trigger: {trigger}. Must be one of {self.VALID_TRIGGERS}")

        webhook = {
            "id": str(uuid.uuid4())[:8],
            "name": name,
            "url": url,
            "trigger": trigger,
            "enabled": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._config.setdefault("webhooks", []).append(webhook)
        _save_config(self._config)
        return webhook

    def update_webhook(self, webhook_id: str, updates: Dict) -> Optional[Dict]:
        """Update an existing webhook."""
        for wh in self._config.get("webhooks", []):
            if wh["id"] == webhook_id:
                allowed_fields = {"name", "url", "trigger", "enabled"}
                for key, value in updates.items():
                    if key in allowed_fields:
                        wh[key] = value
                _save_config(self._config)
                return wh
        return None

    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook by ID."""
        webhooks = self._config.get("webhooks", [])
        original_len = len(webhooks)
        self._config["webhooks"] = [w for w in webhooks if w["id"] != webhook_id]
        if len(self._config["webhooks"]) < original_len:
            _save_config(self._config)
            return True
        return False

    def test_webhook(self, webhook_id: str) -> Dict:
        """Test a webhook connection by sending a ping."""
        webhook = self.get_webhook(webhook_id)
        if not webhook:
            return {"success": False, "error": "Webhook not found"}

        try:
            response = requests.post(
                webhook["url"],
                json={
                    "type": "test",
                    "source": "antonio-evo",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "Webhook test from Antonio Evo",
                },
                timeout=10,
            )
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
            }
        except requests.Timeout:
            return {"success": False, "error": "Connection timed out"}
        except requests.ConnectionError:
            return {"success": False, "error": "Cannot connect to webhook URL"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def trigger_event(self, event: str, data: Dict[str, Any]) -> None:
        """
        Trigger all enabled webhooks for an event.

        Runs in background threads - NEVER blocks the caller.
        """
        matching = [
            w for w in self.webhooks
            if w.get("enabled") and w.get("trigger") == event
        ]
        if not matching:
            return

        payload = {
            "type": event,
            "source": "antonio-evo",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

        for webhook in matching:
            thread = threading.Thread(
                target=self._fire_webhook,
                args=(webhook, payload),
                daemon=True,
            )
            thread.start()

    def _fire_webhook(self, webhook: Dict, payload: Dict) -> None:
        """Fire a single webhook (runs in background thread)."""
        try:
            response = requests.post(
                webhook["url"],
                json=payload,
                timeout=15,
            )
            if response.status_code >= 400:
                logger.warning(
                    f"Webhook '{webhook['name']}' returned {response.status_code}"
                )
        except Exception as e:
            logger.warning(f"Webhook '{webhook['name']}' failed: {e}")


# Singleton
_instance: Optional[WebhookService] = None


def get_webhook_service() -> WebhookService:
    global _instance
    if _instance is None:
        _instance = WebhookService()
    return _instance
