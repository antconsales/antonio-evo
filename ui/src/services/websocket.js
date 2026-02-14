/**
 * WebSocket Service for Antonio Evo
 *
 * Real-time communication with the backend.
 * Handles: chat messages, mood changes, thinking status.
 */

const WS_BASE_URL = 'ws://localhost:8420';

export class AntonioWebSocket {
  constructor(options = {}) {
    this.url = options.url || `${WS_BASE_URL}/ws/chat`;
    this.ws = null;
    this.sessionId = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.reconnectDelay = 3000;
    this.listeners = new Map();
    this.isConnecting = false;
    this.messageQueue = [];
  }

  /**
   * Connect to WebSocket server
   */
  connect() {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return Promise.resolve();
    }

    this.isConnecting = true;

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('[WS] Connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this._flushMessageQueue();
          resolve();
        };

        this.ws.onclose = (event) => {
          console.log('[WS] Disconnected', event.code, event.reason);
          this.isConnecting = false;
          this.sessionId = null;
          this._emit('disconnected', { code: event.code, reason: event.reason });
          this._scheduleReconnect();
        };

        this.ws.onerror = (event) => {
          console.error('[WS] Connection error');
          this.isConnecting = false;
          // Don't emit 'error' for connection issues - onclose handles reconnection.
          // Only server-returned errors (from _handleMessage) should create chat messages.
          reject(new Error('WebSocket connection failed'));
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this._handleMessage(data);
          } catch (e) {
            console.error('[WS] Failed to parse message:', e);
          }
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  /**
   * Send a chat message with optional attachments
   * @param {string} text - Message text
   * @param {Array} attachments - Optional array of attachment objects
   */
  sendMessage(text, attachments = []) {
    const message = {
      type: 'message',
      text: text,
    };

    // Add attachments if present
    if (attachments && attachments.length > 0) {
      message.attachments = attachments.map(att => ({
        name: att.name,
        type: att.type,
        size: att.size,
        data: att.data, // Base64 encoded
      }));
    }

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      this.messageQueue.push(message);
      this.connect();
    }
  }

  /**
   * Subscribe to events
   * @param {string} event - Event name
   * @param {function} callback - Callback function
   * @returns {function} Unsubscribe function
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);

    return () => {
      this.listeners.get(event)?.delete(callback);
    };
  }

  /**
   * Check if connected
   */
  get isConnected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // ========== Private Methods ==========

  _handleMessage(data) {
    console.log('[WS] Event:', data.type, data);

    switch (data.type) {
      case 'connected':
        this.sessionId = data.session_id;
        this._emit('connected', { sessionId: data.session_id });
        break;

      case 'thinking_start':
        this._emit('thinking', { isThinking: true });
        break;

      case 'thinking_end':
        this._emit('thinking', { isThinking: false });
        break;

      case 'mood_change':
        this._emit('mood', { mood: data.mood, persona: data.persona });
        break;

      case 'response':
        this._emit('response', {
          text: data.text,
          persona: data.persona,
          mood: data.mood,
          elapsedMs: data.elapsed_ms,
          neuronStored: data.neuron_stored,
          handler: data.handler,
          toolsUsed: data.tools_used || [],
        });
        break;

      case 'tool_action_start':
        this._emit('tool_action_start', {
          tool: data.tool,
          arguments: data.arguments || {},
        });
        break;

      case 'tool_action_end':
        this._emit('tool_action_end', {
          tool: data.tool,
          success: data.success,
          elapsedMs: data.elapsed_ms,
        });
        break;

      case 'error':
        this._emit('error', { error: data.error });
        break;

      default:
        console.log('[WS] Unknown event:', data.type);
    }
  }

  _emit(event, data) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach((cb) => {
        try {
          cb(data);
        } catch (e) {
          console.error('[WS] Listener error:', e);
        }
      });
    }
  }

  _scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('[WS] Max reconnect attempts reached');
      this._emit('reconnect_failed', {});
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.min(this.reconnectAttempts, 5);

    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    this._emit('reconnecting', { attempt: this.reconnectAttempts, delay });

    setTimeout(() => {
      this.connect().catch(() => {});
    }, delay);
  }

  _flushMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(message));
      }
    }
  }
}

// Singleton instance
let instance = null;

export function getWebSocket(options) {
  if (!instance) {
    instance = new AntonioWebSocket(options);
  }
  return instance;
}

// React hook for WebSocket
export function useAntonioWebSocket() {
  return getWebSocket();
}

export default AntonioWebSocket;
