import React, { useState, useEffect, useCallback } from 'react';
import TitleBar from './components/TitleBar';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import SettingsPanel from './components/SettingsPanel';
import Avatar from './components/Avatar';
import ImageGenerator from './components/ImageGenerator';
import RuntimeOverviewPanel from './components/RuntimeOverviewPanel';
import { api, windowControls } from './api';
import { getWebSocket } from './services/websocket';

const CHAT_HISTORY_KEY = 'antonio_chat_history';
const CONVERSATIONS_KEY = 'antonio_conversations';
const SETTINGS_KEY = 'antonio_settings';

function App() {
  const [messages, setMessages] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [currentView, setCurrentView] = useState('chat'); // 'chat' | 'images' | 'settings'
  const [settings, setSettings] = useState({
    speakResponses: false,
    darkMode: true,
    apiUrl: 'http://localhost:8420',
    useWebSocket: true, // Enable WebSocket by default
    showAvatar: true,   // Show floating avatar
    deepThinking: false, // Enable extended reasoning (slower)
  });

  // Load settings, conversations and current chat from localStorage
  useEffect(() => {
    try {
      // Load settings
      const savedSettings = localStorage.getItem(SETTINGS_KEY);
      if (savedSettings) {
        setSettings(prev => ({ ...prev, ...JSON.parse(savedSettings) }));
      }

      // Load conversations list
      const savedConversations = localStorage.getItem(CONVERSATIONS_KEY);
      if (savedConversations) {
        const convs = JSON.parse(savedConversations);
        setConversations(convs);

        // Load most recent conversation
        if (convs.length > 0) {
          const recentConv = convs[0];
          setCurrentConversationId(recentConv.id);
          const savedMessages = localStorage.getItem(`${CHAT_HISTORY_KEY}_${recentConv.id}`);
          if (savedMessages) {
            setMessages(JSON.parse(savedMessages));
          }
        }
      }
    } catch (e) {
      console.error('Failed to load saved data:', e);
    }
  }, []);

  // Save messages when they change (strip large base64 data to fit localStorage)
  useEffect(() => {
    if (currentConversationId && messages.length > 0) {
      try {
        // Strip base64 data from attachments to prevent localStorage quota overflow
        const messagesForStorage = messages.map(msg => {
          if (!msg.attachments || msg.attachments.length === 0) return msg;
          return {
            ...msg,
            attachments: msg.attachments.map(att => ({
              id: att.id,
              name: att.name,
              type: att.type,
              size: att.size,
              // Don't store full base64 data - it can be several MB per image
              data: att.data && att.data.length > 10000 ? null : att.data,
              preview: null, // blob URLs don't persist anyway
            })),
          };
        });
        localStorage.setItem(`${CHAT_HISTORY_KEY}_${currentConversationId}`, JSON.stringify(messagesForStorage));

        // Update conversation metadata
        const updatedConversations = conversations.map(conv =>
          conv.id === currentConversationId
            ? {
                ...conv,
                lastMessage: messages[messages.length - 1]?.content?.substring(0, 50) || '',
                messageCount: messages.length,
                updatedAt: new Date().toISOString(),
              }
            : conv
        );
        setConversations(updatedConversations);
        localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(updatedConversations));
      } catch (e) {
        console.error('Failed to save messages:', e);
      }
    }
  }, [messages, currentConversationId]);

  // Save settings when they change
  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch (e) {
      console.error('Failed to save settings:', e);
    }
  }, [settings]);

  // Apply dark/light mode class to document root
  useEffect(() => {
    if (settings.darkMode) {
      document.documentElement.classList.remove('light-mode');
    } else {
      document.documentElement.classList.add('light-mode');
    }
  }, [settings.darkMode]);

  // Create new conversation if none exists when first message is sent
  const ensureConversation = useCallback(() => {
    if (!currentConversationId) {
      const newConv = {
        id: Date.now().toString(),
        title: 'New Chat',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        lastMessage: '',
        messageCount: 0,
      };
      const newConversations = [newConv, ...conversations];
      setConversations(newConversations);
      setCurrentConversationId(newConv.id);
      localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(newConversations));
      return newConv.id;
    }
    return currentConversationId;
  }, [currentConversationId, conversations]);

  // Load a conversation
  const loadConversation = useCallback((convId) => {
    try {
      const savedMessages = localStorage.getItem(`${CHAT_HISTORY_KEY}_${convId}`);
      setMessages(savedMessages ? JSON.parse(savedMessages) : []);
      setCurrentConversationId(convId);
    } catch (e) {
      console.error('Failed to load conversation:', e);
      setMessages([]);
    }
  }, []);

  // Delete a conversation
  const deleteConversation = useCallback((convId) => {
    const newConversations = conversations.filter(c => c.id !== convId);
    setConversations(newConversations);
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(newConversations));
    localStorage.removeItem(`${CHAT_HISTORY_KEY}_${convId}`);

    if (currentConversationId === convId) {
      if (newConversations.length > 0) {
        loadConversation(newConversations[0].id);
      } else {
        setMessages([]);
        setCurrentConversationId(null);
      }
    }
  }, [conversations, currentConversationId, loadConversation]);

  // Avatar state
  const [avatarExpanded, setAvatarExpanded] = useState(false);
  const [currentMood, setCurrentMood] = useState('neutral');
  const [currentPersona, setCurrentPersona] = useState(null);
  const [neuronCount, setNeuronCount] = useState(0);

  // Tool actions state (v5.0)
  const [activeToolActions, setActiveToolActions] = useState([]);

  // Global UI State (from /api/ui/state)
  const [uiState, setUiState] = useState(null);
  const [showRuntimeOverview, setShowRuntimeOverview] = useState(false);

  // Fetch global UI state
  const fetchUiState = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8420/api/ui/state');
      const data = await res.json();
      if (data.success) {
        setUiState(data.state);
      }
    } catch (e) {
      console.error('[App] Failed to fetch UI state:', e);
    }
  }, []);

  // Fetch UI state on mount and periodically
  useEffect(() => {
    fetchUiState();
    const interval = setInterval(fetchUiState, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [fetchUiState]);

  // WebSocket instance
  const ws = settings.useWebSocket ? getWebSocket() : null;

  // Setup WebSocket listeners
  useEffect(() => {
    if (!settings.useWebSocket || !ws) return;

    const unsubscribers = [
      ws.on('connected', ({ sessionId }) => {
        console.log('[App] WS Connected, session:', sessionId);
        setIsConnected(true);
        fetchMemoryStats();
      }),

      ws.on('disconnected', () => {
        console.log('[App] WS Disconnected');
        setIsConnected(false);
      }),

      ws.on('thinking', ({ isThinking }) => {
        setIsLoading(isThinking);
      }),

      ws.on('mood', ({ mood, persona }) => {
        setCurrentMood(mood || 'neutral');
        setCurrentPersona(persona || null);
      }),

      ws.on('tool_action_start', ({ tool, arguments: args }) => {
        setActiveToolActions((prev) => [
          ...prev,
          { tool, arguments: args, status: 'running' },
        ]);
      }),

      ws.on('tool_action_end', ({ tool, success, elapsedMs }) => {
        setActiveToolActions((prev) =>
          prev.map((a) =>
            a.tool === tool && a.status === 'running'
              ? { ...a, status: 'done', success, elapsedMs }
              : a
          )
        );
      }),

      // v6.0: Streaming text chunks
      ws.on('chunk', ({ text }) => {
        setStreamingText((prev) => prev + text);
      }),

      ws.on('response', (data) => {
        const responseText = data.text || (data.error ? `Error: ${data.error}` : 'No response');
        const assistantMessage = {
          id: Date.now(),
          role: 'assistant',
          content: responseText,
          timestamp: new Date().toISOString(),
          success: data.success !== false,
          meta: {
            elapsed_ms: data.elapsedMs,
            handler: data.handler,
            persona: data.persona,
            neuron_stored: data.neuronStored,
            tools_used: data.toolsUsed || [],
          },
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setIsLoading(false);
        setStreamingText('');
        setActiveToolActions([]);

        // Update mood
        if (data.mood) {
          setCurrentMood(data.mood);
        }
        if (data.persona) {
          setCurrentPersona(data.persona);
        }

        // Refresh neuron count if stored
        if (data.neuronStored) {
          fetchMemoryStats();
        }

        // Speak response if enabled
        if (settings.speakResponses && data.text) {
          api.speak(data.text, { voice: 'diego' });
        }
      }),

      ws.on('error', ({ error }) => {
        const errorText = typeof error === 'string' ? error : (error?.message || 'Unknown error');
        const errorMessage = {
          id: Date.now(),
          role: 'assistant',
          content: `Error: ${errorText}`,
          timestamp: new Date().toISOString(),
          success: false,
        };
        setMessages((prev) => [...prev, errorMessage]);
        setIsLoading(false);
        setStreamingText('');
        setActiveToolActions([]);
        setCurrentMood('error');
      }),

      ws.on('reconnecting', ({ attempt }) => {
        console.log('[App] Reconnecting, attempt:', attempt);
      }),
    ];

    // Connect WebSocket
    ws.connect().catch((err) => {
      console.error('[App] WS connection failed:', err);
      setIsConnected(false);
    });

    return () => {
      unsubscribers.forEach((unsub) => unsub());
    };
  }, [settings.useWebSocket, settings.speakResponses]);

  // Fallback: Check REST API health if WebSocket disabled
  useEffect(() => {
    if (settings.useWebSocket) return;

    const checkHealth = async () => {
      try {
        const result = await api.health();
        setIsConnected(result.success && result.data?.status === 'ok');
      } catch (e) {
        setIsConnected(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, [settings.useWebSocket]);

  // Fetch memory stats
  const fetchMemoryStats = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8420/api/memory/stats');
      const data = await res.json();
      setNeuronCount(data.total_neurons || 0);
    } catch (e) {
      console.error('[App] Failed to fetch memory stats:', e);
    }
  }, []);

  // Initial memory stats fetch
  useEffect(() => {
    fetchMemoryStats();
  }, [fetchMemoryStats]);

  // Send message (WebSocket or REST)
  const sendMessage = useCallback(async (text, useVoice = false, attachments = []) => {
    if (!text.trim() && attachments.length === 0) return;

    // Ensure conversation exists
    ensureConversation();

    // Add user message with attachments
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
      attachments: attachments.length > 0 ? attachments : undefined,
    };
    setMessages((prev) => [...prev, userMessage]);

    // WebSocket mode
    if (settings.useWebSocket && ws?.isConnected) {
      setIsLoading(true);
      setCurrentMood('thinking');
      ws.sendMessage(text, attachments);
      return;
    }

    // REST API fallback
    setIsLoading(true);

    try {
      const result = await api.ask(text, {
        speak: settings.speakResponses,
        returnAudio: false,
        think: settings.deepThinking,
      });

      let responseText = 'No response';
      if (result.success) {
        const data = result.data || {};
        let msg = data.message || data.data || data;

        if (typeof msg === 'string') {
          try {
            msg = JSON.parse(msg);
          } catch {
            responseText = msg;
            msg = null;
          }
        }

        if (msg && typeof msg === 'object') {
          responseText = msg.conclusion || msg.response || msg.output || msg.analysis || msg.content || msg.text || '';
          if (!responseText || typeof responseText === 'object') {
            const values = Object.values(msg).filter(v => typeof v === 'string' && v.length > 0);
            responseText = values[0] || JSON.stringify(msg, null, 2);
          }
        }
        responseText = responseText || 'No response';
      } else {
        responseText = result.data?.error || result.error || 'Error: Could not get response';
      }

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: responseText,
        timestamp: new Date().toISOString(),
        success: result.success,
        meta: result.data?._meta || null,
      };
      setMessages((prev) => [...prev, assistantMessage]);

      if (settings.speakResponses && result.success && responseText) {
        api.speak(responseText, { voice: 'diego' });
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Error: ${error.message}`,
        timestamp: new Date().toISOString(),
        success: false,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [settings, ws]);

  const clearChat = () => {
    // Create a new conversation
    const newConv = {
      id: Date.now().toString(),
      title: 'New Chat',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      lastMessage: '',
      messageCount: 0,
    };
    const newConversations = [newConv, ...conversations];
    setConversations(newConversations);
    setCurrentConversationId(newConv.id);
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(newConversations));
    setMessages([]);
    setCurrentMood('neutral');
    setCurrentPersona(null);
  };

  const toggleAvatar = () => {
    setAvatarExpanded((prev) => !prev);
  };

  const renderMainContent = () => {
    if (showSettings) {
      return (
        <SettingsPanel
          settings={settings}
          onSettingsChange={setSettings}
          onClose={() => setShowSettings(false)}
          isConnected={isConnected}
        />
      );
    }

    switch (currentView) {
      case 'images':
        return <ImageGenerator isConnected={isConnected} />;
      case 'chat':
      default:
        return (
          <ChatArea
            messages={messages}
            onSendMessage={sendMessage}
            isLoading={isLoading}
            isConnected={isConnected}
            activeToolActions={activeToolActions}
            streamingText={streamingText}
          />
        );
    }
  };

  return (
    <div className="app">
      <TitleBar
        isConnected={isConnected}
        isElectron={windowControls.isElectron}
        uiState={uiState}
        onShowRuntimeOverview={() => setShowRuntimeOverview(true)}
      />
      <div className="app-content">
        <Sidebar
          onNewChat={clearChat}
          onShowSettings={() => setShowSettings(true)}
          onNavigate={setCurrentView}
          currentView={currentView}
          isConnected={isConnected}
          conversations={conversations}
          currentConversationId={currentConversationId}
          onSelectConversation={loadConversation}
          onDeleteConversation={deleteConversation}
        />
        <main className="main-area">
          {renderMainContent()}
        </main>
      </div>

      {/* Floating Avatar Companion */}
      {settings.showAvatar && (
        <Avatar
          mood={currentMood}
          persona={currentPersona}
          isThinking={isLoading}
          isExpanded={avatarExpanded}
          onToggle={toggleAvatar}
          onSendMessage={sendMessage}
          neuronCount={neuronCount}
          isConnected={isConnected}
        />
      )}

      {/* Runtime Overview Panel */}
      {showRuntimeOverview && (
        <RuntimeOverviewPanel
          uiState={uiState}
          onClose={() => setShowRuntimeOverview(false)}
        />
      )}
    </div>
  );
}

export default App;
