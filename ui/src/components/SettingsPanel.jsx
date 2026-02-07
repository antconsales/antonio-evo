import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Key, Plus, Trash2, Eye, EyeOff, Copy, Check, ExternalLink, AlertCircle } from 'lucide-react';

function SettingsPanel({ settings, onSettingsChange, onClose, isConnected }) {
  const [localSettings, setLocalSettings] = useState(settings);
  const [serverInfo, setServerInfo] = useState(null);
  const [activeTab, setActiveTab] = useState('general');

  // API Keys state
  const [apiKeys, setApiKeys] = useState([]);
  const [newKeyName, setNewKeyName] = useState('');
  const [showNewKey, setShowNewKey] = useState(null);
  const [copiedKey, setCopiedKey] = useState(null);

  // External APIs state
  const [externalApis, setExternalApis] = useState({
    openai: { enabled: false, apiKey: '', model: 'gpt-4' },
    anthropic: { enabled: false, apiKey: '', model: 'claude-3-opus-20240229' },
    google: { enabled: false, apiKey: '', model: 'gemini-pro' },
    custom: []
  });
  const [showApiKey, setShowApiKey] = useState({});
  const [newCustomApi, setNewCustomApi] = useState({ name: '', baseUrl: '', apiKey: '', model: '' });

  useEffect(() => {
    loadServerInfo();
    loadApiKeys();
    loadExternalApis();
  }, []);

  const loadServerInfo = async () => {
    try {
      const result = await api.health();
      if (result.success) {
        setServerInfo(result.data);
      }
    } catch (e) {
      console.error('Failed to load server info:', e);
    }
  };

  const loadApiKeys = async () => {
    try {
      const res = await fetch('http://localhost:8420/api/keys');
      const data = await res.json();
      if (data.success) {
        setApiKeys(data.keys || []);
      }
    } catch (e) {
      console.error('Failed to load API keys:', e);
      // Load from localStorage as fallback
      const saved = localStorage.getItem('antonio_api_keys');
      if (saved) setApiKeys(JSON.parse(saved));
    }
  };

  const loadExternalApis = async () => {
    try {
      const res = await fetch('http://localhost:8420/api/external-apis');
      const data = await res.json();
      if (data.success && data.apis) {
        setExternalApis(data.apis);
      }
    } catch (e) {
      console.error('Failed to load external APIs:', e);
      // Load from localStorage as fallback
      const saved = localStorage.getItem('antonio_external_apis');
      if (saved) setExternalApis(JSON.parse(saved));
    }
  };

  const generateApiKey = async () => {
    if (!newKeyName.trim()) return;

    const newKey = {
      id: Date.now().toString(),
      name: newKeyName,
      key: `ak_${generateRandomKey()}`,
      created: new Date().toISOString(),
      lastUsed: null
    };

    try {
      const res = await fetch('http://localhost:8420/api/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newKey)
      });
      const data = await res.json();
      if (data.success) {
        setApiKeys([...apiKeys, newKey]);
        setShowNewKey(newKey.key);
        setNewKeyName('');
      }
    } catch (e) {
      // Save locally as fallback
      const updated = [...apiKeys, newKey];
      setApiKeys(updated);
      localStorage.setItem('antonio_api_keys', JSON.stringify(updated));
      setShowNewKey(newKey.key);
      setNewKeyName('');
    }
  };

  const deleteApiKey = async (keyId) => {
    try {
      await fetch(`http://localhost:8420/api/keys/${keyId}`, { method: 'DELETE' });
    } catch (e) {
      console.error('Failed to delete key from server:', e);
    }
    const updated = apiKeys.filter(k => k.id !== keyId);
    setApiKeys(updated);
    localStorage.setItem('antonio_api_keys', JSON.stringify(updated));
  };

  const copyToClipboard = (text, keyId) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(keyId);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const generateRandomKey = () => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    return Array.from({ length: 48 }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
  };

  const updateExternalApi = async (provider, updates) => {
    const newApis = {
      ...externalApis,
      [provider]: { ...externalApis[provider], ...updates }
    };
    setExternalApis(newApis);

    try {
      await fetch('http://localhost:8420/api/external-apis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, ...updates })
      });
    } catch (e) {
      console.error('Failed to save external API:', e);
    }
    localStorage.setItem('antonio_external_apis', JSON.stringify(newApis));
  };

  const addCustomApi = () => {
    if (!newCustomApi.name || !newCustomApi.baseUrl) return;

    const customApis = [...(externalApis.custom || []), { ...newCustomApi, id: Date.now().toString() }];
    const newApis = { ...externalApis, custom: customApis };
    setExternalApis(newApis);
    localStorage.setItem('antonio_external_apis', JSON.stringify(newApis));
    setNewCustomApi({ name: '', baseUrl: '', apiKey: '', model: '' });
  };

  const removeCustomApi = (apiId) => {
    const customApis = externalApis.custom.filter(a => a.id !== apiId);
    const newApis = { ...externalApis, custom: customApis };
    setExternalApis(newApis);
    localStorage.setItem('antonio_external_apis', JSON.stringify(newApis));
  };

  const handleChange = (key, value) => {
    const newSettings = { ...localSettings, [key]: value };
    setLocalSettings(newSettings);
    onSettingsChange(newSettings);
  };

  const maskApiKey = (key) => {
    if (!key) return '';
    return key.substring(0, 8) + '...' + key.substring(key.length - 4);
  };

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h2>Settings</h2>
        <button className="close-btn" onClick={onClose}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Settings Tabs */}
      <div className="settings-tabs">
        <button
          className={`settings-tab ${activeTab === 'general' ? 'active' : ''}`}
          onClick={() => setActiveTab('general')}
        >
          General
        </button>
        <button
          className={`settings-tab ${activeTab === 'apikeys' ? 'active' : ''}`}
          onClick={() => setActiveTab('apikeys')}
        >
          API Keys
        </button>
        <button
          className={`settings-tab ${activeTab === 'external' ? 'active' : ''}`}
          onClick={() => setActiveTab('external')}
        >
          External APIs
        </button>
      </div>

      <div className="settings-content">
        {/* General Tab */}
        {activeTab === 'general' && (
          <>
            {/* Connection Section */}
            <section className="settings-section">
              <h3>Connection</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>API Server Status</label>
                  <p className="setting-description">Connection to Antonio backend</p>
                </div>
                <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
                  {isConnected ? 'Connected' : 'Disconnected'}
                </div>
              </div>

              {serverInfo && (
                <div className="server-info-card">
                  <div className="info-row">
                    <span>Status</span>
                    <span>{serverInfo.status}</span>
                  </div>
                  <div className="info-row">
                    <span>LLM Available</span>
                    <span>{serverInfo.llm_available ? 'Yes' : 'No'}</span>
                  </div>
                </div>
              )}
            </section>

            {/* Voice Section */}
            <section className="settings-section">
              <h3>Voice</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>Speak Responses</label>
                  <p className="setting-description">Read responses aloud using text-to-speech</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={localSettings.speakResponses}
                    onChange={(e) => handleChange('speakResponses', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            </section>

            {/* Real-time Section */}
            <section className="settings-section">
              <h3>Real-time</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>WebSocket Connection</label>
                  <p className="setting-description">Use real-time WebSocket for faster responses</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={localSettings.useWebSocket}
                    onChange={(e) => handleChange('useWebSocket', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
              <div className="setting-item">
                <div className="setting-info">
                  <label>Avatar Companion</label>
                  <p className="setting-description">Show floating avatar assistant</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={localSettings.showAvatar}
                    onChange={(e) => handleChange('showAvatar', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            </section>

            {/* Appearance Section */}
            <section className="settings-section">
              <h3>Appearance</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>Dark Mode</label>
                  <p className="setting-description">Use dark color scheme</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={localSettings.darkMode}
                    onChange={(e) => handleChange('darkMode', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            </section>

            {/* About Section */}
            <section className="settings-section">
              <h3>About</h3>
              <div className="about-card">
                <div className="about-logo">A</div>
                <div className="about-info">
                  <h4>Antonio Evo</h4>
                  <p>Version 4.0.0</p>
                  <p className="about-description">
                    Local AI assistant with evolutionary memory. Learns from interactions, never forgets, completely private.
                  </p>
                </div>
              </div>
            </section>
          </>
        )}

        {/* API Keys Tab */}
        {activeTab === 'apikeys' && (
          <>
            <section className="settings-section">
              <h3>Antonio Evo API Keys</h3>
              <p className="section-description">
                Generate API keys to access Antonio Evo programmatically. Keep your keys secure.
              </p>

              {/* New Key Alert */}
              {showNewKey && (
                <div className="new-key-alert">
                  <AlertCircle size={20} />
                  <div className="new-key-content">
                    <strong>Save your new API key</strong>
                    <p>This key will only be shown once. Copy it now.</p>
                    <code>{showNewKey}</code>
                  </div>
                  <button
                    className="copy-key-btn"
                    onClick={() => {
                      copyToClipboard(showNewKey, 'new');
                      setShowNewKey(null);
                    }}
                  >
                    {copiedKey === 'new' ? <Check size={16} /> : <Copy size={16} />}
                  </button>
                </div>
              )}

              {/* Create New Key */}
              <div className="create-key-form">
                <input
                  type="text"
                  placeholder="Key name (e.g., Production, Development)"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  className="key-name-input"
                />
                <button
                  className="create-key-btn"
                  onClick={generateApiKey}
                  disabled={!newKeyName.trim()}
                >
                  <Plus size={16} />
                  Generate Key
                </button>
              </div>

              {/* API Keys List */}
              <div className="api-keys-list">
                {apiKeys.length === 0 ? (
                  <div className="empty-keys">
                    <Key size={32} />
                    <p>No API keys yet</p>
                    <span>Create your first API key to get started</span>
                  </div>
                ) : (
                  apiKeys.map((key) => (
                    <div key={key.id} className="api-key-item">
                      <div className="key-info">
                        <span className="key-name">{key.name}</span>
                        <span className="key-value">{maskApiKey(key.key)}</span>
                        <span className="key-date">
                          Created {new Date(key.created).toLocaleDateString()}
                        </span>
                      </div>
                      <div className="key-actions">
                        <button
                          className="key-action-btn"
                          onClick={() => copyToClipboard(key.key, key.id)}
                          title="Copy key"
                        >
                          {copiedKey === key.id ? <Check size={16} /> : <Copy size={16} />}
                        </button>
                        <button
                          className="key-action-btn danger"
                          onClick={() => deleteApiKey(key.id)}
                          title="Delete key"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>

            <section className="settings-section">
              <h3>API Documentation</h3>
              <div className="api-docs-card">
                <p>Use your API key to authenticate requests:</p>
                <code className="code-block">
                  curl -X POST http://localhost:8420/api/ask \<br />
                  {'  '}-H "Authorization: Bearer YOUR_API_KEY" \<br />
                  {'  '}-H "Content-Type: application/json" \<br />
                  {'  '}-d '{`{"question": "Hello"}`}'
                </code>
              </div>
            </section>
          </>
        )}

        {/* External APIs Tab */}
        {activeTab === 'external' && (
          <>
            <section className="settings-section">
              <h3>External LLM Providers</h3>
              <p className="section-description">
                Connect external AI providers for hybrid mode or fallback capabilities.
              </p>

              {/* OpenAI */}
              <div className="external-api-card">
                <div className="api-header">
                  <div className="api-title">
                    <span className="api-logo openai">GPT</span>
                    <span>OpenAI</span>
                  </div>
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={externalApis.openai?.enabled || false}
                      onChange={(e) => updateExternalApi('openai', { enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
                {externalApis.openai?.enabled && (
                  <div className="api-config">
                    <div className="config-row">
                      <label>API Key</label>
                      <div className="key-input-wrapper">
                        <input
                          type={showApiKey.openai ? 'text' : 'password'}
                          value={externalApis.openai?.apiKey || ''}
                          onChange={(e) => updateExternalApi('openai', { apiKey: e.target.value })}
                          placeholder="sk-..."
                        />
                        <button
                          className="toggle-visibility"
                          onClick={() => setShowApiKey({ ...showApiKey, openai: !showApiKey.openai })}
                        >
                          {showApiKey.openai ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                    <div className="config-row">
                      <label>Model</label>
                      <select
                        value={externalApis.openai?.model || 'gpt-4'}
                        onChange={(e) => updateExternalApi('openai', { model: e.target.value })}
                      >
                        <option value="gpt-4">GPT-4</option>
                        <option value="gpt-4-turbo-preview">GPT-4 Turbo</option>
                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>

              {/* Anthropic */}
              <div className="external-api-card">
                <div className="api-header">
                  <div className="api-title">
                    <span className="api-logo anthropic">A</span>
                    <span>Anthropic</span>
                  </div>
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={externalApis.anthropic?.enabled || false}
                      onChange={(e) => updateExternalApi('anthropic', { enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
                {externalApis.anthropic?.enabled && (
                  <div className="api-config">
                    <div className="config-row">
                      <label>API Key</label>
                      <div className="key-input-wrapper">
                        <input
                          type={showApiKey.anthropic ? 'text' : 'password'}
                          value={externalApis.anthropic?.apiKey || ''}
                          onChange={(e) => updateExternalApi('anthropic', { apiKey: e.target.value })}
                          placeholder="sk-ant-..."
                        />
                        <button
                          className="toggle-visibility"
                          onClick={() => setShowApiKey({ ...showApiKey, anthropic: !showApiKey.anthropic })}
                        >
                          {showApiKey.anthropic ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                    <div className="config-row">
                      <label>Model</label>
                      <select
                        value={externalApis.anthropic?.model || 'claude-3-opus-20240229'}
                        onChange={(e) => updateExternalApi('anthropic', { model: e.target.value })}
                      >
                        <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                        <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
                        <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>

              {/* Google */}
              <div className="external-api-card">
                <div className="api-header">
                  <div className="api-title">
                    <span className="api-logo google">G</span>
                    <span>Google AI</span>
                  </div>
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={externalApis.google?.enabled || false}
                      onChange={(e) => updateExternalApi('google', { enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
                {externalApis.google?.enabled && (
                  <div className="api-config">
                    <div className="config-row">
                      <label>API Key</label>
                      <div className="key-input-wrapper">
                        <input
                          type={showApiKey.google ? 'text' : 'password'}
                          value={externalApis.google?.apiKey || ''}
                          onChange={(e) => updateExternalApi('google', { apiKey: e.target.value })}
                          placeholder="AIza..."
                        />
                        <button
                          className="toggle-visibility"
                          onClick={() => setShowApiKey({ ...showApiKey, google: !showApiKey.google })}
                        >
                          {showApiKey.google ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                    <div className="config-row">
                      <label>Model</label>
                      <select
                        value={externalApis.google?.model || 'gemini-pro'}
                        onChange={(e) => updateExternalApi('google', { model: e.target.value })}
                      >
                        <option value="gemini-pro">Gemini Pro</option>
                        <option value="gemini-pro-vision">Gemini Pro Vision</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>
            </section>

            {/* Custom APIs */}
            <section className="settings-section">
              <h3>Custom API Endpoints</h3>
              <p className="section-description">
                Add your own OpenAI-compatible API endpoints (Ollama, LM Studio, etc.)
              </p>

              {/* Custom APIs List */}
              {externalApis.custom?.map((api) => (
                <div key={api.id} className="external-api-card custom">
                  <div className="api-header">
                    <div className="api-title">
                      <span className="api-logo custom">{api.name[0]}</span>
                      <span>{api.name}</span>
                    </div>
                    <button
                      className="remove-api-btn"
                      onClick={() => removeCustomApi(api.id)}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                  <div className="api-details">
                    <span><ExternalLink size={12} /> {api.baseUrl}</span>
                    {api.model && <span>Model: {api.model}</span>}
                  </div>
                </div>
              ))}

              {/* Add Custom API Form */}
              <div className="add-custom-api">
                <h4>Add Custom Endpoint</h4>
                <div className="custom-api-form">
                  <input
                    type="text"
                    placeholder="Name (e.g., Ollama Local)"
                    value={newCustomApi.name}
                    onChange={(e) => setNewCustomApi({ ...newCustomApi, name: e.target.value })}
                  />
                  <input
                    type="text"
                    placeholder="Base URL (e.g., http://localhost:11434)"
                    value={newCustomApi.baseUrl}
                    onChange={(e) => setNewCustomApi({ ...newCustomApi, baseUrl: e.target.value })}
                  />
                  <input
                    type="text"
                    placeholder="Model (e.g., llama2)"
                    value={newCustomApi.model}
                    onChange={(e) => setNewCustomApi({ ...newCustomApi, model: e.target.value })}
                  />
                  <input
                    type="password"
                    placeholder="API Key (optional)"
                    value={newCustomApi.apiKey}
                    onChange={(e) => setNewCustomApi({ ...newCustomApi, apiKey: e.target.value })}
                  />
                  <button
                    className="add-api-btn"
                    onClick={addCustomApi}
                    disabled={!newCustomApi.name || !newCustomApi.baseUrl}
                  >
                    <Plus size={16} />
                    Add Endpoint
                  </button>
                </div>
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}

export default SettingsPanel;
