import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { Key, Plus, Trash2, Eye, EyeOff, Copy, Check, ExternalLink, AlertCircle, Webhook, Play, Loader2, Search, Globe, Database, Upload, RefreshCw } from 'lucide-react';
import { useTranslation } from '../i18n';

function SettingsPanel({ settings, onSettingsChange, onClose, isConnected }) {
  const [localSettings, setLocalSettings] = useState(settings);
  const [serverInfo, setServerInfo] = useState(null);
  const [activeTab, setActiveTab] = useState('general');
  const { t, language, changeLanguage } = useTranslation();

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

  // Webhooks state
  const [webhooks, setWebhooks] = useState([]);
  const [newWebhook, setNewWebhook] = useState({ name: '', url: '', trigger: 'post_response' });
  const [webhookTestStatus, setWebhookTestStatus] = useState({});
  const [webhookTestLoading, setWebhookTestLoading] = useState({});

  // Web Search state
  const [webSearchConfig, setWebSearchConfig] = useState({ api_key: '', auto_search: false, available: false });
  const [webSearchApiKey, setWebSearchApiKey] = useState('');
  const [webSearchAutoSearch, setWebSearchAutoSearch] = useState(false);
  const [webSearchTestResult, setWebSearchTestResult] = useState(null);
  const [webSearchTestLoading, setWebSearchTestLoading] = useState(false);
  const [showWebSearchKey, setShowWebSearchKey] = useState(false);

  // Knowledge Base state (v7.0)
  const [ragStats, setRagStats] = useState(null);
  const [ragIndexing, setRagIndexing] = useState(false);
  const [ragUploading, setRagUploading] = useState(false);
  const [ragUploadResult, setRagUploadResult] = useState(null);

  useEffect(() => {
    loadServerInfo();
    loadApiKeys();
    loadExternalApis();
    loadWebhooks();
    loadWebSearchConfig();
    loadRagStats();
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

  // === Webhook functions ===
  const loadWebhooks = async () => {
    try {
      const res = await fetch('http://localhost:8420/api/webhooks');
      const data = await res.json();
      if (data.success) {
        setWebhooks(data.webhooks || []);
      }
    } catch (e) {
      console.error('Failed to load webhooks:', e);
    }
  };

  const addWebhookHandler = async () => {
    if (!newWebhook.name.trim() || !newWebhook.url.trim()) return;
    try {
      const res = await fetch('http://localhost:8420/api/webhooks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newWebhook)
      });
      const data = await res.json();
      if (data.success) {
        setWebhooks([...webhooks, data.webhook]);
        setNewWebhook({ name: '', url: '', trigger: 'post_response' });
      }
    } catch (e) {
      console.error('Failed to add webhook:', e);
    }
  };

  const toggleWebhook = async (webhookId, enabled) => {
    try {
      const res = await fetch(`http://localhost:8420/api/webhooks/${webhookId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      const data = await res.json();
      if (data.success) {
        setWebhooks(webhooks.map(w => w.id === webhookId ? { ...w, enabled } : w));
      }
    } catch (e) {
      console.error('Failed to toggle webhook:', e);
    }
  };

  const deleteWebhook = async (webhookId) => {
    try {
      await fetch(`http://localhost:8420/api/webhooks/${webhookId}`, { method: 'DELETE' });
      setWebhooks(webhooks.filter(w => w.id !== webhookId));
    } catch (e) {
      console.error('Failed to delete webhook:', e);
    }
  };

  const testWebhook = async (webhookId) => {
    setWebhookTestLoading(prev => ({ ...prev, [webhookId]: true }));
    setWebhookTestStatus(prev => ({ ...prev, [webhookId]: null }));
    try {
      const res = await fetch(`http://localhost:8420/api/webhooks/${webhookId}/test`, { method: 'POST' });
      const data = await res.json();
      setWebhookTestStatus(prev => ({ ...prev, [webhookId]: data.success ? 'success' : 'error' }));
    } catch (e) {
      setWebhookTestStatus(prev => ({ ...prev, [webhookId]: 'error' }));
    }
    setWebhookTestLoading(prev => ({ ...prev, [webhookId]: false }));
    setTimeout(() => {
      setWebhookTestStatus(prev => ({ ...prev, [webhookId]: null }));
    }, 5000);
  };

  // === Web Search functions ===
  const loadWebSearchConfig = async () => {
    try {
      const res = await fetch('http://localhost:8420/api/web-search/status');
      const data = await res.json();
      setWebSearchConfig(data);
      setWebSearchApiKey(data.has_api_key ? '••••••••' : '');
      setWebSearchAutoSearch(data.auto_search || false);
    } catch (e) {
      console.error('Failed to load web search config:', e);
    }
  };

  const saveWebSearchConfig = async () => {
    try {
      const body = { auto_search: webSearchAutoSearch };
      // Only send api_key if user entered a new one (not the masked placeholder)
      if (webSearchApiKey && !webSearchApiKey.startsWith('••')) {
        body.api_key = webSearchApiKey;
      }
      const res = await fetch('http://localhost:8420/api/web-search/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if (data.success) {
        loadWebSearchConfig();
      }
    } catch (e) {
      console.error('Failed to save web search config:', e);
    }
  };

  const testWebSearch = async () => {
    setWebSearchTestLoading(true);
    setWebSearchTestResult(null);
    try {
      const res = await fetch('http://localhost:8420/api/web-search/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'test search' })
      });
      const data = await res.json();
      setWebSearchTestResult(data);
    } catch (e) {
      setWebSearchTestResult({ success: false, error: 'Connection error' });
    }
    setWebSearchTestLoading(false);
    setTimeout(() => setWebSearchTestResult(null), 8000);
  };

  // === Knowledge Base / RAG functions (v7.0) ===
  const loadRagStats = async () => {
    try {
      const res = await fetch('http://localhost:8420/api/rag/stats');
      const data = await res.json();
      if (data.success !== false) {
        setRagStats(data);
      }
    } catch (e) {
      console.error('Failed to load RAG stats:', e);
      setRagStats(null);
    }
  };

  const reindexDocuments = async () => {
    setRagIndexing(true);
    try {
      const res = await fetch('http://localhost:8420/api/rag/index', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        await loadRagStats();
      }
    } catch (e) {
      console.error('Failed to re-index:', e);
    }
    setRagIndexing(false);
  };

  const handleRagFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setRagUploading(true);
    setRagUploadResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('http://localhost:8420/api/rag/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setRagUploadResult(data);
      if (data.success !== false) {
        await loadRagStats();
      }
    } catch (err) {
      setRagUploadResult({ success: false, error: 'Upload failed' });
    }
    setRagUploading(false);
    e.target.value = '';
    setTimeout(() => setRagUploadResult(null), 6000);
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
        <h2>{t('settings.title')}</h2>
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
          {t('settings.general')}
        </button>
        <button
          className={`settings-tab ${activeTab === 'apikeys' ? 'active' : ''}`}
          onClick={() => setActiveTab('apikeys')}
        >
          {t('settings.apiKeys')}
        </button>
        <button
          className={`settings-tab ${activeTab === 'external' ? 'active' : ''}`}
          onClick={() => setActiveTab('external')}
        >
          {t('settings.externalApis')}
        </button>
        <button
          className={`settings-tab ${activeTab === 'webhooks' ? 'active' : ''}`}
          onClick={() => setActiveTab('webhooks')}
        >
          {t('settings.webhooks')}
        </button>
        <button
          className={`settings-tab ${activeTab === 'knowledge' ? 'active' : ''}`}
          onClick={() => setActiveTab('knowledge')}
        >
          {t('settings.knowledgeBase')}
        </button>
      </div>

      <div className="settings-content">
        {/* General Tab */}
        {activeTab === 'general' && (
          <>
            {/* Language Section */}
            <section className="settings-section">
              <h3>{t('settings.language')}</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.uiLanguage')}</label>
                  <p className="setting-description">{t('settings.uiLanguageDesc')}</p>
                </div>
                <select
                  value={language}
                  onChange={(e) => changeLanguage(e.target.value)}
                  className="language-select"
                >
                  <option value="en">English</option>
                  <option value="it">Italiano</option>
                  <option value="fr">Fran&#231;ais</option>
                  <option value="es">Espa&#241;ol</option>
                </select>
              </div>
            </section>

            {/* Connection Section */}
            <section className="settings-section">
              <h3>{t('settings.connection')}</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.apiServerStatus')}</label>
                  <p className="setting-description">{t('settings.connectionToBackend')}</p>
                </div>
                <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
                  {isConnected ? t('settings.connected') : t('settings.disconnected')}
                </div>
              </div>

              {serverInfo && (
                <div className="server-info-card">
                  <div className="info-row">
                    <span>{t('settings.serverStatus')}</span>
                    <span>{serverInfo.status}</span>
                  </div>
                  <div className="info-row">
                    <span>{t('settings.llmAvailable')}</span>
                    <span>{serverInfo.llm_available ? t('settings.yes') : t('settings.no')}</span>
                  </div>
                </div>
              )}
            </section>

            {/* Voice Section (v7.0 enhanced) */}
            <section className="settings-section">
              <h3>{t('settings.voice')}</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.speakResponses')}</label>
                  <p className="setting-description">{t('settings.speakResponsesDesc')}</p>
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
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.voiceLanguage')}</label>
                  <p className="setting-description">{t('settings.voiceLanguageDesc')}</p>
                </div>
                <select
                  value={language}
                  onChange={(e) => changeLanguage(e.target.value)}
                  className="language-select"
                >
                  <option value="en">English</option>
                  <option value="it">Italiano</option>
                  <option value="fr">Fran&ccedil;ais</option>
                  <option value="es">Espa&ntilde;ol</option>
                </select>
              </div>
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.voiceSpeed')}</label>
                  <p className="setting-description">{t('settings.voiceSpeedDesc')}</p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="range"
                    min="0.5"
                    max="2.0"
                    step="0.1"
                    value={localSettings.voiceSpeed || 1.0}
                    onChange={(e) => handleChange('voiceSpeed', parseFloat(e.target.value))}
                    style={{ width: '120px' }}
                  />
                  <span style={{ fontSize: '12px', opacity: 0.7 }}>{localSettings.voiceSpeed || 1.0}x</span>
                </div>
              </div>
            </section>

            {/* AI Reasoning */}
            <section className="settings-section">
              <h3>{t('settings.aiReasoning')}</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.deepThinking')}</label>
                  <p className="setting-description">{t('settings.deepThinkingDesc')}</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={localSettings.deepThinking}
                    onChange={(e) => handleChange('deepThinking', e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            </section>

            {/* Web Search Section */}
            <section className="settings-section">
              <h3><Globe size={18} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px' }} />{t('settings.webSearch')}</h3>
              <p className="section-description">
                {t('settings.webSearchDesc')}
              </p>

              {/* Status indicator */}
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.webSearchStatus')}</label>
                </div>
                <div className={`status-indicator ${webSearchConfig.available ? 'connected' : 'disconnected'}`}>
                  {webSearchConfig.available ? t('settings.webSearchConfigured') : t('settings.webSearchNotConfigured')}
                </div>
              </div>

              {/* API Key input */}
              <div className="setting-item" style={{ flexDirection: 'column', alignItems: 'stretch', gap: '8px' }}>
                <div className="setting-info">
                  <label>{t('settings.webSearchApiKey')}</label>
                  <p className="setting-description">{t('settings.webSearchGetKey')}</p>
                </div>
                <div className="key-input-wrapper">
                  <input
                    type={showWebSearchKey ? 'text' : 'password'}
                    value={webSearchApiKey}
                    onChange={(e) => setWebSearchApiKey(e.target.value)}
                    placeholder={t('settings.webSearchApiKeyPlaceholder')}
                    onFocus={() => { if (webSearchApiKey.startsWith('••')) setWebSearchApiKey(''); }}
                  />
                  <button
                    className="toggle-visibility"
                    onClick={() => setShowWebSearchKey(!showWebSearchKey)}
                  >
                    {showWebSearchKey ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {/* Auto Search toggle */}
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.webSearchAutoSearch')}</label>
                  <p className="setting-description">{t('settings.webSearchAutoSearchDesc')}</p>
                </div>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={webSearchAutoSearch}
                    onChange={(e) => setWebSearchAutoSearch(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>

              {/* Save + Test buttons */}
              <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                <button
                  className="add-api-btn"
                  onClick={saveWebSearchConfig}
                >
                  {t('settings.webSearchSave')}
                </button>
                <button
                  className={`add-api-btn ${webSearchTestResult?.success ? 'success' : webSearchTestResult && !webSearchTestResult.success ? 'danger' : ''}`}
                  onClick={testWebSearch}
                  disabled={webSearchTestLoading || !webSearchConfig.available}
                  style={{ background: webSearchTestResult?.success ? '#22c55e' : webSearchTestResult && !webSearchTestResult.success ? '#ef4444' : undefined }}
                >
                  {webSearchTestLoading ? (
                    <><Loader2 size={16} className="spinning" /> {t('settings.webSearchTesting')}</>
                  ) : (
                    <><Search size={16} /> {t('settings.webSearchTest')}</>
                  )}
                </button>
              </div>

              {/* Test result feedback */}
              {webSearchTestResult && (
                <div className={`new-key-alert ${webSearchTestResult.success ? '' : 'error'}`} style={{ marginTop: '8px' }}>
                  <AlertCircle size={18} />
                  <span>
                    {webSearchTestResult.success
                      ? t('settings.webSearchTestSuccess', { count: webSearchTestResult.results_count || 0 })
                      : t('settings.webSearchTestError', { error: webSearchTestResult.error || 'Unknown error' })
                    }
                  </span>
                </div>
              )}
            </section>

            {/* Real-time Section */}
            <section className="settings-section">
              <h3>{t('settings.realtime')}</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.websocketConnection')}</label>
                  <p className="setting-description">{t('settings.websocketDesc')}</p>
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
                  <label>{t('settings.avatarCompanion')}</label>
                  <p className="setting-description">{t('settings.avatarCompanionDesc')}</p>
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
              <h3>{t('settings.appearance')}</h3>
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.darkMode')}</label>
                  <p className="setting-description">{t('settings.darkModeDesc')}</p>
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
              <h3>{t('settings.about')}</h3>
              <div className="about-card">
                <div className="about-logo">A</div>
                <div className="about-info">
                  <h4>Antonio Evo</h4>
                  <p>{t('settings.version', { version: '4.0.0' })}</p>
                  <p className="about-description">
                    {t('settings.aboutDescription')}
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
              <h3>{t('settings.antonioApiKeys')}</h3>
              <p className="section-description">
                {t('settings.apiKeysDescription')}
              </p>

              {/* New Key Alert */}
              {showNewKey && (
                <div className="new-key-alert">
                  <AlertCircle size={20} />
                  <div className="new-key-content">
                    <strong>{t('settings.saveNewApiKey')}</strong>
                    <p>{t('settings.keyShownOnce')}</p>
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
                  placeholder={t('settings.keyNamePlaceholder')}
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
                  {t('settings.generateKey')}
                </button>
              </div>

              {/* API Keys List */}
              <div className="api-keys-list">
                {apiKeys.length === 0 ? (
                  <div className="empty-keys">
                    <Key size={32} />
                    <p>{t('settings.noApiKeys')}</p>
                    <span>{t('settings.createFirstKey')}</span>
                  </div>
                ) : (
                  apiKeys.map((key) => (
                    <div key={key.id} className="api-key-item">
                      <div className="key-info">
                        <span className="key-name">{key.name}</span>
                        <span className="key-value">{maskApiKey(key.key)}</span>
                        <span className="key-date">
                          {t('settings.created', { date: new Date(key.created).toLocaleDateString() })}
                        </span>
                      </div>
                      <div className="key-actions">
                        <button
                          className="key-action-btn"
                          onClick={() => copyToClipboard(key.key, key.id)}
                          title={t('message.copyKey')}
                        >
                          {copiedKey === key.id ? <Check size={16} /> : <Copy size={16} />}
                        </button>
                        <button
                          className="key-action-btn danger"
                          onClick={() => deleteApiKey(key.id)}
                          title={t('message.deleteKey')}
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
              <h3>{t('settings.apiDocumentation')}</h3>
              <div className="api-docs-card">
                <p>{t('settings.apiDocDesc')}</p>
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
              <h3>{t('settings.externalLlmProviders')}</h3>
              <p className="section-description">
                {t('settings.externalLlmDesc')}
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
                      <label>{t('settings.apiKey')}</label>
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
                      <label>{t('settings.model')}</label>
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
                      <label>{t('settings.apiKey')}</label>
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
                      <label>{t('settings.model')}</label>
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
                      <label>{t('settings.apiKey')}</label>
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
                      <label>{t('settings.model')}</label>
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
              <h3>{t('settings.customApiEndpoints')}</h3>
              <p className="section-description">
                {t('settings.customApiDesc')}
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
                    {api.model && <span>{t('settings.model')}: {api.model}</span>}
                  </div>
                </div>
              ))}

              {/* Add Custom API Form */}
              <div className="add-custom-api">
                <h4>{t('settings.addCustomEndpoint')}</h4>
                <div className="custom-api-form">
                  <input
                    type="text"
                    placeholder={t('settings.namePlaceholder')}
                    value={newCustomApi.name}
                    onChange={(e) => setNewCustomApi({ ...newCustomApi, name: e.target.value })}
                  />
                  <input
                    type="text"
                    placeholder={t('settings.baseUrlPlaceholder')}
                    value={newCustomApi.baseUrl}
                    onChange={(e) => setNewCustomApi({ ...newCustomApi, baseUrl: e.target.value })}
                  />
                  <input
                    type="text"
                    placeholder={t('settings.modelPlaceholder')}
                    value={newCustomApi.model}
                    onChange={(e) => setNewCustomApi({ ...newCustomApi, model: e.target.value })}
                  />
                  <input
                    type="password"
                    placeholder={t('settings.apiKeyOptional')}
                    value={newCustomApi.apiKey}
                    onChange={(e) => setNewCustomApi({ ...newCustomApi, apiKey: e.target.value })}
                  />
                  <button
                    className="add-api-btn"
                    onClick={addCustomApi}
                    disabled={!newCustomApi.name || !newCustomApi.baseUrl}
                  >
                    <Plus size={16} />
                    {t('settings.addEndpoint')}
                  </button>
                </div>
              </div>
            </section>
          </>
        )}

        {/* Knowledge Base Tab (v7.0) */}
        {activeTab === 'knowledge' && (
          <>
            <section className="settings-section">
              <h3><Database size={18} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '6px' }} />{t('settings.knowledgeBase')}</h3>
              <p className="section-description">
                {t('settings.knowledgeDesc')}
              </p>

              {/* RAG Status */}
              <div className="setting-item">
                <div className="setting-info">
                  <label>{t('settings.ragStatus')}</label>
                </div>
                <div className={`status-indicator ${ragStats?.available ? 'connected' : 'disconnected'}`}>
                  {ragStats?.available ? t('settings.ragAvailable') : t('settings.ragUnavailable')}
                </div>
              </div>

              {/* Stats */}
              {ragStats?.available && (
                <div className="server-info-card">
                  <div className="info-row">
                    <span>{t('settings.indexedChunks')}</span>
                    <span>{ragStats.total_chunks ?? 0}</span>
                  </div>
                  <div className="info-row">
                    <span>{t('settings.embeddingModel')}</span>
                    <span>{ragStats.embedding_model || 'all-MiniLM-L6-v2'}</span>
                  </div>
                  <div className="info-row">
                    <span>{t('settings.docsPath')}</span>
                    <span>{ragStats.docs_path || 'data/knowledge'}</span>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                <button
                  className="add-api-btn"
                  onClick={reindexDocuments}
                  disabled={ragIndexing || !ragStats?.available}
                >
                  {ragIndexing ? (
                    <><Loader2 size={16} className="spinning" /> {t('settings.indexing')}</>
                  ) : (
                    <><RefreshCw size={16} /> {t('settings.reindexDocuments')}</>
                  )}
                </button>
                <label className="add-api-btn" style={{ cursor: ragUploading || !ragStats?.available ? 'not-allowed' : 'pointer', opacity: ragUploading || !ragStats?.available ? 0.5 : 1 }}>
                  {ragUploading ? (
                    <><Loader2 size={16} className="spinning" /> {t('settings.uploading')}</>
                  ) : (
                    <><Upload size={16} /> {t('settings.uploadDocument')}</>
                  )}
                  <input
                    type="file"
                    accept=".md,.txt"
                    onChange={handleRagFileUpload}
                    disabled={ragUploading || !ragStats?.available}
                    style={{ display: 'none' }}
                  />
                </label>
              </div>

              {/* Upload result feedback */}
              {ragUploadResult && (
                <div className={`new-key-alert ${ragUploadResult.success !== false ? '' : 'error'}`} style={{ marginTop: '8px' }}>
                  <AlertCircle size={18} />
                  <span>
                    {ragUploadResult.success !== false
                      ? t('settings.uploadSuccess', { name: ragUploadResult.filename || 'file' })
                      : t('settings.uploadError', { error: ragUploadResult.error || 'Unknown error' })
                    }
                  </span>
                </div>
              )}
            </section>

            {/* How it works */}
            <section className="settings-section">
              <h3>{t('settings.howRagWorks')}</h3>
              <div className="api-docs-card">
                <p>{t('settings.ragHowDesc')}</p>
                <div style={{ marginTop: '8px', fontSize: '0.85em', color: 'var(--text-secondary)' }}>
                  1. {t('settings.ragStep1')}<br />
                  2. {t('settings.ragStep2')}<br />
                  3. {t('settings.ragStep3')}
                </div>
              </div>
            </section>
          </>
        )}

        {/* Webhooks Tab */}
        {activeTab === 'webhooks' && (
          <>
            <section className="settings-section">
              <h3>{t('settings.n8nWebhooks')}</h3>
              <p className="section-description">
                {t('settings.webhooksDesc')}
              </p>

              {/* Webhooks List */}
              {webhooks.length === 0 ? (
                <div className="empty-keys">
                  <Webhook size={32} />
                  <p>{t('settings.noWebhooks')}</p>
                  <span>{t('settings.addFirstWebhook')}</span>
                </div>
              ) : (
                <div className="api-keys-list">
                  {webhooks.map((wh) => (
                    <div key={wh.id} className="external-api-card">
                      <div className="api-header">
                        <div className="api-title">
                          <span className={`webhook-status-dot ${wh.enabled ? 'active' : 'inactive'}`} />
                          <span>{wh.name}</span>
                          <span className="webhook-trigger-badge">{wh.trigger}</span>
                        </div>
                        <label className="toggle">
                          <input
                            type="checkbox"
                            checked={wh.enabled}
                            onChange={(e) => toggleWebhook(wh.id, e.target.checked)}
                          />
                          <span className="toggle-slider"></span>
                        </label>
                      </div>
                      <div className="api-details" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          <ExternalLink size={12} /> {wh.url}
                        </span>
                        <div style={{ display: 'flex', gap: '6px', marginLeft: '8px' }}>
                          <button
                            className={`key-action-btn ${webhookTestStatus[wh.id] === 'success' ? 'success' : webhookTestStatus[wh.id] === 'error' ? 'danger' : ''}`}
                            onClick={() => testWebhook(wh.id)}
                            disabled={webhookTestLoading[wh.id]}
                            title={t('settings.testWebhook')}
                          >
                            {webhookTestLoading[wh.id] ? <Loader2 size={16} className="spinning" /> :
                              webhookTestStatus[wh.id] === 'success' ? <Check size={16} /> :
                              <Play size={16} />}
                          </button>
                          <button
                            className="key-action-btn danger"
                            onClick={() => deleteWebhook(wh.id)}
                            title={t('settings.deleteWebhook')}
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* Add Webhook Form */}
            <section className="settings-section">
              <h3>{t('settings.addWebhook')}</h3>
              <div className="add-custom-api">
                <div className="custom-api-form">
                  <input
                    type="text"
                    placeholder={t('settings.webhookNamePlaceholder')}
                    value={newWebhook.name}
                    onChange={(e) => setNewWebhook({ ...newWebhook, name: e.target.value })}
                  />
                  <input
                    type="text"
                    placeholder={t('settings.webhookUrlPlaceholder')}
                    value={newWebhook.url}
                    onChange={(e) => setNewWebhook({ ...newWebhook, url: e.target.value })}
                  />
                  <select
                    value={newWebhook.trigger}
                    onChange={(e) => setNewWebhook({ ...newWebhook, trigger: e.target.value })}
                  >
                    <option value="post_response">{t('settings.afterResponse')}</option>
                    <option value="on_memory">{t('settings.onMemoryCreate')}</option>
                    <option value="on_error">{t('settings.onError')}</option>
                    <option value="manual">{t('settings.manualTrigger')}</option>
                  </select>
                  <button
                    className="add-api-btn"
                    onClick={addWebhookHandler}
                    disabled={!newWebhook.name.trim() || !newWebhook.url.trim()}
                  >
                    <Plus size={16} />
                    {t('settings.addWebhookBtn')}
                  </button>
                </div>
              </div>
            </section>

            {/* Webhook Docs */}
            <section className="settings-section">
              <h3>{t('settings.howItWorks')}</h3>
              <div className="api-docs-card">
                <p>{t('settings.webhookDocDesc')}</p>
                <code className="code-block" style={{ fontSize: '0.8em' }}>
                  {'{'}<br />
                  {'  '}"type": "post_response",<br />
                  {'  '}"source": "antonio-evo",<br />
                  {'  '}"timestamp": "2026-02-13T...",<br />
                  {'  '}"data": {'{'} "request_text": "...", "response_text": "...", "elapsed_ms": 42 {'}'}<br />
                  {'}'}
                </code>
                <div style={{ marginTop: '12px', fontSize: '0.85em', color: 'var(--text-secondary)' }}>
                  <strong>{t('settings.triggerTypes')}</strong><br />
                  <span>{t('settings.afterResponse')}</span> - {t('settings.triggerAfterResponse')}<br />
                  <span>{t('settings.onMemoryCreate')}</span> - {t('settings.triggerOnMemory')}<br />
                  <span>{t('settings.onError')}</span> - {t('settings.triggerOnError')}<br />
                  <span>{t('settings.manualTrigger')}</span> - {t('settings.triggerManual')}
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
