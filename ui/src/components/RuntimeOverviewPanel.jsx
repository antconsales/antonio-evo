import React, { useState, useEffect } from 'react';
import { useTranslation } from '../i18n';

function RuntimeOverviewPanel({ uiState, onClose }) {
  const [llmStatus, setLlmStatus] = useState(null);
  const [memoryStats, setMemoryStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const { t } = useTranslation();

  useEffect(() => {
    const fetchDetails = async () => {
      try {
        const [llmRes, memRes] = await Promise.all([
          fetch('http://localhost:8420/api/llm/status'),
          fetch('http://localhost:8420/api/memory/stats'),
        ]);

        const llmData = await llmRes.json();
        const memData = await memRes.json();

        setLlmStatus(llmData);
        setMemoryStats(memData);
      } catch (e) {
        console.error('[RuntimeOverview] Failed to fetch details:', e);
      } finally {
        setLoading(false);
      }
    };

    fetchDetails();
  }, []);

  if (!uiState) {
    return (
      <div className="runtime-overlay" onClick={onClose}>
        <div className="runtime-panel" onClick={(e) => e.stopPropagation()}>
          <div className="runtime-loading">{t('runtime.loadingText')}</div>
        </div>
      </div>
    );
  }

  const profile = uiState.profile || {};
  const connection = uiState.connection || {};
  const features = uiState.features || {};
  const consent = uiState.consent || {};

  const getProfileColor = (name) => {
    const colors = {
      'evo-lite': '#10b981',
      'evo-standard': '#6366f1',
      'evo-full': '#8b5cf6',
      'evo-hybrid': '#f59e0b',
    };
    return colors[name] || '#6366f1';
  };

  return (
    <div className="runtime-overlay" onClick={onClose}>
      <div className="runtime-panel" onClick={(e) => e.stopPropagation()}>
        <div className="runtime-header">
          <div className="runtime-title">
            <h2>{uiState.assistant_name || 'Antonio Evo'}</h2>
            <span className="runtime-version">v{uiState.version || '2.0'}</span>
          </div>
          <button className="runtime-close" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="runtime-content">
          {/* Profile Section */}
          <div className="runtime-section">
            <h3>{t('runtime.runtimeProfile')}</h3>
            <div
              className="profile-card"
              style={{ borderColor: getProfileColor(profile.name) }}
            >
              <div
                className="profile-indicator"
                style={{ background: getProfileColor(profile.name) }}
              />
              <div className="profile-details">
                <span className="profile-name">{profile.name || 'unknown'}</span>
                <span className="profile-type">
                  {profile.is_local_first ? t('runtime.localFirst') : t('runtime.hybridMode')}
                </span>
              </div>
            </div>
          </div>

          {/* Connections Section */}
          <div className="runtime-section">
            <h3>{t('runtime.connections')}</h3>
            <div className="connection-grid">
              <div className={`connection-item ${connection.local_llm ? 'active' : ''}`}>
                <span className="connection-dot" />
                <span>{t('runtime.localLlm')}</span>
              </div>
              <div className={`connection-item ${connection.memory ? 'active' : ''}`}>
                <span className="connection-dot" />
                <span>{t('runtime.memory')}</span>
              </div>
              <div className={`connection-item ${connection.rag ? 'active' : ''}`}>
                <span className="connection-dot" />
                <span>{t('runtime.rag')}</span>
              </div>
            </div>
          </div>

          {/* Features Section */}
          <div className="runtime-section">
            <h3>{t('runtime.activeFeatures')}</h3>
            <div className="features-grid">
              {Object.entries(features).map(([key, value]) => (
                <div
                  key={key}
                  className={`feature-item ${value ? 'enabled' : 'disabled'}`}
                >
                  <span className="feature-icon">{value ? '●' : '○'}</span>
                  <span className="feature-name">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Memory Stats */}
          {memoryStats && (
            <div className="runtime-section">
              <h3>{t('runtime.memoryTitle')}</h3>
              <div className="stats-row">
                <span>{t('runtime.totalNeurons')}</span>
                <strong>{memoryStats.total_neurons || 0}</strong>
              </div>
              <div className="stats-row">
                <span>{t('runtime.avgConfidence')}</span>
                <strong>{((memoryStats.avg_confidence || 0) * 100).toFixed(0)}%</strong>
              </div>
            </div>
          )}

          {/* LLM Status */}
          {llmStatus && llmStatus.endpoints && (
            <div className="runtime-section">
              <h3>{t('runtime.llmEndpoints')}</h3>
              {Object.entries(llmStatus.endpoints).map(([name, info]) => (
                <div key={name} className="endpoint-item">
                  <div className="endpoint-header">
                    <span className={`endpoint-status ${info.status}`} />
                    <span className="endpoint-name">{name}</span>
                  </div>
                  {info.model && (
                    <span className="endpoint-model">{info.model}</span>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Consent Status */}
          <div className="runtime-section">
            <h3>{t('runtime.consentSettings')}</h3>
            <div className="consent-grid">
              {Object.entries(consent).map(([key, value]) => (
                <div
                  key={key}
                  className={`consent-item ${value?.allowed ? 'allowed' : 'denied'}`}
                >
                  <span className="consent-icon">
                    {value?.allowed ? '✓' : '✗'}
                  </span>
                  <span className="consent-name">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                  <span className="consent-scope">{value?.scope || 'none'}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Pending Items */}
          {uiState.pending && (uiState.pending.tasks > 0 || uiState.pending.insights > 0) && (
            <div className="runtime-section pending-section">
              <h3>{t('runtime.pending')}</h3>
              {uiState.pending.tasks > 0 && (
                <div className="pending-item tasks">
                  <span className="pending-count">{uiState.pending.tasks}</span>
                  <span>{t('runtime.tasksAwaiting')}</span>
                </div>
              )}
              {uiState.pending.insights > 0 && (
                <div className="pending-item insights">
                  <span className="pending-count">{uiState.pending.insights}</span>
                  <span>{t('runtime.newInsights')}</span>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="runtime-footer">
          <span className="session-id">{t('runtime.session', { id: uiState.session_id || 'N/A' })}</span>
        </div>
      </div>
    </div>
  );
}

export default RuntimeOverviewPanel;
