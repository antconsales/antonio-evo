import React from 'react';
import { useTranslation } from '../i18n';

function Sidebar({
  onNewChat,
  onShowSettings,
  onNavigate,
  currentView,
  isConnected,
  conversations = [],
  currentConversationId,
  onSelectConversation,
  onDeleteConversation,
}) {
  const { t } = useTranslation();

  const formatDate = (isoString) => {
    const date = new Date(isoString);
    const now = new Date();
    const diff = now - date;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return t('common.today');
    if (days === 1) return t('common.yesterday');
    if (days < 7) return t('common.daysAgo', { count: days });
    return date.toLocaleDateString();
  };

  const truncate = (str, len = 30) => {
    if (!str) return t('common.emptyChat');
    return str.length > len ? str.substring(0, len) + '...' : str;
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={onNewChat}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 5v14M5 12h14" />
          </svg>
          <span>{t('nav.newChat')}</span>
        </button>
      </div>

      <div className="sidebar-content">
        {/* Navigation */}
        <div className="sidebar-section">
          <h3 className="sidebar-section-title">{t('nav.navigation')}</h3>
          <div className="sidebar-nav">
            <button
              className={`sidebar-nav-item ${currentView === 'chat' ? 'active' : ''}`}
              onClick={() => onNavigate('chat')}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              <span>{t('nav.chat')}</span>
            </button>
            <button
              className={`sidebar-nav-item ${currentView === 'images' ? 'active' : ''}`}
              onClick={() => onNavigate('images')}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
              <span>{t('nav.zImage')}</span>
            </button>
          </div>
        </div>

        {/* Chat History */}
        {conversations.length > 0 && (
          <div className="sidebar-section">
            <h3 className="sidebar-section-title">{t('nav.history', { count: conversations.length })}</h3>
            <div className="conversation-list">
              {conversations.slice(0, 10).map((conv) => (
                <div
                  key={conv.id}
                  className={`conversation-item ${conv.id === currentConversationId ? 'active' : ''}`}
                  onClick={() => {
                    onSelectConversation?.(conv.id);
                    onNavigate('chat');
                  }}
                >
                  <div className="conversation-content">
                    <span className="conversation-title">{truncate(conv.lastMessage || conv.title)}</span>
                    <span className="conversation-date">{formatDate(conv.updatedAt || conv.createdAt)}</span>
                  </div>
                  <button
                    className="conversation-delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation?.(conv.id);
                    }}
                    title={t('nav.deleteConversation')}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="sidebar-section">
          <h3 className="sidebar-section-title">{t('nav.status')}</h3>
          <div className="status-card">
            <div className="status-row">
              <span>{t('common.apiServer')}</span>
              <span className={`status-badge ${isConnected ? 'online' : 'offline'}`}>
                {isConnected ? t('common.online') : t('common.offline')}
              </span>
            </div>
            <div className="status-row">
              <span>{t('common.voiceInput')}</span>
              <span className="status-badge available">{t('common.available')}</span>
            </div>
            <div className="status-row">
              <span>{t('common.voiceOutput')}</span>
              <span className="status-badge available">{t('common.available')}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="sidebar-footer">
        <button className="settings-btn" onClick={onShowSettings}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
          <span>{t('nav.settings')}</span>
        </button>
      </div>
    </aside>
  );
}

export default Sidebar;
