import React, { useState, useEffect } from 'react';
import { windowControls } from '../api';

/**
 * TitleBar with Runtime Profile Badge
 *
 * Shows: Assistant name • Profile • Local/Hybrid indicator • Connection status
 * Per UI System Prompt: "The user always knows where computation happens"
 */
function TitleBar({ isConnected, isElectron, uiState, onShowRuntimeOverview }) {
  const handleMinimize = () => windowControls.minimize();
  const handleMaximize = () => windowControls.maximize();
  const handleClose = () => windowControls.close();

  // Extract profile info from uiState
  const profileName = uiState?.profile?.name || 'loading...';
  const isLocalFirst = uiState?.profile?.is_local_first ?? true;

  // Profile badge colors
  const profileColors = {
    'evo-lite': '#f59e0b',      // amber
    'evo-standard': '#3b82f6',  // blue
    'evo-full': '#10b981',      // green
    'evo-hybrid': '#8b5cf6',    // purple
  };

  const profileColor = profileColors[profileName] || '#6b7280';

  return (
    <div className="title-bar" style={!isElectron ? { WebkitAppRegion: 'no-drag' } : {}}>
      <div className="title-bar-drag">
        {/* Logo and Name */}
        <div className="title-bar-logo">
          <span className="logo-icon">A</span>
          <span className="logo-text">Antonio Evo</span>
        </div>

        {/* Runtime Status Badges */}
        <div
          className="runtime-badges"
          onClick={onShowRuntimeOverview}
          style={{ cursor: 'pointer' }}
          title="Click for Runtime Overview"
        >
          {/* Profile Badge */}
          <span
            className="badge profile-badge"
            style={{ backgroundColor: profileColor }}
          >
            {profileName.toUpperCase().replace('EVO-', '')}
          </span>

          {/* Local/Hybrid Indicator */}
          <span className={`badge mode-badge ${isLocalFirst ? 'local' : 'hybrid'}`}>
            {isLocalFirst ? 'Local-first' : 'Hybrid'}
          </span>

          {/* Connection Status */}
          <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot"></span>
            <span className="status-text">{isConnected ? 'Connected' : 'Offline'}</span>
          </div>
        </div>
      </div>

      {/* Window Controls (Electron only) */}
      {isElectron && (
        <div className="title-bar-controls">
          <button className="title-bar-btn minimize" onClick={handleMinimize} title="Minimize">
            <svg width="12" height="12" viewBox="0 0 12 12">
              <rect y="5" width="12" height="2" fill="currentColor" />
            </svg>
          </button>
          <button className="title-bar-btn maximize" onClick={handleMaximize} title="Maximize">
            <svg width="12" height="12" viewBox="0 0 12 12">
              <rect x="1" y="1" width="10" height="10" fill="none" stroke="currentColor" strokeWidth="2" />
            </svg>
          </button>
          <button className="title-bar-btn close" onClick={handleClose} title="Close">
            <svg width="12" height="12" viewBox="0 0 12 12">
              <path d="M1 1L11 11M1 11L11 1" stroke="currentColor" strokeWidth="2" />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}

export default TitleBar;
