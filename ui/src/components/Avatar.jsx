import React, { useState, useEffect, useRef } from 'react';
import './Avatar.css';
import { useTranslation } from '../i18n';

/**
 * Avatar Companion Component
 *
 * Floating assistant avatar that reacts to mood and state.
 * Inspired by stack-chan but modernized for web.
 *
 * Moods: neutral, helpful, curious, analytical, cautious
 * States: idle, thinking, speaking
 */

// Mood to expression mapping (labels are now translated via i18n)
const MOOD_CONFIG = {
  neutral: {
    emoji: '\uD83D\uDE10',
    color: '#6b7280',
    animation: 'idle',
    labelKey: 'avatar.neutral',
  },
  friendly: {
    emoji: '\uD83D\uDE0A',
    color: '#22c55e',
    animation: 'bounce',
    labelKey: 'avatar.friendly',
  },
  helpful: {
    emoji: '\uD83E\uDD1D',
    color: '#22c55e',
    animation: 'bounce',
    labelKey: 'avatar.helpful',
  },
  curious: {
    emoji: '\uD83E\uDD14',
    color: '#3b82f6',
    animation: 'tilt',
    labelKey: 'avatar.curious',
  },
  analytical: {
    emoji: '\uD83E\uDDD0',
    color: '#8b5cf6',
    animation: 'focus',
    labelKey: 'avatar.analytical',
  },
  cautious: {
    emoji: '\uD83D\uDE1F',
    color: '#f59e0b',
    animation: 'shake',
    labelKey: 'avatar.cautious',
  },
  thinking: {
    emoji: '\uD83D\uDCAD',
    color: '#6366f1',
    animation: 'pulse',
    labelKey: 'avatar.thinking',
  },
  error: {
    emoji: '\uD83D\uDE13',
    color: '#ef4444',
    animation: 'shake',
    labelKey: 'avatar.error',
  },
};

// Persona colors
const PERSONA_COLORS = {
  social: '#22c55e',
  logic: '#8b5cf6',
  unknown: '#6b7280',
};

function Avatar({
  mood = 'neutral',
  persona = null,
  isThinking = false,
  isExpanded = false,
  onToggle,
  onSendMessage,
  neuronCount = 0,
  isConnected = false,
}) {
  const [currentMood, setCurrentMood] = useState(mood);
  const [inputText, setInputText] = useState('');
  const [position, setPosition] = useState({ x: 20, y: 20 });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef(null);
  const inputRef = useRef(null);
  const { t } = useTranslation();

  // Update mood with animation delay
  useEffect(() => {
    if (isThinking) {
      setCurrentMood('thinking');
    } else {
      setCurrentMood(mood);
    }
  }, [mood, isThinking]);

  // Focus input when expanded
  useEffect(() => {
    if (isExpanded && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isExpanded]);

  // Get current mood config
  const moodConfig = MOOD_CONFIG[currentMood] || MOOD_CONFIG.neutral;
  const personaColor = persona ? PERSONA_COLORS[persona] : null;

  // Handle drag
  const handleMouseDown = (e) => {
    if (e.target.classList.contains('avatar-face')) {
      setIsDragging(true);
      dragRef.current = {
        startX: e.clientX - position.x,
        startY: e.clientY - position.y,
      };
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging && dragRef.current) {
      const newX = Math.max(0, Math.min(window.innerWidth - 100, e.clientX - dragRef.current.startX));
      const newY = Math.max(0, Math.min(window.innerHeight - 100, e.clientY - dragRef.current.startY));
      setPosition({ x: newX, y: newY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging]);

  // Handle mini chat input
  const handleSend = () => {
    if (inputText.trim() && onSendMessage) {
      onSendMessage(inputText);
      setInputText('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    if (e.key === 'Escape') {
      onToggle?.();
    }
  };

  return (
    <div
      className={`avatar-container ${isExpanded ? 'expanded' : ''} ${isDragging ? 'dragging' : ''}`}
      style={{
        right: `${position.x}px`,
        bottom: `${position.y}px`,
      }}
    >
      {/* Main Avatar Face */}
      <div
        className={`avatar-face ${moodConfig.animation}`}
        onClick={() => !isDragging && onToggle?.()}
        onMouseDown={handleMouseDown}
        style={{
          '--mood-color': personaColor || moodConfig.color,
          borderColor: isConnected ? (personaColor || moodConfig.color) : '#ef4444',
        }}
      >
        <span className="avatar-emoji">{moodConfig.emoji}</span>

        {/* Connection indicator */}
        <div className={`connection-dot ${isConnected ? 'connected' : ''}`} />

        {/* Persona badge */}
        {persona && (
          <div
            className="persona-badge"
            style={{ backgroundColor: PERSONA_COLORS[persona] }}
          >
            {persona.charAt(0).toUpperCase()}
          </div>
        )}
      </div>

      {/* Status label */}
      <div className="avatar-label">
        {t(moodConfig.labelKey)}
      </div>

      {/* Expanded Mini Chat */}
      {isExpanded && (
        <div className="avatar-chat">
          <div className="chat-header">
            <span className="chat-title">{t('chat.antonioEvo')}</span>
            <span className="chat-neurons">{t('avatar.neurons', { count: neuronCount })}</span>
            <button className="chat-close" onClick={onToggle}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="chat-input-area">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isConnected ? t('avatar.askSomething') : t('avatar.notConnected')}
              disabled={!isConnected || isThinking}
            />
            <button
              className="chat-send"
              onClick={handleSend}
              disabled={!inputText.trim() || !isConnected || isThinking}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
              </svg>
            </button>
          </div>

          <div className="chat-hint">
            {t('avatar.pressEscToClose')}
          </div>
        </div>
      )}
    </div>
  );
}

export default Avatar;
