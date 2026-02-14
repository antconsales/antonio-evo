import React, { useState, useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import VoiceButton from './VoiceButton';
import ToolActionIndicator from './ToolActionIndicator';
import { Paperclip, Send, X, Image, FileText, File } from 'lucide-react';
import { useTranslation } from '../i18n';

function ChatArea({ messages, onSendMessage, isLoading, isConnected, activeToolActions = [], streamingText = '' }) {
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const { t } = useTranslation();

  // Auto-scroll to bottom when new messages arrive or streaming updates
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((!input.trim() && attachments.length === 0) || isLoading || !isConnected) return;
    onSendMessage(input, false, attachments);
    setInput('');
    setAttachments([]);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleVoiceResult = (text) => {
    if (text) {
      onSendMessage(text, true, attachments);
      setAttachments([]);
    }
  };

  const handleFileSelect = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    const newAttachments = await Promise.all(
      files.map(async (file) => {
        const base64 = await fileToBase64(file);
        return {
          id: Date.now() + Math.random(),
          name: file.name,
          type: file.type,
          size: file.size,
          data: base64,
          preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : null,
        };
      })
    );

    setAttachments((prev) => [...prev, ...newAttachments]);
    e.target.value = ''; // Reset input
  };

  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });
  };

  const removeAttachment = (id) => {
    setAttachments((prev) => {
      const att = prev.find((a) => a.id === id);
      if (att?.preview) {
        URL.revokeObjectURL(att.preview);
      }
      return prev.filter((a) => a.id !== id);
    });
  };

  const getFileIcon = (type) => {
    if (type.startsWith('image/')) return <Image size={16} />;
    if (type.includes('pdf') || type.includes('document')) return <FileText size={16} />;
    return <File size={16} />;
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    const newAttachments = await Promise.all(
      files.map(async (file) => {
        const base64 = await fileToBase64(file);
        return {
          id: Date.now() + Math.random(),
          name: file.name,
          type: file.type,
          size: file.size,
          data: base64,
          preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : null,
        };
      })
    );

    setAttachments((prev) => [...prev, ...newAttachments]);
  };

  return (
    <div className="chat-area" onDragOver={handleDragOver} onDrop={handleDrop}>
      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-logo">A</div>
            <h1>{t('chat.antonio')}</h1>
            <p>{t('chat.yourLocalAssistant')}</p>
            <div className="welcome-suggestions">
              <button onClick={() => onSendMessage(t('chat.suggestionHelp'))}>
                {t('chat.suggestionHelp')}
              </button>
              <button onClick={() => onSendMessage(t('chat.suggestionAbout'))}>
                {t('chat.suggestionAbout')}
              </button>
              <button onClick={() => onSendMessage(t('chat.suggestionTime'))}>
                {t('chat.suggestionTime')}
              </button>
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isLoading && (
              <div className="message assistant loading">
                <div className="message-avatar">
                  <span className="assistant-avatar">A</span>
                </div>
                <div className="message-body">
                  {activeToolActions.length > 0 && (
                    <ToolActionIndicator actions={activeToolActions} />
                  )}
                  {streamingText ? (
                    <div className="message-content streaming-content">
                      {streamingText.split('\n').map((line, i) => (
                        <p key={i}>{line}</p>
                      ))}
                      <span className="streaming-cursor">|</span>
                    </div>
                  ) : (
                    !activeToolActions.length && (
                      <div className="message-content">
                        <div className="typing-indicator">
                          <span></span>
                          <span></span>
                          <span></span>
                        </div>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input Area */}
      <div className="input-area">
        {/* Attachment Preview */}
        {attachments.length > 0 && (
          <div className="attachments-preview">
            {attachments.map((att) => (
              <div key={att.id} className="attachment-preview-item">
                {att.type?.startsWith('image/') ? (
                  <img
                    src={att.data || att.preview}
                    alt={att.name}
                    className="attachment-thumbnail"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.replaceWith(Object.assign(document.createElement('div'), {
                        className: 'attachment-icon',
                        innerHTML: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>',
                      }));
                    }}
                  />
                ) : (
                  <div className="attachment-icon">{getFileIcon(att.type)}</div>
                )}
                <div className="attachment-info">
                  <span className="attachment-name">{att.name}</span>
                  <span className="attachment-size">{formatFileSize(att.size)}</span>
                </div>
                <button
                  className="attachment-remove"
                  onClick={() => removeAttachment(att.id)}
                  title={t('chat.removeAttachment')}
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="input-form">
          <div className="input-container">
            {/* Hidden file input */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              multiple
              style={{ display: 'none' }}
              accept="image/*,.pdf,.txt,.md,.json,.csv,.doc,.docx,.xls,.xlsx,.py,.js,.ts,.html,.css"
            />

            {/* Attachment button */}
            <button
              type="button"
              className="attach-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={!isConnected || isLoading}
              title={t('chat.attachFiles')}
            >
              <Paperclip size={20} />
            </button>

            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isConnected ? t('chat.typeMessage') : t('chat.connectingToServer')}
              disabled={!isConnected || isLoading}
              rows={1}
            />
            <div className="input-actions">
              <VoiceButton onResult={handleVoiceResult} disabled={!isConnected || isLoading} />
              <button
                type="submit"
                className="send-btn"
                disabled={(!input.trim() && attachments.length === 0) || !isConnected || isLoading}
              >
                <Send size={20} />
              </button>
            </div>
          </div>
        </form>
        <p className="input-hint">
          {t('chat.hintSend')} <kbd>{t('chat.hintEnter')}</kbd> {t('chat.hintToSend')} <kbd>{t('chat.hintShiftEnter')}</kbd> {t('chat.hintNewLine')}
          {attachments.length > 0 && ` • ${t('chat.filesAttached', { count: attachments.length })}`}
        </p>
      </div>
    </div>
  );
}

export default ChatArea;
