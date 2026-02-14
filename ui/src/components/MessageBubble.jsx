import React, { useState } from 'react';
import { Image, FileText, File, Download, ExternalLink, X, Search, Code, FolderOpen, Wrench } from 'lucide-react';
import { useTranslation } from '../i18n';

const TOOL_ICON_MAP = {
  web_search: Search,
  read_file: FileText,
  write_file: FileText,
  list_directory: FolderOpen,
  execute_code: Code,
  analyze_image: Image,
};

function MessageBubble({ message }) {
  const { role, content, timestamp, success, meta, attachments } = message;
  const isUser = role === 'user';
  const [lightboxImage, setLightboxImage] = useState(null);
  const { t } = useTranslation();

  const formatTime = (ts) => {
    const date = new Date(ts);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const getFileIcon = (type) => {
    if (type?.startsWith('image/')) return <Image size={16} />;
    if (type?.includes('pdf') || type?.includes('document')) return <FileText size={16} />;
    return <File size={16} />;
  };

  const downloadAttachment = (att) => {
    const link = document.createElement('a');
    link.href = att.data;
    link.download = att.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const openImageInLightbox = (att) => {
    setLightboxImage(att);
  };

  const renderAttachments = () => {
    if (!attachments || attachments.length === 0) return null;

    return (
      <div className="message-attachments">
        {attachments.map((att, index) => (
          <div key={att.id || index} className="message-attachment">
            {att.type?.startsWith('image/') ? (
              <div className="attachment-image-wrapper" onClick={() => openImageInLightbox(att)}>
                <img
                  src={att.data || att.preview}
                  alt={att.name}
                  className="attachment-image"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
                <div className="attachment-image-overlay">
                  <ExternalLink size={20} />
                </div>
              </div>
            ) : (
              <div className="attachment-file">
                <div className="attachment-file-icon">{getFileIcon(att.type)}</div>
                <div className="attachment-file-info">
                  <span className="attachment-file-name">{att.name}</span>
                  <span className="attachment-file-size">{formatFileSize(att.size)}</span>
                </div>
                <button
                  className="attachment-download-btn"
                  onClick={() => downloadAttachment(att)}
                  title={t('message.download')}
                >
                  <Download size={16} />
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <>
      <div className={`message ${role} ${success === false ? 'error' : ''}`}>
        <div className="message-avatar">
          {isUser ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
            </svg>
          ) : (
            <span className="assistant-avatar">A</span>
          )}
        </div>
        <div className="message-body">
          <div className="message-header">
            <span className="message-role">{isUser ? t('message.you') : t('message.antonio')}</span>
            <span className="message-time">{formatTime(timestamp)}</span>
          </div>

          {/* Attachments */}
          {renderAttachments()}

          {/* Text content */}
          {content && (
            <div className="message-content">
              {content.split('\n').map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>
          )}

          {meta && !isUser && (
            <div className="message-meta">
              <span className="meta-item">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 6v6l4 2" />
                </svg>
                {meta.elapsed_ms}ms
              </span>
              {meta.handler && (
                <span className="meta-item">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <path d="M22 4L12 14.01l-3-3" />
                  </svg>
                  {meta.handler}
                </span>
              )}
              {meta.persona && (
                <span className={`meta-item persona-${meta.persona}`}>
                  {meta.persona.toUpperCase()}
                </span>
              )}
              {meta.neuron_stored && (
                <span className="meta-item learned">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z" />
                    <path d="M12 6v6l4 4" />
                  </svg>
                  {t('message.learned')}
                </span>
              )}
            </div>
          )}

          {/* Tools used (v5.0) */}
          {meta?.tools_used && meta.tools_used.length > 0 && !isUser && (
            <details className="tools-used-details">
              <summary className="tools-used-summary">
                <Wrench size={12} />
                {t('tools.toolsUsed') || 'Tools used'} ({meta.tools_used.length})
              </summary>
              <div className="tools-used-list">
                {meta.tools_used.map((tool, idx) => {
                  const ToolIcon = TOOL_ICON_MAP[tool.tool] || Wrench;
                  return (
                    <div key={idx} className={`tools-used-item ${tool.success ? '' : 'tools-used-failed'}`}>
                      <ToolIcon size={12} />
                      <span className="tools-used-name">{tool.tool}</span>
                      {tool.elapsed_ms > 0 && (
                        <span className="tools-used-time">{tool.elapsed_ms}ms</span>
                      )}
                      <span className={`tools-used-status ${tool.success ? 'success' : 'failed'}`}>
                        {tool.success ? '\u2713' : '\u2717'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </details>
          )}
        </div>
      </div>

      {/* Image Lightbox */}
      {lightboxImage && (
        <div className="lightbox-overlay" onClick={() => setLightboxImage(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setLightboxImage(null)}>
              <X size={24} />
            </button>
            <img
              src={lightboxImage.data || lightboxImage.preview}
              alt={lightboxImage.name}
              className="lightbox-image"
              onError={(e) => { e.target.style.display = 'none'; }}
            />
            <div className="lightbox-footer">
              <span>{lightboxImage.name}</span>
              <button onClick={() => downloadAttachment(lightboxImage)}>
                <Download size={16} /> {t('message.download')}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default MessageBubble;
