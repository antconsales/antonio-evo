import React, { useState, useEffect } from 'react';
import { useTranslation } from '../i18n';

const GALLERY_KEY = 'antonio_image_gallery';

function ImageGenerator({ isConnected }) {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [error, setError] = useState(null);
  const [gallery, setGallery] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [settings, setSettings] = useState({
    width: 512,
    height: 512,
    steps: 8,
  });
  const { t } = useTranslation();

  // Load gallery from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(GALLERY_KEY);
      if (saved) {
        setGallery(JSON.parse(saved));
      }
    } catch (e) {
      console.error('Failed to load gallery:', e);
    }
  }, []);

  // Save gallery to localStorage
  const saveGallery = (newGallery) => {
    try {
      localStorage.setItem(GALLERY_KEY, JSON.stringify(newGallery));
    } catch (e) {
      console.error('Failed to save gallery:', e);
    }
  };

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isGenerating || !isConnected) return;

    setIsGenerating(true);
    setError(null);
    setGeneratedImage(null);

    try {
      const response = await fetch('http://localhost:8420/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          width: settings.width,
          height: settings.height,
          steps: settings.steps,
        }),
      });

      const data = await response.json();

      if (data.success) {
        const newImage = {
          id: Date.now(),
          url: `http://localhost:8420${data.image_url}`,
          path: data.image_path,
          prompt: data.prompt,
          time: data.generation_time,
          width: settings.width,
          height: settings.height,
          steps: settings.steps,
          createdAt: new Date().toISOString(),
        };

        setGeneratedImage(newImage);

        // Add to gallery
        const newGallery = [newImage, ...gallery].slice(0, 50); // Keep last 50 images
        setGallery(newGallery);
        saveGallery(newGallery);
      } else {
        setError(data.error || t('image.generationFailed'));
      }
    } catch (err) {
      setError(err.message || t('image.connectionError'));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDeleteImage = (id) => {
    const newGallery = gallery.filter(img => img.id !== id);
    setGallery(newGallery);
    saveGallery(newGallery);
    if (selectedImage?.id === id) {
      setSelectedImage(null);
    }
  };

  const handleClearGallery = () => {
    if (window.confirm(t('image.clearGalleryConfirm'))) {
      setGallery([]);
      saveGallery([]);
      setSelectedImage(null);
    }
  };

  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="image-generator">
      <h2>{t('image.title')}</h2>
      <p className="subtitle">{t('image.subtitle')}</p>

      <div className="generator-layout">
        {/* Left: Form and current result */}
        <div className="generator-main">
          <form onSubmit={handleGenerate} className="generator-form">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={t('image.promptPlaceholder')}
              disabled={isGenerating || !isConnected}
              rows={3}
            />

            <div className="generator-settings">
              <label>
                {t('image.size')}
                <select
                  value={`${settings.width}x${settings.height}`}
                  onChange={(e) => {
                    const [w, h] = e.target.value.split('x').map(Number);
                    setSettings((s) => ({ ...s, width: w, height: h }));
                  }}
                  disabled={isGenerating}
                >
                  <option value="256x256">{t('image.sizeFast')}</option>
                  <option value="512x512">{t('image.sizeDefault')}</option>
                  <option value="768x768">{t('image.sizeLarge')}</option>
                </select>
              </label>

              <label>
                {t('image.steps')}
                <select
                  value={settings.steps}
                  onChange={(e) => setSettings((s) => ({ ...s, steps: Number(e.target.value) }))}
                  disabled={isGenerating}
                >
                  <option value="4">{t('image.stepsFast')}</option>
                  <option value="8">{t('image.stepsDefault')}</option>
                  <option value="12">{t('image.stepsQuality')}</option>
                </select>
              </label>
            </div>

            <button
              type="submit"
              className="generate-btn"
              disabled={!prompt.trim() || isGenerating || !isConnected}
            >
              {isGenerating ? (
                <>
                  <span className="spinner"></span>
                  {t('image.generating')}
                </>
              ) : (
                t('image.generateImage')
              )}
            </button>
          </form>

          {error && (
            <div className="generator-error">
              {error}
            </div>
          )}

          {/* Current generated image */}
          {generatedImage && !selectedImage && (
            <div className="generated-result">
              <h3>{t('image.generatedImage')}</h3>
              <div className="image-preview-large">
                <img
                  src={generatedImage.url}
                  alt={generatedImage.prompt}
                  className="generated-image-main"
                  onClick={() => setSelectedImage(generatedImage)}
                />
              </div>
              <div className="image-info">
                <p><strong>{t('image.prompt')}</strong> {generatedImage.prompt}</p>
                <p><strong>{t('image.sizeLabel')}</strong> {generatedImage.width}x{generatedImage.height} | <strong>{t('image.stepsLabel')}</strong> {generatedImage.steps} | <strong>{t('image.timeLabel')}</strong> {generatedImage.time?.toFixed(1)}s</p>
                <div className="image-actions">
                  <a
                    href={generatedImage.url}
                    download={`zimage_${generatedImage.id}.png`}
                    className="action-btn primary"
                  >
                    {t('image.download')}
                  </a>
                  <button
                    className="action-btn"
                    onClick={() => setPrompt(generatedImage.prompt)}
                  >
                    {t('image.reusePrompt')}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Selected image from gallery */}
          {selectedImage && (
            <div className="generated-result">
              <div className="result-header">
                <h3>{t('image.imageDetails')}</h3>
                <button className="close-preview" onClick={() => setSelectedImage(null)}>
                  {t('image.closePlaceholder')}
                </button>
              </div>
              <div className="image-preview-large">
                <img
                  src={selectedImage.url}
                  alt={selectedImage.prompt}
                  className="generated-image-main"
                />
              </div>
              <div className="image-info">
                <p><strong>{t('image.prompt')}</strong> {selectedImage.prompt}</p>
                <p><strong>{t('image.sizeLabel')}</strong> {selectedImage.width}x{selectedImage.height} | <strong>{t('image.stepsLabel')}</strong> {selectedImage.steps}</p>
                <p><strong>{t('image.createdLabel')}</strong> {formatDate(selectedImage.createdAt)}</p>
                <div className="image-actions">
                  <a
                    href={selectedImage.url}
                    download={`zimage_${selectedImage.id}.png`}
                    className="action-btn primary"
                  >
                    {t('image.download')}
                  </a>
                  <button
                    className="action-btn"
                    onClick={() => setPrompt(selectedImage.prompt)}
                  >
                    {t('image.reusePrompt')}
                  </button>
                  <button
                    className="action-btn danger"
                    onClick={() => handleDeleteImage(selectedImage.id)}
                  >
                    {t('image.delete')}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right: Gallery */}
        <div className="image-gallery">
          <div className="gallery-header">
            <h3>{t('image.gallery', { count: gallery.length })}</h3>
            {gallery.length > 0 && (
              <button className="clear-gallery" onClick={handleClearGallery}>
                {t('image.clearAll')}
              </button>
            )}
          </div>

          {gallery.length === 0 ? (
            <div className="gallery-empty">
              <p>{t('image.noImages')}</p>
              <p className="hint">{t('image.generatedImagesHint')}</p>
            </div>
          ) : (
            <div className="gallery-grid">
              {gallery.map((img) => (
                <div
                  key={img.id}
                  className={`gallery-item ${selectedImage?.id === img.id ? 'selected' : ''}`}
                  onClick={() => setSelectedImage(img)}
                >
                  <img src={img.url} alt={img.prompt} />
                  <div className="gallery-item-overlay">
                    <span className="gallery-item-time">{img.time?.toFixed(1)}s</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {isGenerating && (
        <div className="generating-overlay">
          <div className="generating-spinner"></div>
          <p>{t('image.generatingOnCpu')}</p>
          <p className="generating-hint">{t('image.generatingHint')}</p>
        </div>
      )}
    </div>
  );
}

export default ImageGenerator;
