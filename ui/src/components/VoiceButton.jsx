import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useTranslation } from '../i18n';

const API_BASE_URL = 'http://localhost:8420';

// BCP-47 language map for Web Speech API (v7.0)
const LANG_MAP = { en: 'en-US', it: 'it-IT', fr: 'fr-FR', es: 'es-ES' };

function VoiceButton({ onResult, disabled, language = 'en' }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [useServerTranscription, setUseServerTranscription] = useState(false);
  const recognitionRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const { t } = useTranslation();

  const handleResult = useCallback((text) => {
    if (onResult && text) onResult(text);
  }, [onResult]);

  // Server-side transcription fallback
  const transcribeOnServer = async (audioBlob) => {
    try {
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);

      return new Promise((resolve, reject) => {
        reader.onloadend = async () => {
          const base64 = reader.result.split(',')[1];

          try {
            const response = await fetch(`${API_BASE_URL}/api/listen`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ audio: base64, format: 'webm' }),
            });

            const data = await response.json();
            if (data.success && data.text) {
              resolve(data.text);
            } else {
              reject(new Error(data.error || 'Transcription failed'));
            }
          } catch (e) {
            reject(e);
          }
        };
        reader.onerror = reject;
      });
    } catch (error) {
      console.error('[Voice] Server transcription failed:', error);
      throw error;
    }
  };

  // Start MediaRecorder for server-side transcription
  const startMediaRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm',
      });

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        setIsProcessing(true);
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        stream.getTracks().forEach((track) => track.stop());

        try {
          const text = await transcribeOnServer(audioBlob);
          handleResult(text);
        } catch (error) {
          console.error('[Voice] Transcription error:', error);
          alert(t('voice.transcriptionFailed'));
        } finally {
          setIsProcessing(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      console.log('[Voice] MediaRecorder started (server-side mode)');
    } catch (error) {
      console.error('[Voice] Failed to start MediaRecorder:', error);
      alert(t('voice.microphoneError'));
    }
  };

  const stopMediaRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Initialize Web Speech API
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.log('[Voice] Web Speech API not available, using server-side transcription');
      setUseServerTranscription(true);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = LANG_MAP[language] || 'en-US';

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      console.log('[Voice] Web Speech transcription:', transcript);
      handleResult(transcript);
      setIsProcessing(false);
      setIsRecording(false);
    };

    recognition.onerror = (event) => {
      console.error('[Voice] Web Speech error:', event.error);
      setIsRecording(false);

      if (event.error === 'network') {
        console.log('[Voice] Network error, switching to server-side transcription');
        setUseServerTranscription(true);
        // Auto-start with MediaRecorder
        startMediaRecording();
      } else if (event.error === 'not-allowed') {
        alert(t('voice.microphoneDenied'));
        setIsProcessing(false);
      } else {
        setIsProcessing(false);
      }
    };

    recognition.onend = () => {
      if (!useServerTranscription) {
        setIsRecording(false);
        setIsProcessing(false);
      }
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch (e) {}
      }
    };
  }, [handleResult, useServerTranscription, language]);

  const startRecording = () => {
    if (useServerTranscription) {
      startMediaRecording();
    } else if (recognitionRef.current) {
      try {
        recognitionRef.current.start();
        setIsRecording(true);
        console.log('[Voice] Web Speech started');
      } catch (error) {
        console.error('[Voice] Web Speech start failed:', error);
        // Fallback to server
        setUseServerTranscription(true);
        startMediaRecording();
      }
    } else {
      startMediaRecording();
    }
  };

  const stopRecording = () => {
    if (useServerTranscription || mediaRecorderRef.current?.state === 'recording') {
      stopMediaRecording();
    } else if (recognitionRef.current && isRecording) {
      try {
        recognitionRef.current.stop();
        setIsProcessing(true);
      } catch (e) {
        setIsRecording(false);
        setIsProcessing(false);
      }
    }
  };

  const handleClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <button
      type="button"
      className={`voice-btn ${isRecording ? 'recording' : ''} ${isProcessing ? 'processing' : ''}`}
      onClick={handleClick}
      disabled={disabled || isProcessing}
      title={isRecording ? t('voice.stopRecording') : t('voice.startVoiceInput')}
    >
      {isProcessing ? (
        <svg className="spinner" width="20" height="20" viewBox="0 0 24 24">
          <circle
            cx="12"
            cy="12"
            r="10"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeDasharray="60"
            strokeLinecap="round"
          />
        </svg>
      ) : (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
          <line x1="12" y1="19" x2="12" y2="23" />
          <line x1="8" y1="23" x2="16" y2="23" />
        </svg>
      )}
      {isRecording && <span className="recording-pulse"></span>}
    </button>
  );
}

export default VoiceButton;
