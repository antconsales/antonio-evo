/**
 * API helper that works both in Electron and browser
 */

const API_BASE_URL = 'http://localhost:8420';
// TTS: Piper (server-side, high quality) with Web Speech API fallback

// Check if we're running in Electron
const isElectron = () => {
  return typeof window !== 'undefined' && window.antonio !== undefined;
};

// Browser-based API calls
const browserApi = {
  async health() {
    console.log('[API] health() called, fetching:', `${API_BASE_URL}/api/health`);
    try {
      const response = await fetch(`${API_BASE_URL}/api/health`);
      console.log('[API] health() response status:', response.status);
      const data = await response.json();
      console.log('[API] health() data:', data);
      return { success: true, data };
    } catch (error) {
      console.error('[API] health() error:', error);
      return { success: false, error: error.message };
    }
  },

  async ask(text, options = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          session_id: options.sessionId || null,
          think: options.think || false,
        }),
      });
      const data = await response.json();
      return { success: data.success !== false, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  async listen(audioBase64, options = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/listen`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio: audioBase64,
          format: options.format || 'wav',
          process: options.process || false,
        }),
      });
      const data = await response.json();
      return { success: data.success !== false, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  async speak(text, options = {}) {
    // Try Piper TTS (server-side, high quality) first, fallback to Web Speech API
    try {
      const piperResult = await this._speakPiper(text);
      if (piperResult.success) {
        return piperResult;
      }
      console.log('[TTS] Piper unavailable, falling back to Web Speech API');
    } catch (e) {
      console.log('[TTS] Piper error, falling back:', e.message);
    }

    return this._speakWebSpeech(text, options);
  },

  async _speakPiper(text) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();

      if (data.success && data.audio) {
        const audioBytes = Uint8Array.from(atob(data.audio), c => c.charCodeAt(0));
        const blob = new Blob([audioBytes], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);

        return new Promise((resolve) => {
          const audio = new Audio(url);
          audio.onended = () => {
            URL.revokeObjectURL(url);
            resolve({ success: true, source: 'piper' });
          };
          audio.onerror = () => {
            URL.revokeObjectURL(url);
            resolve({ success: false, error: 'Audio playback failed' });
          };
          audio.play().catch(() => resolve({ success: false, error: 'Playback blocked' }));
        });
      }

      return { success: false, error: data.error || 'Piper TTS failed' };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  async _speakWebSpeech(text, options = {}) {
    try {
      if (!('speechSynthesis' in window)) {
        return { success: false, error: 'TTS not supported' };
      }

      window.speechSynthesis.cancel();

      const getVoices = () => new Promise((resolve) => {
        let voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) { resolve(voices); return; }
        window.speechSynthesis.onvoiceschanged = () => resolve(window.speechSynthesis.getVoices());
        setTimeout(() => resolve([]), 1000);
      });

      const voices = await getVoices();
      const utterance = new SpeechSynthesisUtterance(text);

      const italianVoice = voices.find(v => v.lang.startsWith('it'))
        || voices.find(v => v.name.toLowerCase().includes('italian'))
        || voices[0];

      if (italianVoice) utterance.voice = italianVoice;
      utterance.rate = options.rate || 1.0;
      utterance.pitch = options.pitch || 1.0;
      utterance.volume = options.volume || 1.0;

      return new Promise((resolve) => {
        utterance.onend = () => resolve({ success: true, source: 'web-speech' });
        utterance.onerror = (event) => resolve({ success: false, error: event.error || 'Speech failed' });
        window.speechSynthesis.speak(utterance);
      });
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  async ttsHealth() {
    // Web Speech API is always "available" if supported
    const supported = 'speechSynthesis' in window;
    return {
      success: supported,
      data: {
        status: supported ? 'ok' : 'not_supported',
        engine: 'Web Speech API'
      }
    };
  },
};

// Export the appropriate API
const usingElectron = isElectron();
console.log('[API] isElectron:', usingElectron);
export const api = usingElectron ? window.antonio : browserApi;

// Window controls (only work in Electron)
export const windowControls = {
  minimize: () => isElectron() && window.antonio.minimize(),
  maximize: () => isElectron() && window.antonio.maximize(),
  close: () => isElectron() && window.antonio.close(),
  isElectron: isElectron(),
};
