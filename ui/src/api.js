/**
 * API helper that works both in Electron and browser
 */

const API_BASE_URL = 'http://localhost:8420';
// TTS now uses Web Speech API (browser-native) instead of separate server

// Check if we're running in Electron
const isElectron = () => {
  return typeof window !== 'undefined' && window.antonio !== undefined;
};

// Browser-based API calls
const browserApi = {
  async health() {
    console.log('[API] health() called, fetching:', `${API_BASE_URL}/health`);
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      console.log('[API] health() response status:', response.status);
      const data = await response.json();
      console.log('[API] health() data:', data);
      return { success: true, data };
    } catch (error) {
      console.error('[API] health() error:', error);
      return { success: false, error: error.message };
    }
  },

  async ask(question, options = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          output_mode: 'json',
          speak: options.speak || false,
          return_audio: options.returnAudio || false,
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
      const response = await fetch(`${API_BASE_URL}/listen`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio: audioBase64,
          format: 'wav',
          process: options.process || false,
          output_mode: 'json',
        }),
      });
      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  async speak(text, options = {}) {
    // Use Web Speech API for TTS (works in all modern browsers)
    try {
      if (!('speechSynthesis' in window)) {
        console.error('[TTS] Web Speech API not supported');
        return { success: false, error: 'TTS not supported in this browser' };
      }

      // Cancel any ongoing speech
      window.speechSynthesis.cancel();

      // Helper to get voices (may need to wait for them to load)
      const getVoices = () => {
        return new Promise((resolve) => {
          let voices = window.speechSynthesis.getVoices();
          if (voices.length > 0) {
            resolve(voices);
            return;
          }
          // Voices not loaded yet, wait for them
          window.speechSynthesis.onvoiceschanged = () => {
            voices = window.speechSynthesis.getVoices();
            resolve(voices);
          };
          // Timeout fallback
          setTimeout(() => resolve([]), 1000);
        });
      };

      const voices = await getVoices();
      const utterance = new SpeechSynthesisUtterance(text);

      // Try to find an Italian voice, fallback to default
      const italianVoice = voices.find(v => v.lang.startsWith('it'))
        || voices.find(v => v.name.toLowerCase().includes('italian'))
        || voices[0];

      if (italianVoice) {
        utterance.voice = italianVoice;
        console.log('[TTS] Using voice:', italianVoice.name, italianVoice.lang);
      }

      utterance.rate = options.rate || 1.0;
      utterance.pitch = options.pitch || 1.0;
      utterance.volume = options.volume || 1.0;

      return new Promise((resolve) => {
        utterance.onend = () => {
          resolve({ success: true });
        };
        utterance.onerror = (event) => {
          console.error('[TTS] Speech error:', event);
          resolve({ success: false, error: event.error || 'Speech failed' });
        };

        window.speechSynthesis.speak(utterance);
      });
    } catch (error) {
      console.error('[TTS] Error:', error);
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
