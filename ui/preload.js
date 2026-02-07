/**
 * Antonio Desktop - Preload Script
 *
 * Exposes secure APIs to the renderer process via contextBridge.
 * This is the only way the renderer can communicate with the main process.
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to the renderer
contextBridge.exposeInMainWorld('antonio', {
  // Window controls
  minimize: () => ipcRenderer.send('window-minimize'),
  maximize: () => ipcRenderer.send('window-maximize'),
  close: () => ipcRenderer.send('window-close'),

  // API methods
  health: () => ipcRenderer.invoke('api-health'),

  ask: (question, options) => ipcRenderer.invoke('api-ask', question, options),

  listen: (audioBase64, options) => ipcRenderer.invoke('api-listen', audioBase64, options),

  // App info
  getAppInfo: () => ipcRenderer.invoke('get-app-info'),
});

// Expose platform info
contextBridge.exposeInMainWorld('platform', {
  isWindows: process.platform === 'win32',
  isMac: process.platform === 'darwin',
  isLinux: process.platform === 'linux',
});
