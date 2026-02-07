/**
 * Antonio Desktop - Electron Main Process
 */

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const http = require('http');

// Configuration
const API_BASE_URL = 'http://localhost:8420';

let mainWindow = null;

function isDev() {
  return process.env.NODE_ENV === 'development' ||
         process.defaultApp ||
         /[\\/]electron[\\/]/.test(process.execPath);
}

/**
 * Create the main application window
 */
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    frame: false,
    backgroundColor: '#1a1a2e',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    icon: path.join(__dirname, 'public', 'icon.svg'),
  });

  if (isDev()) {
    mainWindow.loadURL('http://localhost:5174');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

/**
 * Make HTTP request to Antonio API
 */
function makeApiRequest(endpoint, method, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(endpoint, API_BASE_URL);

    const options = {
      hostname: url.hostname,
      port: url.port || 8420,
      path: url.pathname,
      method: method,
      headers: { 'Content-Type': 'application/json' },
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(data) });
        } catch (e) {
          resolve({ status: res.statusCode, data: data });
        }
      });
    });

    req.on('error', reject);
    req.setTimeout(60000, () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });

    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

/**
 * Register IPC handlers
 */
function registerIpcHandlers() {
  // Window controls
  ipcMain.on('window-minimize', () => mainWindow?.minimize());
  ipcMain.on('window-maximize', () => {
    if (mainWindow?.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow?.maximize();
    }
  });
  ipcMain.on('window-close', () => mainWindow?.close());

  // API: Health check
  ipcMain.handle('api-health', async () => {
    try {
      const result = await makeApiRequest('/health', 'GET');
      return { success: true, data: result.data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  // API: Send message
  ipcMain.handle('api-ask', async (event, question, options = {}) => {
    try {
      const body = {
        question: question,
        output_mode: 'json',
        speak: options.speak || false,
        return_audio: options.returnAudio || false,
      };
      const result = await makeApiRequest('/ask', 'POST', body);
      return { success: true, data: result.data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  // API: Voice transcription
  ipcMain.handle('api-listen', async (event, audioBase64, options = {}) => {
    try {
      const body = {
        audio: audioBase64,
        format: 'wav',
        process: options.process || false,
        output_mode: 'json',
      };
      const result = await makeApiRequest('/listen', 'POST', body);
      return { success: true, data: result.data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  });

  // Get app info
  ipcMain.handle('get-app-info', () => ({
    version: app.getVersion(),
    name: app.getName(),
    isDev: isDev(),
  }));
}

// App lifecycle
app.whenReady().then(() => {
  registerIpcHandlers();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
