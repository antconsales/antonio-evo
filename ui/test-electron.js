// Test electron loading
try {
  const electron = require('electron');
  console.log('Electron module type:', typeof electron);
  console.log('Electron module:', electron);
  console.log('Keys:', Object.keys(electron));

  if (electron.app) {
    console.log('app is available');
  } else {
    console.log('app is NOT available');
    console.log('This might mean we are in renderer process or Node.js context');
  }
} catch (e) {
  console.error('Error loading electron:', e);
}
