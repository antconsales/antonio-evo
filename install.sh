#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Antonio Evo — One-liner Install Script
# Usage: curl -sSL https://raw.githubusercontent.com/antconsales/antonio-evo/main/install.sh | bash
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

echo ""
echo "==========================================="
echo "  Antonio Evo — Installer"
echo "  Local Cognitive Runtime"
echo "==========================================="
echo ""

# --- Detect OS ---
OS="$(uname -s)"
case "$OS" in
  Darwin) PLATFORM="macOS" ;;
  Linux)  PLATFORM="Linux" ;;
  *)      fail "Unsupported platform: $OS. Only macOS and Linux are supported." ;;
esac
info "Platform: $PLATFORM"

# --- Check Python 3.11+ ---
info "Checking Python..."
if command -v python3 &>/dev/null; then
  PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
  PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
  if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
    ok "Python $PY_VERSION found"
  else
    fail "Python 3.11+ required, found $PY_VERSION. Install with: brew install python@3.11 (macOS) or sudo apt install python3.11 (Linux)"
  fi
else
  fail "Python3 not found. Install with: brew install python@3.11 (macOS) or sudo apt install python3.11 (Linux)"
fi

# --- Check Node.js 18+ ---
info "Checking Node.js..."
if command -v node &>/dev/null; then
  NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
  if [ "$NODE_VERSION" -ge 18 ]; then
    ok "Node.js v$(node -v | sed 's/v//') found"
  else
    fail "Node.js 18+ required, found v$(node -v). Install with: brew install node (macOS) or see https://nodejs.org"
  fi
else
  fail "Node.js not found. Install with: brew install node (macOS) or see https://nodejs.org"
fi

# --- Check/Install Ollama ---
info "Checking Ollama..."
if command -v ollama &>/dev/null; then
  ok "Ollama found at $(which ollama)"
else
  warn "Ollama not found. Installing..."
  if [ "$PLATFORM" = "macOS" ]; then
    if command -v brew &>/dev/null; then
      brew install ollama
    else
      fail "Homebrew not found. Install Ollama manually from https://ollama.com/download"
    fi
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  ok "Ollama installed"
fi

# --- Start Ollama if not running ---
info "Ensuring Ollama is running..."
if ! ollama list &>/dev/null 2>&1; then
  info "Starting Ollama server..."
  ollama serve &>/dev/null &
  sleep 3
fi
ok "Ollama is running"

# --- Pull Mistral model ---
info "Checking Mistral model..."
if ollama list 2>/dev/null | grep -q "mistral"; then
  ok "Mistral model already available"
else
  info "Pulling Mistral model (this may take a few minutes on first run)..."
  ollama pull mistral
  ok "Mistral model downloaded"
fi

# --- Setup Python environment ---
info "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
  python3 -m venv venv
  ok "Virtual environment created"
else
  ok "Virtual environment already exists"
fi

source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
ok "Python dependencies installed"

# --- Setup .env ---
if [ ! -f ".env" ]; then
  info "Creating .env from .env.example..."
  if [ -f ".env.example" ]; then
    cp .env.example .env
  else
    cat > .env << 'ENVEOF'
LLM_SERVER=http://localhost:11434
OLLAMA_MODEL=mistral
LLM_TIMEOUT=120
ASR_SERVER=http://localhost:8803
TTS_SERVER=http://localhost:8804
WHISPER_MODEL=base
PIPER_VOICE=it_IT-riccardo-x_low
EXTERNAL_API_KEY=
ENVEOF
  fi
  ok ".env file created"
else
  ok ".env file already exists"
fi

# --- Create required directories ---
mkdir -p data logs
ok "Data and logs directories ready"

# --- Install UI dependencies ---
info "Installing UI dependencies..."
cd ui
npm install --quiet 2>/dev/null
cd ..
ok "UI dependencies installed"

echo ""
echo "==========================================="
echo "  Installation complete!"
echo "==========================================="
echo ""
echo "To start Antonio Evo:"
echo ""
echo "  Terminal 1 (Backend):"
echo "    source venv/bin/activate"
echo "    python -m src.api.websocket_server"
echo ""
echo "  Terminal 2 (UI):"
echo "    cd ui && npm run dev"
echo ""
echo "  Then open: http://localhost:5173"
echo ""
echo "==========================================="
