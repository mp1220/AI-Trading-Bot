#!/bin/bash
#Make it executable: chmod +x setup_env.sh
#Execute it: ./setup_env.sh
#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "──────────────────────────────────────────────"
echo "🔧 TraderV1 Environment Setup"
echo "──────────────────────────────────────────────"

# --- STEP 1: Navigate to script directory (safety) ---
cd "$(dirname "$0")"

# --- STEP 2: Create virtual environment if missing ---
if [ ! -d ".venv" ]; then
  echo "📦 Creating new virtual environment (.venv)..."
  python3 -m venv .venv
else
  echo "✅ Existing virtual environment found."
fi

# --- STEP 3: Activate environment ---
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# --- STEP 4: Upgrade pip + build tools ---
echo "⬆️  Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# --- STEP 5: Install project dependencies ---
echo "📥 Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# --- STEP 6: Confirmation message ---
echo "──────────────────────────────────────────────"
echo "✅ TraderV1 environment setup complete!"
echo "To start the orchestrator, run:"
echo ""
echo "    source .venv/bin/activate"
echo "    python -m app.orchestrator"
echo ""
echo "──────────────────────────────────────────────"
