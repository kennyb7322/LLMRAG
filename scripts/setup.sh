#!/bin/bash
# ============================================================================
# LLMRAG Quick Setup Script
# Run: bash scripts/setup.sh
# ============================================================================

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║     LLMRAG — Quick Setup                   ║"
echo "║     Automated LLM + RAG Deployment          ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p data/documents data/vectordb logs models/embeddings models/llm configs

# Python venv
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy env template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env from template — edit with your API keys"
fi

# Run setup wizard
echo ""
python main.py setup

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ LLMRAG is ready!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Add documents to data/documents/"
echo "  3. Run: python main.py ingest"
echo "  4. Query: python main.py query 'Your question'"
echo "  5. Serve: python main.py serve"
echo "═══════════════════════════════════════════════"
