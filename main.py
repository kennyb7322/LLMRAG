#!/usr/bin/env python3
"""
LLMRAG — Main Entry Point
============================
Automated LLM & RAG Deployment for AI/AGI/GI Genetic Frameworks

Quick Start:
    python main.py setup        # First-time setup
    python main.py ingest       # Load and index data
    python main.py query "..."  # Ask a question
    python main.py serve        # Start API server
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.tools.cli import main

if __name__ == "__main__":
    main()
