#!/bin/bash
# Start Video Translator with AMD GPU (Linux)
cd "$(dirname "$0")"
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d
sleep 2
xdg-open http://localhost:7860 2>/dev/null || echo "Open http://localhost:7860"
echo "✓ Video Translator running (AMD GPU) at http://localhost:7860"
