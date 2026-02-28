#!/bin/bash
# Run this to start Video Translator (Linux)
cd "$(dirname "$0")"
docker compose up -d
sleep 2
xdg-open http://localhost:7860 2>/dev/null || echo "Open http://localhost:7860 in your browser"
echo "✓ Video Translator running at http://localhost:7860"
echo "  To stop: ./stop.sh"
