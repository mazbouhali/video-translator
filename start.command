#!/bin/bash
# Double-click this file to start Video Translator (Mac)
cd "$(dirname "$0")"
docker compose up -d
sleep 2
open http://localhost:7860
echo "✓ Video Translator running at http://localhost:7860"
echo "  Close this window anytime — it keeps running in background"
echo "  To stop: double-click stop.command"
