#!/bin/bash
# Run this to stop Video Translator (Linux)
cd "$(dirname "$0")"
docker compose down
echo "✓ Video Translator stopped"
