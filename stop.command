#!/bin/bash
# Double-click this file to stop Video Translator (Mac)
cd "$(dirname "$0")"
docker compose down
echo "✓ Video Translator stopped"
