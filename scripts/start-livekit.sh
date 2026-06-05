#!/bin/bash
# Start LiveKit server in dev mode.
# Dev mode uses API key "devkey" and secret "secret".
# Install: brew install livekit  (or download from https://github.com/livekit/livekit/releases)
set -e
exec livekit-server --dev --bind 0.0.0.0
