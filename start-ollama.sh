#!/bin/sh
set -e

# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready using Python
echo "⏳ Waiting for Ollama server to be ready..."
for i in $(seq 1 10); do
  python -c "
import urllib.request
try:
    urllib.request.urlopen('http://localhost:11434/', timeout=1)
    print('✅ Ollama server is up!')
except:
    exit(1)
" && break

  echo "⏳ Waiting ($i)..."
  sleep 1
done

# Pull your model
ollama pull gemma:2b

# Keep container running
tail -f /dev/null
