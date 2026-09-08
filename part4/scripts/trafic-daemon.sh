#!/bin/bash
# Perpetual synthetic traffic against the fraud-detector FastAPI app
# so Prometheus has continuous samples and the Grafana dashboard
# renders a non-flat line as soon as the reader builds it. Launched
# by startup-script.sh in the background; not intended to be invoked
# by the reader.
set -u

while true; do
  curl -s -o /dev/null -X POST http://localhost:8085/predict \
    -H 'Content-Type: application/json' \
    -d '{"features": [100.5, 12, 3]}' 2>/dev/null
  curl -s -o /dev/null http://localhost:8085/health 2>/dev/null
  sleep 0.5
done
