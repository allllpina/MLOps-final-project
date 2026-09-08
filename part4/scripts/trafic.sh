#!/bin/bash
# Generate synthetic traffic against the fraud-detector FastAPI app
# for 60 seconds so Prometheus has samples to scrape and the Grafana
# dashboard panel has data to render.
set -u

end=$(( $(date +%s) + 60 ))
count=0
while [ "$(date +%s)" -lt "$end" ]; do
  curl -s -o /dev/null -X POST http://localhost:8085/predict \
    -H 'Content-Type: application/json' \
    -d '{"features": [100.5, 12, 3]}'
  curl -s -o /dev/null http://localhost:8085/health
  count=$(( count + 2 ))
  sleep 0.1
done
echo "traffic.sh done -- $count requests sent"
