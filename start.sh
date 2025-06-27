#!/bin/bash

# Start Rasa server with API enabled and CORS allowed
rasa run \
  --enable-api \
  --cors "*" \
  --port ${PORT:-10000}
