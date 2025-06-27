#!/bin/bash

# Start Rasa custom actions server
rasa run actions --port ${PORT:-5055}
