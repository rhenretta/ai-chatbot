#!/bin/bash

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Starting Redis..."
    brew services start redis
    sleep 2
fi

# Verify Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Error: Redis failed to start"
    exit 1
fi

echo "Redis is running"

# Kill any existing uvicorn processes
pkill -f uvicorn

# Start the FastAPI server
echo "Starting AI Memory Chatbot..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 