#!/bin/sh
set -e

# Extract hostname and port from BACKEND_URL
host="${BACKEND_URL#*://}"
host="${host%%/*}"
port="${host##*:}"
host="${host%%:*}"

# Validate extracted host and port
if [ -z "$host" ] || [ -z "$port" ]; then
    echo "Error: BACKEND_URL is invalid or missing. Provided: $BACKEND_URL"
    exit 1
fi

echo "Waiting for backend at $host:$port to be ready..."
while ! nc -z "$host" "$port"; do
    sleep 1
done

echo "Backend is ready. Starting application layer..."
exec "$@"
