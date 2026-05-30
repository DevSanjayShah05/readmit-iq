#!/usr/bin/env bash
# Setup script for ReadmitIQ.
# Brings up the full local environment: Postgres, dependencies, JDBC driver,
# database schema, sample data, and trained model. Idempotent — safe to re-run.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> ReadmitIQ setup"
echo "    Project root: $PROJECT_ROOT"
echo

# 1. Check prerequisites
echo "==> Checking prerequisites"
command -v python >/dev/null || { echo "ERROR: python not found"; exit 1; }
command -v docker >/dev/null || { echo "ERROR: docker not found"; exit 1; }
command -v java >/dev/null || { echo "ERROR: java not found (install OpenJDK 17)"; exit 1; }
if [[ -z "${JAVA_HOME:-}" ]]; then
    echo "ERROR: JAVA_HOME not set"; exit 1
fi
PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PY_VERSION" != "3.11" ]]; then
    echo "WARNING: Python $PY_VERSION detected; project tested on 3.11"
fi
echo "    OK"
echo

# 2. Virtual environment
if [[ ! -d ".venv" ]]; then
    echo "==> Creating virtual environment"
    python -m venv .venv
fi
echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Install dependencies
echo "==> Installing Python dependencies"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e ".[dev]"

# 4. Postgres JDBC driver (required by gold ingestion)
JDBC_JAR=".venv/jars/postgresql-42.7.3.jar"
if [[ ! -f "$JDBC_JAR" ]]; then
    echo "==> Downloading Postgres JDBC driver"
    mkdir -p .venv/jars
    curl -sL -o "$JDBC_JAR" \
        https://jdbc.postgresql.org/download/postgresql-42.7.3.jar
fi
echo "    JDBC driver: $JDBC_JAR"

# 5. Start Postgres
echo "==> Starting Postgres (Docker Compose)"
docker compose -f infra/docker/docker-compose.yml up -d
echo "    Waiting for Postgres to be healthy..."
for i in {1..30}; do
    if docker compose -f infra/docker/docker-compose.yml ps postgres \
       | grep -q "(healthy)"; then
        echo "    Postgres ready"
        break
    fi
    sleep 1
done

# 6. Run database migrations
echo "==> Running Alembic migrations"
.venv/bin/alembic upgrade head

# 7. Seed sample data + train model
echo "==> Seeding sample patients (n=2000)"
.venv/bin/python -m readmit_iq.scripts.seed_data --count 2000 --truncate

echo "==> Training model"
.venv/bin/python -m readmit_iq.ml.train --output models/readmit_rf.joblib

echo
echo "==> Setup complete."
echo
echo "Next steps:"
echo "  Run tests:        pytest -v"
echo "  Start API server: uvicorn readmit_iq.api.app:app --reload --port 8000"
echo "  API docs:         http://localhost:8000/docs"
