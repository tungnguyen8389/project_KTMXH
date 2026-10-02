#!/usr/bin/env bash
# Reproduce the same environment + HR data (1,470 rows) on a fresh clone.
# Data is loaded from data/init.sql (seed_datasets; runserver also auto-seeds).
# Usage: bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

# 1. Virtual environment.
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -s)" == CYGWIN* ]]; then
    python.exe -m venv .venv
    PYTHON=.venv/Scripts/python.exe
    # shellcheck disable=SC1090
    source .venv/Scripts/activate
else
    PYTHON=python3
    python3 -m venv ~/.venv
    # shellcheck disable=SC1090
    source ~/.venv/bin/activate
fi

# 2. Dependencies.
"$PYTHON" -m pip install -U pip
"$PYTHON" -m pip install -r requirements.txt

# 3. Build database tables (seeding is skipped during migrate).
"$PYTHON" manage.py migrate

# 4. Load HR data from data/init.sql (no-op if already seeded).
"$PYTHON" manage.py seed_datasets

# 5. Verify row count.
"$PYTHON" manage.py shell -c "from core.models import Employee; print('Employee rows:', Employee.objects.count())"

# 6. Admin login (change password before any non-local use).
DJANGO_SUPERUSER_PASSWORD=admin123 "$PYTHON" manage.py createsuperuser \
    --username admin --email admin@example.com --noinput || true

echo
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -s)" == CYGWIN* ]]; then
    echo "Done. Run:  source .venv/Scripts/activate && python manage.py runserver"
else
    echo "Done. Run:  source ~/.venv/bin/activate && python manage.py runserver"
fi
echo "Admin: http://localhost:8000/admin/   user=admin  pass=admin123"
