#!/usr/bin/env bash
# Reproduce the same environment + HR data (1,470 rows) on a fresh clone.
# Data is loaded from data/init.sql (seed_datasets; runserver also auto-seeds).
# Usage: bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

# 1. Virtual environment (shared global venv, matches project convention).
python3 -m venv ~/.venv
# shellcheck disable=SC1090
source ~/.venv/bin/activate

# 2. Dependencies.
python -m pip install -U pip
pip install -r requirements.txt

# 3. Build database tables (seeding is skipped during migrate).
python manage.py migrate

# 4. Load HR data from data/init.sql (no-op if already seeded).
python manage.py seed_datasets

# 5. Verify row count.
python manage.py shell -c "from core.models import Employee; print('Employee rows:', Employee.objects.count())"

# 6. Admin login (change password before any non-local use).
DJANGO_SUPERUSER_PASSWORD=admin123 python manage.py createsuperuser \
    --username admin --email admin@example.com --noinput || true

echo
echo "Done. Run:  source ~/.venv/bin/activate && python manage.py runserver"
echo "Admin: http://localhost:8000/admin/   user=admin  pass=admin123"
