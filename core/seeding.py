"""Seed the Employee table from data/init.sql (replaces the old CSV import).

data/init.sql is the source of record for the HR dataset. It is generated once
from the cleaned data and holds one INSERT per employee row. The app runs it
automatically on first launch (see core/apps.py) so no CSV file is needed.
"""
from pathlib import Path

from django.conf import settings
from django.db import connection

from .models import Employee

INIT_SQL_PATH = Path(settings.BASE_DIR) / "data" / "init.sql"

# The 31 data columns, in a stable order. Used to serialise Employee rows back
# into the plain record dicts every algorithm engine expects (no pk field).
HR_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction",
    "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole",
    "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


def hr_records():
    """Return the HR dataset as a list of plain dicts (algorithm input shape)."""
    return list(Employee.objects.all().values(*HR_COLUMNS))


def seed_employees(force=False):
    """Load data/init.sql into the Employee table if it is empty.

    Returns the number of rows in the table afterwards. With force=True the
    table is wiped and re-seeded. Raises if data/init.sql is missing.
    """
    if Employee.objects.exists() and not force:
        return Employee.objects.count()

    if not INIT_SQL_PATH.exists():
        raise FileNotFoundError(f"init.sql không tồn tại: {INIT_SQL_PATH}")

    if force:
        Employee.objects.all().delete()

    sql = INIT_SQL_PATH.read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        if hasattr(cursor, "executescript"):
            cursor.executescript(sql)  # SQLite: runs everything at once.
        else:
            # MySQL / Postgres: strip leading '-- ...' comment lines, then split
            # on ';'. init.sql has no embedded semicolons inside values so a
            # naive split is safe.
            body = "\n".join(
                line for line in sql.splitlines() if not line.lstrip().startswith("--")
            )
            for stmt in filter(None, (s.strip() for s in body.split(";"))):
                cursor.execute(stmt)

    return Employee.objects.count()
