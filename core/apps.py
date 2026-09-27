import sys

from django.apps import AppConfig
from django.db import connection

# Management commands that must NOT trigger seeding (they may run before the
# table exists, or must see an empty table). Anything else — runserver and WSGI
# servers in production — is treated as "the app is starting" and gets seeded.
_SKIP_COMMANDS = {
    "makemigrations", "migrate", "test", "collectstatic", "shell",
    "createsuperuser", "dumpdata", "loaddata", "showmigrations",
    "sqlmigrate", "check", "flush", "seed_datasets",
}


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        # Auto-seed the HR dataset from data/init.sql when the app starts serving.
        if len(sys.argv) > 1 and sys.argv[1] in _SKIP_COMMANDS:
            return
        try:
            from .models import Employee
            from .seeding import seed_employees

            if Employee._meta.db_table not in connection.introspection.table_names():
                return
            seed_employees()
        except Exception as exc:  # never block startup on a seed failure
            sys.stderr.write(f"[core] Auto-seed bỏ qua: {exc}\n")
