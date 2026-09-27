from django.core.management.base import BaseCommand

from core.seeding import seed_employees


class Command(BaseCommand):
    help = "Seed the Employee table from data/init.sql (the HR dataset source of record)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force", action="store_true",
            help="Wipe the Employee table and re-seed from data/init.sql.",
        )

    def handle(self, *args, **options):
        count = seed_employees(force=options["force"])
        self.stdout.write(self.style.SUCCESS(
            f"Bảng Employee hiện có {count} dòng (nguồn: data/init.sql)."
        ))
