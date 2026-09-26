import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from core.data_cleaning import clean_dataframe
from core.models import Dataset

# Old sample datasets replaced by the HR Attrition set.
OLD_SEED_NAMES = [
    "Điểm thi học phần (Preprocessing)",
    "Hệ thông tin Tập thô 8 đối tượng mẫu",
    "Tập tọa độ 2D K-Means (8 điểm)",
    "Giao dịch siêu thị (Supermarket Transactions)",
    "Tập dữ liệu PlayTennis (ID3 & Naive Bayes)",
]

HR_DATASET_NAME = "IBM HR Employee Attrition (Classification)"
HR_CSV_PATH = Path(settings.BASE_DIR) / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


class Command(BaseCommand):
    help = "Seed the IBM HR Employee Attrition dataset as the baseline (replaces old samples)"

    def handle(self, *args, **options):
        if not HR_CSV_PATH.exists():
            self.stderr.write(f"CSV không tồn tại: {HR_CSV_PATH}")
            return

        # Remove old sample datasets so the HR set becomes the new baseline.
        deleted, _ = Dataset.objects.filter(name__in=OLD_SEED_NAMES).delete()
        if deleted:
            self.stdout.write(f"Đã xoá {deleted} bộ dữ liệu mẫu cũ.")

        df = pd.read_csv(HR_CSV_PATH)
        df, cleaning = clean_dataframe(df)
        self.stdout.write(
            f"Data cleaning: bỏ cột {cleaning['dropped_columns']}, "
            f"điền {cleaning['missing_filled']} ô thiếu, "
            f"còn {cleaning['rows_after']} dòng x {cleaning['columns_after']} cột."
        )
        # to_json -> loads keeps native Python scalars, JSONField-safe.
        records = json.loads(df.to_json(orient="records"))

        dataset, created = Dataset.objects.update_or_create(
            name=HR_DATASET_NAME,
            defaults={
                "category": "CLASSIFICATION",
                "description": (
                    f"Bộ dữ liệu IBM HR Employee Attrition: {len(records)} nhân viên, "
                    f"{len(df.columns)} thuộc tính. Thuộc tính quyết định: Attrition (Yes/No)."
                ),
                "data_json": records,
            },
        )

        action = "Tạo mới" if created else "Cập nhật"
        self.stdout.write(self.style.SUCCESS(
            f"{action} bộ dữ liệu '{HR_DATASET_NAME}' ({len(records)} dòng)."
        ))
