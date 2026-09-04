from django.core.management.base import BaseCommand
from core.models import Dataset

class Command(BaseCommand):
    help = 'Seed baseline datasets for all 5 algorithm groups'

    def handle(self, *args, **options):
        self.stdout.write("Populating baseline seed datasets...")

        # 1. Preprocessing Dataset
        ds_prep, _ = Dataset.objects.get_or_create(
            name="Điểm thi học phần (Preprocessing)",
            category="PREPROCESSING",
            defaults={
                "description": "Bảng điểm thi học phần môn Khai thác dữ liệu của sinh viên",
                "data_json": [
                    {"sv_id": "SV01", "name": "Nguyễn Văn A", "score": 4.5},
                    {"sv_id": "SV02", "name": "Trần Thị B", "score": 6.0},
                    {"sv_id": "SV03", "name": "Lê Văn C", "score": 7.5},
                    {"sv_id": "SV04", "name": "Phạm Thị D", "score": 9.0},
                    {"sv_id": "SV05", "name": "Hoàng Văn E", "score": 10.0}
                ]
            }
        )

        # 2. Rough Set & Reduct Dataset (Classic 10-object Information System)
        ds_rs, _ = Dataset.objects.get_or_create(
            name="Hệ thông tin Tập thô 8 đối tượng mẫu",
            category="ROUGH_SET",
            defaults={
                "description": "Bảng dữ liệu tập thô gồm các thuộc tính điều kiện (a, b, c, d) và quyết định (e)",
                "data_json": [
                    {"id": "x1", "a": 1, "b": 1, "c": 0, "d": 1, "e": 1},
                    {"id": "x2", "a": 1, "b": 0, "c": 0, "d": 1, "e": 1},
                    {"id": "x3", "a": 0, "b": 0, "c": 0, "d": 0, "e": 0},
                    {"id": "x4", "a": 1, "b": 1, "c": 0, "d": 0, "e": 0},
                    {"id": "x5", "a": 0, "b": 1, "c": 0, "d": 1, "e": 1},
                    {"id": "x6", "a": 0, "b": 0, "c": 0, "d": 1, "e": 0},
                    {"id": "x7", "a": 1, "b": 0, "c": 1, "d": 1, "e": 1},
                    {"id": "x8", "a": 0, "b": 1, "c": 1, "d": 0, "e": 0}
                ]
            }
        )

        # 3. K-Means Dataset (2D Spatial Coordinates)
        ds_km, _ = Dataset.objects.get_or_create(
            name="Tập tọa độ 2D K-Means (8 điểm)",
            category="KMEANS",
            defaults={
                "description": "Các tọa độ không gian 2D để thử nghiệm thuật toán K-Means",
                "data_json": [
                    {"id": "P1", "x": 2.0, "y": 10.0},
                    {"id": "P2", "x": 2.0, "y": 5.0},
                    {"id": "P3", "x": 8.0, "y": 4.0},
                    {"id": "P4", "x": 5.0, "y": 8.0},
                    {"id": "P5", "x": 7.0, "y": 5.0},
                    {"id": "P6", "x": 6.0, "y": 4.0},
                    {"id": "P7", "x": 1.0, "y": 2.0},
                    {"id": "P8", "x": 4.0, "y": 9.0}
                ]
            }
        )

        # 4. Apriori Dataset (Supermarket Transactions)
        ds_ap, _ = Dataset.objects.get_or_create(
            name="Giao dịch siêu thị (Supermarket Transactions)",
            category="APRIORI",
            defaults={
                "description": "Bảng giao dịch mua sắm của các khách hàng tại siêu thị",
                "data_json": [
                    {"tid": "T1", "items": ["Sữa", "Tã", "Bia"]},
                    {"tid": "T2", "items": ["Sữa", "Tã", "Bánh mì"]},
                    {"tid": "T3", "items": ["Sữa", "Bia", "Bánh mì"]},
                    {"tid": "T4", "items": ["Tã", "Bia", "Bánh mì"]},
                    {"tid": "T5", "items": ["Sữa", "Tã", "Bia", "Bánh mì"]}
                ]
            }
        )

        # 5. PlayTennis Classification Dataset (ID3 & Naive Bayes)
        ds_cl, _ = Dataset.objects.get_or_create(
            name="Tập dữ liệu PlayTennis (ID3 & Naive Bayes)",
            category="CLASSIFICATION",
            defaults={
                "description": "Bộ dữ liệu thời tiết kinh điển PlayTennis / Golf Weather",
                "data_json": [
                    {"Day": "D1", "Outlook": "Sunny", "Temp": "Hot", "Humidity": "High", "Wind": "Weak", "Play": "No"},
                    {"Day": "D2", "Outlook": "Sunny", "Temp": "Hot", "Humidity": "High", "Wind": "Strong", "Play": "No"},
                    {"Day": "D3", "Outlook": "Overcast", "Temp": "Hot", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
                    {"Day": "D4", "Outlook": "Rain", "Temp": "Mild", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
                    {"Day": "D5", "Outlook": "Rain", "Temp": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
                    {"Day": "D6", "Outlook": "Rain", "Temp": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "No"},
                    {"Day": "D7", "Outlook": "Overcast", "Temp": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
                    {"Day": "D8", "Outlook": "Sunny", "Temp": "Mild", "Humidity": "High", "Wind": "Weak", "Play": "No"},
                    {"Day": "D9", "Outlook": "Sunny", "Temp": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
                    {"Day": "D10", "Outlook": "Rain", "Temp": "Mild", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
                    {"Day": "D11", "Outlook": "Sunny", "Temp": "Mild", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
                    {"Day": "D12", "Outlook": "Overcast", "Temp": "Mild", "Humidity": "High", "Wind": "Strong", "Play": "Yes"},
                    {"Day": "D13", "Outlook": "Overcast", "Temp": "Hot", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
                    {"Day": "D14", "Outlook": "Rain", "Temp": "Mild", "Humidity": "High", "Wind": "Strong", "Play": "No"}
                ]
            }
        )

        self.stdout.write(self.style.SUCCESS("Successfully seeded baseline datasets!"))
