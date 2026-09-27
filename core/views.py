import pandas as pd
from django.views.generic import TemplateView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import Employee
from .seeding import hr_records
from .algorithms.preprocessing import PreprocessingEngine
from .algorithms.rough_set import RoughSetEngine
from .algorithms.reduct import ReductEngine
from .algorithms.kmeans import KMeansEngine
from .algorithms.association import AprioriEngine
from .algorithms.classification import ClassificationEngine
from .transaction_encoder import encode_transactions, encode_records


class IndexView(TemplateView):
    """[TV4] Render main interactive visualizer workspace dashboard"""
    template_name = 'index.html'


class DatasetListAPIView(APIView):
    """[TV3] Return the seeded HR dataset as a single dataset entry.

    Kept as a list of one so the frontend contract (list[0].data_json) is
    unchanged after switching the data source from a JSON blob to the
    relational Employee table.
    """
    def get(self, request):
        records = hr_records()
        if not records:
            return Response([], status=status.HTTP_200_OK)
        data = [{
            'id': 1,
            'name': 'IBM HR Employee Attrition',
            'category': 'CLASSIFICATION',
            'description': f'Bộ dữ liệu nhân sự: {len(records)} nhân viên, '
                           f'{len(records[0])} thuộc tính.',
            'data_json': records,
        }]
        return Response(data, status=status.HTTP_200_OK)


class PreprocessingAPIView(APIView):
    """[TV3] Dispatcher API for Min-Max and Z-Score Normalization"""
    def post(self, request):
        try:
            payload = request.data
            algo_type = payload.get('type', 'minmax')
            data_list = payload.get('data', [])
            feature_name = payload.get('feature')
            
            if not data_list or not feature_name:
                return Response({'error': 'Thiếu tham số data hoặc feature!'}, status=status.HTTP_400_BAD_REQUEST)

            if algo_type == 'minmax':
                new_min = float(payload.get('new_min', 0.0))
                new_max = float(payload.get('new_max', 1.0))
                result = PreprocessingEngine.min_max_normalize(data_list, feature_name, new_min, new_max)
            else:
                result = PreprocessingEngine.z_score_normalize(data_list, feature_name)

            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class RoughSetAPIView(APIView):
    """[TV3] Dispatcher API for Rough Set Approximations & Dependency Degree k"""
    def post(self, request):
        try:
            payload = request.data
            data_list = payload.get('data', [])
            condition_attrs = payload.get('condition_attrs', [])
            decision_attr = payload.get('decision_attr')

            if not data_list or not condition_attrs or not decision_attr:
                return Response({'error': 'Thiếu tham số data, condition_attrs hoặc decision_attr!'}, status=status.HTTP_400_BAD_REQUEST)

            result = RoughSetEngine.analyze_rough_set(data_list, condition_attrs, decision_attr)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ReductAPIView(APIView):
    """[TV3] Dispatcher API for Discriminiability Matrix n x n & Boolean Reducts"""
    def post(self, request):
        try:
            payload = request.data
            data_list = payload.get('data', [])
            condition_attrs = payload.get('condition_attrs', [])
            decision_attr = payload.get('decision_attr')

            if not data_list or not condition_attrs or not decision_attr:
                return Response({'error': 'Thiếu tham số data, condition_attrs hoặc decision_attr!'}, status=status.HTTP_400_BAD_REQUEST)

            result = ReductEngine.compute_reducts(data_list, condition_attrs, decision_attr)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


NUMERIC_HR_FEATURES = [
    {"field": "Age", "label": "Tuổi (Age)"},
    {"field": "MonthlyIncome", "label": "Thu nhập hàng tháng ($)"},
    {"field": "TotalWorkingYears", "label": "Tổng số năm kinh nghiệm"},
    {"field": "YearsAtCompany", "label": "Số năm tại công ty"},
    {"field": "DistanceFromHome", "label": "Khoảng cách đến cty (km)"},
    {"field": "DailyRate", "label": "Mức lương theo ngày ($)"},
    {"field": "HourlyRate", "label": "Mức lương theo giờ ($)"},
    {"field": "PercentSalaryHike", "label": "% Tăng lương gần nhất"},
    {"field": "YearsInCurrentRole", "label": "Số năm ở vị trí hiện tại"},
    {"field": "YearsSinceLastPromotion", "label": "Số năm từ lần thăng chức cuối"},
    {"field": "YearsWithCurrManager", "label": "Số năm làm việc với sếp hiện tại"},
    {"field": "WorkLifeBalance", "label": "Cân bằng công việc/cuộc sống (1-4)"},
    {"field": "JobSatisfaction", "label": "Hài lòng công việc (1-4)"},
    {"field": "EnvironmentSatisfaction", "label": "Hài lòng môi trường (1-4)"},
]


class KMeansAPIView(APIView):
    """[TV3] Dispatcher API for K-Means Clustering step-by-step from DB or custom JSON"""
    def get(self, request):
        """Return available numeric features and total employee count in DB"""
        emp_count = Employee.objects.count()
        return Response({
            "features": NUMERIC_HR_FEATURES,
            "total_employees": emp_count
        }, status=status.HTTP_200_OK)

    def post(self, request):
        try:
            payload = request.data
            use_db = payload.get('use_db', False) or (payload.get('source') == 'db')
            k = int(payload.get('k', 3))
            max_iter = int(payload.get('max_iter', 15))
            init_method = payload.get('init_method', 'kmeans++')
            normalize = bool(payload.get('normalize', False))
            initial_centroids = payload.get('initial_centroids', None)

            feature_x_name = "X"
            feature_y_name = "Y"

            if use_db or ('feature_x' in payload and 'feature_y' in payload and not payload.get('points')):
                feature_x = payload.get('feature_x', 'Age')
                feature_y = payload.get('feature_y', 'MonthlyIncome')
                sample_size = int(payload.get('sample_size', 150))

                feature_x_name = next((f['label'] for f in NUMERIC_HR_FEATURES if f['field'] == feature_x), feature_x)
                feature_y_name = next((f['label'] for f in NUMERIC_HR_FEATURES if f['field'] == feature_y), feature_y)

                qs = Employee.objects.all()
                if not qs.exists():
                    return Response({'error': 'Cơ sở dữ liệu nhân viên đang trống. Hãy kiểm tra lại data/init.sql!'}, status=status.HTTP_400_BAD_REQUEST)

                if sample_size > 0:
                    qs = qs[:sample_size]

                points = []
                for emp in qs:
                    val_x = getattr(emp, feature_x, 0.0)
                    val_y = getattr(emp, feature_y, 0.0)
                    points.append({
                        'id': f"NV{emp.pk}",
                        'x': float(val_x),
                        'y': float(val_y),
                        'meta': {
                            'attrition': emp.Attrition,
                            'job_role': emp.JobRole,
                            'department': emp.Department
                        }
                    })
            else:
                points = payload.get('points', [])
                if not points:
                    return Response({'error': 'Danh sách điểm points không được rỗng!'}, status=status.HTTP_400_BAD_REQUEST)

            result = KMeansEngine.run_kmeans(
                points=points,
                k=k,
                initial_centroids=initial_centroids,
                max_iter=max_iter,
                init_method=init_method,
                normalize=normalize
            )
            result['feature_x_name'] = feature_x_name
            result['feature_y_name'] = feature_y_name
            result['total_points'] = len(points)

            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)



class AprioriAPIView(APIView):
    """[TV3] Dispatcher API for Apriori Association Rule Mining"""
    def post(self, request):
        try:
            payload = request.data
            transactions = payload.get('transactions', [])
            min_supp_pct = float(payload.get('min_supp', 50.0))
            min_conf_pct = float(payload.get('min_conf', 70.0))
            min_lift = float(payload.get('min_lift', 0.0))
            max_len_raw = payload.get('max_len', None)
            max_len = int(max_len_raw) if max_len_raw not in (None, '', 0, '0') else None
            target_mode = payload.get('target_mode', 'all')
            analysis_mode = payload.get('analysis_mode', 'class')

            if not transactions:
                return Response({'error': 'Danh sách giao dịch transactions không được rỗng!'}, status=status.HTTP_400_BAD_REQUEST)

            result = AprioriEngine.run_apriori(
                transactions, min_supp_pct, min_conf_pct,
                max_len=max_len, min_lift=min_lift, target_mode=target_mode,
                analysis_mode=analysis_mode)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class EncodeTransactionsAPIView(APIView):
    """[TV3] Encode the stored HR dataset into Apriori transactions (GET)."""
    def get(self, request):
        if not Employee.objects.exists():
            return Response({'error': 'Chưa có dữ liệu nhân sự trong hệ thống.'},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            df = pd.DataFrame(hr_records())
            transactions, report = encode_transactions(df)
            return Response({'transactions': transactions, 'report': report},
                            status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ClassificationDataAPIView(APIView):
    """[TV3] Prefill classification tab (ID3 / Naive Bayes) from the HR DB.

    Reuses the Apriori binning so numeric columns come back as readable labels.
    Response shape matches what tab_classification.html needs to fill textarea +
    condition/target fields directly.
    """
    def get(self, request):
        if not Employee.objects.exists():
            return Response({'error': 'Chưa có dữ liệu nhân sự trong hệ thống.'},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            df = pd.DataFrame(hr_records())
            records, report = encode_records(df)
            return Response({'data': records, 'report': report},
                            status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ID3APIView(APIView):
    """[TV3] Dispatcher API for ID3 Decision Tree Construction & Mermaid graph"""
    def post(self, request):
        try:
            payload = request.data
            data_list = payload.get('data', [])
            condition_attrs = payload.get('condition_attrs', [])
            target_attr = payload.get('target_attr')

            if not data_list or not condition_attrs or not target_attr:
                return Response({'error': 'Thiếu tham số data, condition_attrs hoặc target_attr!'}, status=status.HTTP_400_BAD_REQUEST)

            criterion = payload.get('criterion', 'gain')
            result = ClassificationEngine.run_id3(data_list, condition_attrs, target_attr, criterion=criterion)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class NaiveBayesAPIView(APIView):
    """[TV3] Dispatcher API for Naive Bayes Classifier"""
    def post(self, request):
        try:
            payload = request.data
            data_list = payload.get('data', [])
            condition_attrs = payload.get('condition_attrs', [])
            target_attr = payload.get('target_attr')
            test_instance = payload.get('test_instance', {})
            use_laplace = bool(payload.get('use_laplace', False))

            if not data_list or not condition_attrs or not target_attr or not test_instance:
                return Response({'error': 'Thiếu tham số data, condition_attrs, target_attr hoặc test_instance!'}, status=status.HTTP_400_BAD_REQUEST)

            result = ClassificationEngine.run_naive_bayes(data_list, condition_attrs, target_attr, test_instance, use_laplace)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
