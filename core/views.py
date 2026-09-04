import json
from django.views.generic import TemplateView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import Dataset
from .algorithms.preprocessing import PreprocessingEngine
from .algorithms.rough_set import RoughSetEngine
from .algorithms.reduct import ReductEngine
from .algorithms.kmeans import KMeansEngine
from .algorithms.association import AprioriEngine
from .algorithms.classification import ClassificationEngine


class IndexView(TemplateView):
    """[TV4] Render main interactive visualizer workspace dashboard"""
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['datasets'] = Dataset.objects.all()
        return context


class DatasetListAPIView(APIView):
    """[TV3] REST API endpoint returning pre-loaded seed datasets"""
    def get(self, request):
        datasets = Dataset.objects.all()
        data = [{
            'id': ds.id,
            'name': ds.name,
            'category': ds.category,
            'description': ds.description,
            'data_json': ds.data_json
        } for ds in datasets]
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


class KMeansAPIView(APIView):
    """[TV3] Dispatcher API for K-Means Clustering step-by-step"""
    def post(self, request):
        try:
            payload = request.data
            points = payload.get('points', [])
            k = int(payload.get('k', 2))
            initial_centroids = payload.get('initial_centroids', None)
            max_iter = int(payload.get('max_iter', 10))

            if not points:
                return Response({'error': 'Danh sách điểm points không được rỗng!'}, status=status.HTTP_400_BAD_REQUEST)

            result = KMeansEngine.run_kmeans(points, k, initial_centroids, max_iter)
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

            if not transactions:
                return Response({'error': 'Danh sách giao dịch transactions không được rỗng!'}, status=status.HTTP_400_BAD_REQUEST)

            result = AprioriEngine.run_apriori(transactions, min_supp_pct, min_conf_pct)
            return Response(result, status=status.HTTP_200_OK)
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

            result = ClassificationEngine.run_id3(data_list, condition_attrs, target_attr)
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
