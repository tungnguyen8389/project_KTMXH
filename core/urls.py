from django.urls import path
from .views import (
    DatasetListAPIView,
    PreprocessingAPIView,
    RoughSetAPIView,
    ReductAPIView,
    KMeansAPIView,
    AprioriAPIView,
    EncodeTransactionsAPIView,
    ClassificationDataAPIView,
    ID3APIView,
    NaiveBayesAPIView
)

urlpatterns = [
    path('datasets/', DatasetListAPIView.as_view(), name='api_datasets'),
    path('preprocessing/', PreprocessingAPIView.as_view(), name='api_preprocessing'),
    path('rough-set/', RoughSetAPIView.as_view(), name='api_rough_set'),
    path('reduct/', ReductAPIView.as_view(), name='api_reduct'),
    path('kmeans/', KMeansAPIView.as_view(), name='api_kmeans'),
    path('apriori/', AprioriAPIView.as_view(), name='api_apriori'),
    path('encode-transactions/', EncodeTransactionsAPIView.as_view(), name='api_encode_transactions'),
    path('classification-data/', ClassificationDataAPIView.as_view(), name='api_classification_data'),
    path('id3/', ID3APIView.as_view(), name='api_id3'),
    path('naive-bayes/', NaiveBayesAPIView.as_view(), name='api_naive_bayes'),
]
