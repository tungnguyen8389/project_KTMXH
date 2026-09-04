from django.db import models

class Dataset(models.Model):
    """
    [TV1] Model managed by TV1 to store seed datasets and user custom datasets
    for each algorithm group.
    """
    CATEGORY_CHOICES = [
        ('PREPROCESSING', 'Tiền xử lý (Min-Max, Z-score)'),
        ('ROUGH_SET', 'Tập thô & Rút gọn thuộc tính'),
        ('KMEANS', 'Phân cụm K-Means'),
        ('APRIORI', 'Luật kết hợp Apriori'),
        ('CLASSIFICATION', 'Phân lớp (ID3 & Naive Bayes)'),
    ]

    name = models.CharField(max_length=250, verbose_name="Tên bộ dữ liệu")
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, verbose_name="Phân loại thuật toán")
    description = models.TextField(blank=True, verbose_name="Mô tả chi tiết")
    data_json = models.JSONField(verbose_name="Nội dung dữ liệu (JSON format)")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Bộ dữ liệu mẫu"
        verbose_name_plural = "Danh sách bộ dữ liệu mẫu"
        ordering = ['category', 'name']

    def __str__(self):
        return f"[{self.get_category_display()}] - {self.name}"


class ExecutionHistory(models.Model):
    """
    Model tracking algorithm execution logs for student reference.
    """
    algorithm_name = models.CharField(max_length=100)
    input_parameters = models.JSONField()
    execution_result = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Lịch sử thực thi"
        verbose_name_plural = "Lịch sử thực thi thuật toán"
        ordering = ['-created_at']
