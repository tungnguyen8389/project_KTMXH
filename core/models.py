from django.db import models


class Employee(models.Model):
    """
    One IBM HR Employee Attrition record (already cleaned by data_cleaning).
    This relational table is the single source of truth for every algorithm
    tab; it is seeded from data/init.sql on first run (see core/seeding.py).

    Fields carry defaults so partial rows (tests, admin) can be created without
    supplying all 31 columns; the SQL seed always provides full values.
    """
    # Categorical / text columns.
    Attrition = models.CharField(max_length=100, default="", verbose_name="Nghỉ việc")
    BusinessTravel = models.CharField(max_length=100, default="")
    Department = models.CharField(max_length=100, default="")
    EducationField = models.CharField(max_length=100, default="")
    Gender = models.CharField(max_length=100, default="")
    JobRole = models.CharField(max_length=100, default="")
    MaritalStatus = models.CharField(max_length=100, default="")
    OverTime = models.CharField(max_length=100, default="")

    # Integer columns.
    Age = models.IntegerField(default=0)
    DailyRate = models.IntegerField(default=0)
    DistanceFromHome = models.IntegerField(default=0)
    Education = models.IntegerField(default=0)
    EnvironmentSatisfaction = models.IntegerField(default=0)
    HourlyRate = models.IntegerField(default=0)
    JobInvolvement = models.IntegerField(default=0)
    JobLevel = models.IntegerField(default=0)
    JobSatisfaction = models.IntegerField(default=0)
    MonthlyIncome = models.IntegerField(default=0)
    MonthlyRate = models.IntegerField(default=0)
    PercentSalaryHike = models.IntegerField(default=0)
    PerformanceRating = models.IntegerField(default=0)
    RelationshipSatisfaction = models.IntegerField(default=0)
    WorkLifeBalance = models.IntegerField(default=0)
    YearsAtCompany = models.IntegerField(default=0)

    # Float columns (median-filled / IQR-capped during cleaning may be fractional).
    NumCompaniesWorked = models.FloatField(default=0.0)
    StockOptionLevel = models.FloatField(default=0.0)
    TotalWorkingYears = models.FloatField(default=0.0)
    TrainingTimesLastYear = models.FloatField(default=0.0)
    YearsInCurrentRole = models.FloatField(default=0.0)
    YearsSinceLastPromotion = models.FloatField(default=0.0)
    YearsWithCurrManager = models.FloatField(default=0.0)

    class Meta:
        verbose_name = "Nhân viên (HR Attrition)"
        verbose_name_plural = "Bộ dữ liệu nhân viên (HR Attrition)"

    def __str__(self):
        return f"Employee #{self.pk} - Attrition={self.Attrition}"


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
