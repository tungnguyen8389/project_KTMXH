from django.contrib import admin
from .models import Employee, ExecutionHistory

@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('id', 'Age', 'Department', 'JobRole', 'Attrition')
    list_filter = ('Attrition', 'Department', 'JobRole')
    search_fields = ('JobRole', 'Department')

@admin.register(ExecutionHistory)
class ExecutionHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'algorithm_name', 'created_at')
    ordering = ('-created_at',)
