from django.contrib import admin
from .models import Dataset, ExecutionHistory

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'category', 'created_at')
    list_filter = ('category',)
    search_fields = ('name', 'description')

@admin.register(ExecutionHistory)
class ExecutionHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'algorithm_name', 'created_at')
    ordering = ('-created_at',)
