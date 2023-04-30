from django.urls import path
from deploy.views import MLModel

urlpatterns = [
    path('deploy/', MLModel.as_view())
]
