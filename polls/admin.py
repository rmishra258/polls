from django.contrib import admin

# Register your models here.

from .models import Question , Choices

admin.site.register(Question)
admin.site.register(Choices)
