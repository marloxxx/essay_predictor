# predictor/models.py

from django.db import models


class Essay(models.Model):
    text = models.TextField()
    predicted = models.IntegerField(choices=[(0, 'Student'), (1, 'Generated')])
