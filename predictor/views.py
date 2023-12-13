from django.shortcuts import render
from django.http import JsonResponse
from .models import Essay
from transformers import BertForSequenceClassification, BertTokenizer
import torch


def index(request):
    return render(request, 'predictor/index.html')


def predict(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        model_path = './bert_model'
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        # Tokenize and classify the essay
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        predicted = 'Student' if predicted_class == 0 else 'Generated'

        essay = Essay.objects.create(text=text, predicted=predicted_class)
        essay.save()

        return JsonResponse({'predicted': predicted})
    return JsonResponse({'error': 'Invalid request'})
