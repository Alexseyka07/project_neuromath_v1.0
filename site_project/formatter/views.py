from django.shortcuts import render
from .format import to_expr


def index(request):
    if request.method == 'POST':
        input = request.POST['input']
        result = to_expr(input)
        return render(request, 'formatter/index.html', {'result': result, 'input': input})
        
    return render(request, 'formatter/index.html')  
