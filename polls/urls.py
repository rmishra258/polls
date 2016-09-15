from django.conf.urls import url
from . import views


urlpatterns = [

    url(r'^$',views.index, name='index'),
    #127.0.0.1/polls ie what we defined in the mysite.urls.py
    url(r'^(?P<question_id>[0-9]+)/$',views.detail, name='detail'),
    url(r'^(?P<question_id>[0-9]+)/results$', views.results, name='results'),
    url(r'^(?P<question_id>[0-9]+)/vote$', views.vote, name='vote'),

]

