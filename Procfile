release: python manage.py makemigrations && python manage.py migrate
web: gunicorn recruitai_backend_interview.wsgi
worker: celery -A recruitai_backend_interview worker -l info