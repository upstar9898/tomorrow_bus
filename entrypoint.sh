#!/bin/sh

echo "Waiting for database..."

while ! nc -z $DB_HOST $DB_PORT; do
  sleep 1
done

echo "Database started"

python manage.py migrate --noinput
python manage.py collectstatic --noinput

gunicorn tommorow_bus.wsgi:application --bind 0.0.0.0:8000