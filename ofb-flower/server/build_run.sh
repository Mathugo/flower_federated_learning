#!bin/bash
docker-compose down
docker compose up --build --wait --remove-orphans --force-recreate db web nginx flower
#docker inspect mlflow
docker compose logs --follow db web flower