#!bin/bash
docker compose up --build --wait --remove-orphans --force-recreate db web nginx flower 
#docker inspect mlflow
docker compose logs --follow