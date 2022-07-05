#!bin/bash
docker compose up --build --wait --remove-orphans --force-recreate sftp db web nginx flower 
docker compose logs --follow