
# Federated Learning for Computer Vision
Federated Learning for vision classification using Flower and Pytorch

## Description

This repository implements several models and custom strategies for federated learning in computer vision using flower for multilabel classification.

## Getting Started

### Dependencies

* Docker

### File structure

```text
flower_federated_learning/
└── ofb-flower/
    ├── server/
    │   ├── flower/
    │   │   ├── run.sh
    │   │   ├── requirements.txt
    │   │   ├── ...
    │   │   └── src/
    │   ├── build_run.sh
    │   ├── ...
    │   └── mlflow/
    └── client/
        ├── src/
        ├── requirements.txt
        ├── ...
        └── run_python.sh   

```

### Setting environment
Environment variables used in docker-compose are in *client/.env* and *server/.env*
You have to set at least the correct IP address and port for the clients to target mlflow and flower server.

### Start server and clients
server and client builder located respectively in ofb-flower/server and ofb-flower/client
```
bash build_run.sh
```

## Help

If you want to run server without docker make sure you have the necessary requirements for *Python* packages and at least *Python3.7*

```
python3 -m pip install requirements.txt
```

## Authors

Contributors names and contact info

Hugo Math [@mathugo](https://hugomath.com/)

