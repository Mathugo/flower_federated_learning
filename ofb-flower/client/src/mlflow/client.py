import mlflow

class MLFlowClient:
    def __init__(self, client_name: str, server_ip: str="localhost", port: int=4040):
        self._server_ip = server_ip
        self._server_port = port
        self._client_name = client_name
        self._tracking_uri = f"http://{self._server_ip}:{self._server_port}"
        print(f"[MLFlow] Connecting to {self._tracking_uri}") 
        mlflow.set_tracking_uri(self._tracking_uri)
        print(f"[MLFlow] Done ..")
    
    @property
    def ClientName(self):
        return self._client_name

    def new_experiment(self, name: str):
        print(f"[MLFLow] New experiment {name} created !")
        mlflow.set_experiment(name)
    
    def __del__(self):
        pass
