import sys, mlflow
import flwr as fl
sys.path.append("..")
from utils import MLFlowClient
from src.client import GearClassifyClient
from src.app import Args

def main() -> None:
    args = Args.get_args()
    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    mlflow_client = MLFlowClient(args.name, args.mlflow_server_ip, args.mlflow_server_port)
    client = GearClassifyClient(args, mlflow_client=mlflow_client)
    
    addr = f"{args.server_address}:{args.server_port}"
    print(f"[CLIENT] Starting client to {addr}")
    # each time you start client -> start ml experiment 
    mlflow_client.set_experiment(f"{args.cid}-{mlflow_client.ClientName}")

    # Start a run with federated learning and mlflow
    with mlflow.start_run(run_name=f"{args.run_name}") as run:
        fl.client.start_client(addr, client)

if __name__ == "__main__":
    main()


# TODO LOAD MODEL WITH CONFIG