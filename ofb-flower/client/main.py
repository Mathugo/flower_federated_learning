import torch, sys
import flwr as fl
sys.path.append("..")
from utils import load_model, load_classify_dataset, MLFlowClient
from src.client import GearClassifyClient
from src.app import Args

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main() -> None:
    args = Args.get_args()
    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)
    model, trf = load_model(args.model, False, n_classes=args.n_classes)
    trainset, testset = load_classify_dataset(args.data_dir, transforms=trf, data_augmentation=args.data_augmentation)
    model.to(DEVICE)

    mlflow_client = MLFlowClient(args.name, args.mlflow_server_ip, args.mlflow_server_port)
    #model.save_pqt_quantized(testset[0])
    # Start client TODO if server not reachable, start the inference and load old weights 
    client = GearClassifyClient(args.cid, model, trainset, testset, mlflow_client=mlflow_client)
    addr = f"{args.server_address}:{args.server_port}"
    print(f"[CLIENT] Starting client to {addr}")

    # each time you start client -> start ml experiment 
    mlflow_client.new_experiment(f"{args.cid}-{mlflow_client.ClientName}")
    fl.client.start_client(addr, client)

if __name__ == "__main__":
    main()


