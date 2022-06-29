import argparse, torch, sys
import flwr as fl
sys.path.append("..")
from utils import load_model, load_classify_dataset
from src.client import GearClassifyClient
from src.pipeline import resnet18_transform
from torchsummary import summary 

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

class Args:
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="Flower")
        parser.add_argument(
            "--server_address",
            type=str,
            required=True,
            help=f"gRPC server address",
        )
        parser.add_argument(
            "--cid", type=str, required=True, help="Client CID (no default)"
        )
        parser.add_argument(
            "--log_host",
            type=str,
            help="Logserver address (no default)",
        )
        parser.add_argument(
            "--data_dir",
            default="data/Gear_Classify.v3-gear-fl-raw",
            type=str,
            help="Directory where the dataset lives",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="ResNet18",
            choices=["HugoNet", "ResNet18", "ViT"],
            help="model to train",
        )
        parser.add_argument(
            "--n_classes",
            type=int,
            default=3,
            help="number of classes to detect"
        )
        return parser.parse_args()

def main() -> None:
    """Load data, create and start CifarClient."""
    args = Args.get_args()
    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)
    model, trf = load_model(args.model, n_classes=args.n_classes)
    trainset, testset = load_classify_dataset(args.data_dir, transforms=trf)
    model.to(DEVICE)

    # Start client TODO if server not reachable, start the inference and load old weights 
    client = GearClassifyClient(args.cid, model, trainset, testset)
    print("[CLIENT] Starting client ..")
    fl.client.start_client(args.server_address, client)

if __name__ == "__main__":
    main()

