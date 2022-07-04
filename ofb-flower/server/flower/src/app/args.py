import argparse

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
            "--rounds",
            type=int,
            default=1,
            help="Number of rounds of federated learning (default: 1)",
        )
        parser.add_argument(
            "--fraction_fit",
            type=float,
            default=0.5,
            help="Fraction of available clients used for fit (default: 0.5)",
        )
        parser.add_argument(
            "--fraction_eval",
            type=float,
            default=0.5,
            help="Fraction of available clients used for evaluate (default: 0.5)",
        )
        parser.add_argument(
            "--min_sample_size",
            type=int,
            default=2,
            help="Minimum number of clients used for fit/evaluate (default: 2)",
        )
        parser.add_argument(
            "--min_num_clients",
            type=int,
            default=2,
            help="Minimum number of available clients required for sampling (default: 2)",
        )
        parser.add_argument(
            "--log_host",
            type=str,
            help="Logserver address (no default)",
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
            default="3",
            help="Number of classes of the model",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="training batch size",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="number of workers for dataset reading",
        )
        parser.add_argument(
            "--local_epochs",
            type=int,
            default=3,
            help="local epochs to perform on each client"
        )
        parser.add_argument(
            "--mlflow_server_ip",
            type=str,
            default="localhost",
            help="IP address of the mlflow server"
        )
        parser.add_argument(
            "--mlflow_server_port",
            type=int,
            default=4040,
            help="Port of the mlflow server"
        )
        parser.add_argument("--pin_memory", action="store_true")
        parser.add_argument("--load_weights", action="store_true")
        return parser.parse_args()