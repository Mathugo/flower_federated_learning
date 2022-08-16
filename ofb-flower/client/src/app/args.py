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
            "--server_port",
            type=str,
            required=False,
            default=5000, 
            help=f"gRPC server port"
        )
        parser.add_argument(
            "--cid", type=str, required=True, help="Client CID (no default)"
        )
        parser.add_argument(
            "--name", type=str, required=True, help="Client name (no default)"
        )
        parser.add_argument(
            "--run_name", type=str, help="name of run for mlflow (default: random_run)", default="random_run"
        )
        parser.add_argument(
            "--log_host",
            type=str,
            help="Logserver address (no default)",
        )
        parser.add_argument(
            "--data_dir",
            default="data/Gear3",
            type=str,
            help="Directory where the dataset lives",
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
        parser.add_argument("--data_augmentation", action="store_true")
        parser.add_argument("--load_mlflow_model", action="store_true")
        return parser.parse_args()
