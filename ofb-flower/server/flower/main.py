"""Minimal example on how to start a simple Flower server."""
from src.server import ClassificationServer
from src.app.args import Args
import time, sys, mlflow

def main() -> None:
    """Start server and train five rounds."""
    serv = ClassificationServer(Args())
    mlflow.end_run()
    while True:
        serv.configure_strategy()
        # TODO start server with certificates-
        serv.start(run_name="FedAvg-reshugonet")
        print("[SERVER] Strategy finished, restarting the server ..", file=sys.stderr)
        time.sleep(10)

if __name__ == "__main__":
    main()

