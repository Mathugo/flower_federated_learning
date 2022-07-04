"""Minimal example on how to start a simple Flower server."""
from src.server import ClassificationServer
from src.app.args import Args
from utils import MLFlowClient
import time

def main() -> None:
    """Start server and train five rounds."""
    serv = ClassificationServer(Args.get_args())
    serv.configure_strategy()
    # TODO start server with certificates 
    serv.start()
    #serv.model.save()
    time.sleep(60)
    serv.configure_strategy()
    serv.start()

if __name__ == "__main__":
    main()

