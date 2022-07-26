"""Minimal example on how to start a simple Flower server."""
from src.server import ClassificationServer
from src.app.args import Args
import time, sys

def main() -> None:
    """Start server and train five rounds."""
    serv = ClassificationServer(Args.get_args())
    while True:
        serv.configure_strategy()
        # TODO start server with certificates 
        serv.start()
        print("[SERVER] Strategy finished, restarting the server ..", file=sys.stderr)
        time.sleep(10)

if __name__ == "__main__":
    main()

