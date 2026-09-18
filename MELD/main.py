import argparse
from pathlib import Path

from flask import Flask

from Logger import get_meld_logger
from ModelManager import ContractService
from ModelEnvironment import ExecutionService
from utils.config import API_HOST, API_PORT, CONTRACTS_DIR


def create_app(contract_service: ContractService | None = None,
               execution_service: ExecutionService | None = None) -> Flask:
    """Create the Flask application and register the MELD API endpoints."""
    from Server import bp

    app = Flask(__name__)
    app.register_blueprint(bp)
    app.extensions["contract_service"] = (
        contract_service if contract_service is not None else ContractService()
    )
    app.extensions["execution_service"] = (
        execution_service if execution_service is not None else ExecutionService()
    )
    app.logger = get_meld_logger()

    return app


app = create_app()


def _contract_path(contract: str) -> str:
    relative_path = Path(contract)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("The contract must be a relative path inside the contracts directory")
    return str(Path(CONTRACTS_DIR) / relative_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="MELD", description="Run MELD jobs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("run", "pull"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("contract")

    remove_parser = subparsers.add_parser("remove", aliases=["delete"])
    remove_parser.add_argument("contract")

    serve_parser = subparsers.add_parser(
        "serve",
        aliases=["server"],
        help="Start the Flask API server",
    )
    serve_parser.add_argument("--host", default=API_HOST)
    serve_parser.add_argument("--port", type=int, default=API_PORT)
    serve_parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    if args.command in {"remove", "delete"}:
        from ModelManager import load_contract, remove_runtime

        remove_runtime(load_contract(_contract_path(args.contract)))
    elif args.command == "pull":
        from ModelManager import load_contract, pull_runtime

        pull_runtime(load_contract(_contract_path(args.contract)))
    elif args.command == "run":
        from ModelManager import load_contract, run_inference

        run_inference(load_contract(_contract_path(args.contract)))
    elif args.command in {"serve", "server"}:
        app.run(host=args.host, port=args.port, debug=args.debug)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
