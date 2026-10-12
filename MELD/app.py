from flask import Flask

from ExecutionScheduler import SchedulerService
from Logger import get_meld_logger
from ModelEnvironment import ExecutionService
from ModelManager import ContractService


def create_app() -> Flask:
    """Create the Flask application and register the MELD API endpoints."""
    from Server import bp

    app = Flask(__name__)

    app.register_blueprint(bp)

    execution_service = ExecutionService()
    scheduler_service = SchedulerService(execution_service)
    contract_service = ContractService(scheduler_service=scheduler_service, execution_service=execution_service)
    app.extensions["scheduler_service"] = scheduler_service
    app.extensions["contract_service"] = contract_service
    app.extensions["execution_service"] = execution_service
    contract_service.restore_scheduler_jobs()
    scheduler_service.start()
    app.logger = get_meld_logger()

    return app
