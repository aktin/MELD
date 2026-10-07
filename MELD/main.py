from app import create_app
from utils.config import API_HOST, API_PORT

app = create_app()


def main() -> None:
    app.run(host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
