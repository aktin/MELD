from pathlib import Path

version = Path(__file__).with_name("VERSION").read_text().strip()
