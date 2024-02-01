import sys

from .main import CONFIG, main as program


def main() -> int:
    program(CONFIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
