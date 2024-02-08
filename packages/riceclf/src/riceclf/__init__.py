import sys

from .main import main as program


def main() -> int:
    program()
    return 0


if __name__ == "__main__":
    sys.exit(main())
