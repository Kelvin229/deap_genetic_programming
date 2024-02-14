# deap_genetic_programming

## Requirements

- Python
- Pip

## Usage

1. Install assignment packages

```sh
$ pip install -e packages/symreg packages/riceclf
```

2. Run the symbolic regression experiments

```sh
$ symreg
```

- For usage help:

```sh
$ symreg --help
```

3. Run the rice classification experiment

```sh
$ riceclf experiment -o tmp/result.csv --seed 123 456 789 2974 2479 24755 74593 57993 24749 279
$ riceclf plot tmp/result.csv tmp/result.png
```

- For an individual run, do:

```sh
$ riceclf run -o tmp/run123.csv --seed 123
```

- For usage help:

```sh
$ riceclf --help
$ riceclf run --help
$ riceclf experiment --help
$ riceclf plot --help
```
