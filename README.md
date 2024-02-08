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
$ riceclf
```

- For reproducible results, specify a random `seed`:

```sh
$ riceclf --seed 123
```

- For usage help:

```sh
$ riceclf --help
```
