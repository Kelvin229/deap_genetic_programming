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

3. Run the rice classification experiment.
- We can run the experiment and plot the results using the following commands:
```sh
$ riceclf experiment -o tmp --seed 123 456 789 2974 2479 24755 74593 57993 24749 279
```
```sh
$ riceclf plot tmp/result.csv tmp/result.png
```
Now give it some time for each (alot of time, go get a coffee or something... and come back to the program), to process it until its all done, then you can check the results in the riceclf_output folder at the root of the project.

4. For an individual run, create a folder `riceclf_output` and then do:
```sh
$ mkdir riceclf_output
```
- Then run the following command to run the experiment and save the results to a file `riceclf_output/run123.csv`:

```sh
$ riceclf run -o riceclf_output/run123.csv --seed 123
```
- Then to plot the results, run the following command:

```sh
$ riceclf plot riceclf_output/run123.csv riceclf_output/run123.png
```

5. For usage help:

```sh
$ riceclf --help
$ riceclf run --help
$ riceclf experiment --help
$ riceclf plot --help
```

## References:
   - F.-A. Fortin, F.-M. De Rainville, M.-A. Gardner, M. Parizeau, and C. Gagné, “DEAP: Evolutionary algorithms made easy,” Journal of Machine Learning Research, vol. 13, pp. 2171–2175, jul 2012. https://deap.readthedocs.io/en/master/
   - The riceclf project uses the Rice Classification dataset, which is a dataset that contains 166 images of rice diseases and 166 images of healthy rice plants. The code documentation is at:
   Cinar, I. and Koklu, M. (2019). Classification of Rice Varieties Using Artificial Intelligence Methods. International Journal of Intelligent Systems and Applications in Engineering, vol.7, no.3 (Sep. 2019), pp.188-194. https://doi.org/10.18201/ijisae.2019355381.
