# NNGP for active learning

This repository showcases application of NNGP approach for active learning. The approach is described in the paper [Deeper Connections between Neural Networks and Gaussian Processes
Speed-up Active Learning by Evgenii Tsymbalov, Sergei Makarychev, Alexander Shapeev and Maxim Panov](http://www.gatsby.ucl.ac.uk/~balaji/udl2019/accepted-papers/UDL2019-paper-53.pdf)

Model task is 10D [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) regression; we start from small training set and then sampling additional data from bigger data pool on each iteration. The goal is to evaluate which samples from the pool will speed-up the training by using uncertainty estimation.

We compare three approaches:
- NNGP (presented approach)
- MCDUE (common approach for uncertainty estimation)
- Random sampling

```shell script
python al_rosenbrock_experiment.py
```

You can tweak some training parameters; to get the list of parameters, read the help

```shell script
python al_rosenbrock_experiment.py --help
```

