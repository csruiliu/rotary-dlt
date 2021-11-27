# Resource Arbitration System

A system can arbitrate and orchestrate resource for iterative jobs. 

## Workload

Under the `workload` directory, `workload_generator.py` will generate the metadata of workloads. The `tensorflow_cifar` and `tensorflow_nlp` stores the source code of CV and NLP models, respectively. 

## Models

The `models` directory includes the source code of some widely used CV and NLP models.

## Profiler

Under the `profiler` directory, measuring the training time and accuracy for CV and NLP models 

## Sched

We have various mechanisms, including Highest Accuracy First (HAF), Least Convergence First (LCF), Shortest Runtime First (SRF)



