# Rotary-DLT

A resource arbitration system for deep learning training jobs, build on top of TensorFlow.

## Prerequisite

+ Python: 3.7
+ TensorFlow: 1.15

Other dependencies can be found in `Pipfile`

## Rotary-DLT Installation

Rotary-DLT assumes that we have the TensorFlow installed and takes TensorFlow as the computation engine. 

It is recommended to use a Python virtual environment to install Rotary-DLT and any dependency libraries.

Run the following command to install Rotary-DLT, or just run the script `build-install-rotary.sh`

```bash
python setup.py egg_info bdist_wheel
pip3 install dist/rotary-1.0-py3-none-any.whl
```

## Source Code 

1. `demo_launch.py` can be used to reproduce the evaluation results in the paper. Simply running `python demo_launch.py -s rotary` will trigger our Rotary-DLT on a deep lerning training workload, similarly `python demo_launch.py -s <laf,bcf,srf>` will run other baselines on the same workload.

2. `rotary` maintains the core source code of Rotary-DLT. Within it, `models` store all the deep learning models we implemented from scratch, `sched` contains our implementation of our Rotary-DLT resource arbitration mechanism and other baselines we used in the paper. Under the `profiler` directory, measuring the training time and accuracy for CV and NLP models 

3. `scripts/profile-model-accuracy.sh` can be used to create a repository that consist of some historical deep learning training job. This repsoitory can be exploited by training epoch estimator (TEE) and training memory estimator (TME) in Rotary-DLT to estimate training epoch and memory for deep learning training jobs.  

## Survey Report

The survey we mentioned in the paper has been include in `survey_report.pdf`.

