# Asynchronous Advantage Actor-Critic (A3C)

## Overview

[Advantage Actor-Critic](https://arxiv.org/pdf/1602.01783.pdf) proposes two distributed model-free on-policy RL algorithms, A3C and A2C.
These algorithms are distributed versions of the vanilla Policy Gradient (PG) algorithm with different distributed execution patterns.
The paper suggests accelerating training via scaling data collection, i.e. introducing worker nodes,
which carry copies of the central node's policy network and collect data from the environment in parallel.
This data is used on each worker to compute gradients. The central node applies each of these gradients and then sends updated weights back to the workers.

In A3C, the worker nodes generate data asynchronously and compute gradients from the data. These
computed gradients are then sent to the central node.
Note that the workers in A3C may be slightly out-of-sync with the central node due to asynchrony,
which may induce biases in learning (on-policy loss function).


## Documentation & Implementation of A3C:

**[Detailed Documentation](https://docs.ray.io/en/master/rllib-algorithms.html#a3c)**

**[Implementation](https://github.com/ray-project/ray/blob/master/rllib/algorithms/a3c/a3c.py)**
