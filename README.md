# MAML (Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks)

[MAML](https://arxiv.org/abs/1703.03400) is an on-policy meta RL algorithm. Unlike standard RL algorithms, which aim to maximize the sum of rewards into the future for a single task (e.g. HalfCheetah), meta RL algorithms seek to maximize the sum of rewards for *a given distribution of tasks*. 

On a high level, MAML seeks to learn quick adaptation across different tasks (e.g. different velocities for HalfCheetah). Quick adaptation is defined by the number of gradient steps it takes to adapt. MAML aims to maximize the RL objective for each task after `X` gradient steps. Doing this requires partitioning the algorithm into two steps. The first step is data collection. This involves collecting data for each task for each step of adaptation (from `1, 2, ..., X`). The second step is the meta-update step. This second step takes all the aggregated ddata from the first step and computes the meta-gradient. 

Code here is adapted from https://github.com/jonasrothfuss, which outperforms vanilla MAML and avoids computation of the higher order gradients during the meta-update step. MAML is evaluated on custom environments that are described in greater detail here.

MAML uses additional metrics to measure performance; episode_reward_mean measures the agent’s returns before adaptation, episode_reward_mean_adapt_N measures the agent’s returns after N gradient steps of inner adaptation, and adaptation_delta measures the difference in performance before and after adaptation.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Maintenance](#maintenance)
- [Getting Involved](#getting-involved)

## Installation

```
conda create -n rllib-maml python=3.10
conda activate rllib-maml
python -m pip install ray==2.3.1
python -m pip install -e '.[development]'
pre-commit install
```

## Usage

Instructions on how to use the project. Include any code examples or screenshots if necessary.

## Maintenance

The RLlib maintainers team at Anyscale has made the decision to deprecate some less commonly used or more experimental algorithms in RLlib. While these algorithms will no longer have active support from the RLlib team, we are accepting contributions related to bug fixes, quality improvements, and new features that can potentially help increase the algorithm' ussage (e.g. adding support for new types of action or observation spaces.) This is to reduce the maintenance burden on the team and to focus on the most commonly used algorithms. If we find that there is a strong demand for this
algorithm, we will consider re-adding it back to RLlib in the future. Please let us know if you have any questions or concerns on
the [Ray Discuss forum](discuss.ray.io).

## Getting Involved

Getting Involved
----------------

.. list-table::
   :widths: 25 50 25 25
   :header-rows: 1

   * - Platform
     - Purpose
     - Estimated Response Time
     - Support Level
   * - `Discourse Forum`_
     - For discussions about development and questions about usage.
     - < 1 week
     - Community
   * - `GitHub Issues`_
     - For reporting bugs and filing feature requests.
     - N/A
     - Community
   * - `Slack`_
     - For collaborating with other Ray users.
     - < 2 days
     - Community