import gymnasium as gym
import numpy as np

from rllib_contrib_shared.utils import check


def get_mean_action(alg: "ray.rllib.algorithm.Algorithm", obs: np.array):  # noqa: F821
    """Compute the mean action of an algorithm's policy over 5000 samples."""
    out = []
    for _ in range(5000):
        out.append(float(alg.compute_single_action(obs)))
    return np.mean(out)


def ckpt_restore_test(
    config: "ray.rllib.algorithm_config.AlgorithmConfig",  # noqa: F821
    env_name: str,
    tf2: bool = False,
    object_store: bool = False,
    replay_buffer: bool = False,
):
    """Test that after an algorithm is trained, its checkpoint can be restored.

    Check the replay buffers of the algorithm to see if they have identical data.
    Check the optimizer weights of the policy on the algorithm to see if they're
    identical.

    Args:
        config: The config of the algorithm to be trained.
        env_name: The name of the gymansium environment to be trained on.
        tf2: Whether to test the algorithm with the tf2 framework or not.
        object_store: Whether to test checkpointing with objects from the object store.
        replay_buffer: Whether to test checkpointing with replay buffers.

    """
    # If required, store replay buffer data in checkpoints as well.
    if replay_buffer:
        config = config.training(store_buffer_in_checkpoints=True)

    frameworks = (["tf2"] if tf2 else []) + ["torch", "tf"]
    for fw in frameworks:
        for use_object_store in [False, True] if object_store else [False]:
            print("use_object_store={}".format(use_object_store))
            env = gym.make(env_name)
            alg1 = config.environment(env_name).framework(fw).build()
            alg2 = config.environment(env_name).build()

            policy1 = alg1.get_policy()

            res = alg1.train()
            print("current status: " + str(res))

            # Check optimizer state as well.
            optim_state = policy1.get_state().get("_optimizer_variables")

            if use_object_store:
                checkpoint = alg1.save_to_object()
            else:
                checkpoint = alg1.save()

            # Test if we can restore multiple times (at least twice, assuming failure
            # would mainly stem from improperly reused variables)
            for _ in range(2):
                # Sync the models
                if use_object_store:
                    alg2.restore_from_object(checkpoint)
                else:
                    alg2.restore(checkpoint)

            # Compare optimizer state with re-loaded one.
            if optim_state:
                s2 = alg2.get_policy().get_state().get("_optimizer_variables")
                # Tf -> Compare states 1:1.
                if fw in ["tf2", "tf"]:
                    check(s2, optim_state)
                # For torch, optimizers have state_dicts with keys=params,
                # which are different for the two models (ignore these
                # different keys, but compare all values nevertheless).
                else:
                    for i, s2_ in enumerate(s2):
                        check(
                            list(s2_["state"].values()),
                            list(optim_state[i]["state"].values()),
                        )

            # Compare buffer content with restored one.
            if replay_buffer:
                data = alg1.local_replay_buffer.replay_buffers[
                    "default_policy"
                ]._storage[42 : 42 + 42]
                new_data = alg2.local_replay_buffer.replay_buffers[
                    "default_policy"
                ]._storage[42 : 42 + 42]
                check(data, new_data)

            for _ in range(1):
                obs = env.observation_space.sample()
                a1 = get_mean_action(alg1, obs)
                a2 = get_mean_action(alg2, obs)
                print("Checking computed actions", alg1, obs, a1, a2)
                if abs(a1 - a2) > 0.1:
                    raise AssertionError(
                        "algo={} [a1={} a2={}]".format(str(alg1.__class__), a1, a2)
                    )
            # Stop both algos.
            alg1.stop()
            alg2.stop()
