import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from PIL import Image

from libero_utils import get_imgs_from_obs, construct_policy_input, _get_libero_env

import time

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    task_id: int = 0  # Task ID to run
    episode_id: int = 0  # Episode ID to run

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 42  # Random Seed (for reproducibility)

def test_env_copy_and_reset(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    task = task_suite.get_task(args.task_id)
    task_2 = task_suite.get_task(args.task_id + 1)
    initial_states = task_suite.get_task_init_states(args.task_id)
    initial_states_2 = task_suite.get_task_init_states(args.task_id + 1)
    
    # Initialize LIBERO environment and task description
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    env_copy, task_description_copy = _get_libero_env(task_2, LIBERO_ENV_RESOLUTION, args.seed)

    # Reset environment
    env.reset()
    env_copy.reset()

    # Set initial states
    obs = env.set_init_state(initial_states[0])
    env_state = env.get_sim_state()
    obs_copy = env_copy.set_init_state(initial_states_2[0])

    img, wrist_img = get_imgs_from_obs(obs, args.resize_size)
    img_copy, wrist_img_copy = get_imgs_from_obs(obs_copy, args.resize_size)

    Image.fromarray(img).save(f"{args.video_out_path}/img.png")
    Image.fromarray(wrist_img).save(f"{args.video_out_path}/wrist_img.png")
    Image.fromarray(img_copy).save(f"{args.video_out_path}/img_copy.png")
    Image.fromarray(wrist_img_copy).save(f"{args.video_out_path}/wrist_img_copy.png")

    obs_copy2 = env_copy.regenerate_obs_from_state(env_state)
    # env_copy.set_state(env_state)
    # obs_copy2 = env_copy.env._get_observations()

    img_copy2, wrist_img_copy2 = get_imgs_from_obs(obs_copy2, args.resize_size)

    Image.fromarray(img_copy2).save(f"{args.video_out_path}/img_copy2.png")
    Image.fromarray(wrist_img_copy2).save(f"{args.video_out_path}/wrist_img_copy2.png")


def run_sample_actions_test(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)

    # Initialize LIBERO environment and task description
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    env_copy, task_description_copy = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    # Reset environment
    env.reset()

    # Set initial states
    obs = env.set_init_state(initial_states[2])
    policy_input = construct_policy_input(obs, task_description, args.resize_size)
    input = {
        'obs': [policy_input],
        'k' : 5,
        'temperature' : 1.0,
    }

    # Run once to jit compile
    start_time = time.time()
    out = client.infer(input)
    end_time = time.time()
    print(f"Initial time: {end_time - start_time}")

    # Run 10 times and take the average
    obs = env.set_init_state(initial_states[1])
    policy_input = construct_policy_input(obs, task_description, args.resize_size)
    input = {
        'obs': [policy_input],
        'k' : 5,
        'temperature' : 1.0,
    }

    timing_list = []
    for i in range(10):
        start_time = time.time()
        out = client.infer(input)
        end_time = time.time()
        timing_list.append(end_time - start_time)
        print(f"Time {i}: {end_time - start_time}")
    print(f"Average time: {sum(timing_list) / len(timing_list)}")

    # k_list = [1, 5, 10, 20, 32]
    # for k in k_list:

    #     print(f'Starting k = {k}')     
    #     client.infer(policy_input)
    #     input = {
    #         'obs': [policy_input],
    #         'k' : k,
    #         'temperature' : 1.0,
    #     }
    #     client.infer(input)

    #     start_time = time.time()
    #     action_chunks_sequential = []
    #     for i in range(k):
    #         action_chunks_sequential.append(client.infer(policy_input)['actions'])
    #     end_time = time.time()

    #     print(f"Sequential time with k={k}: {end_time - start_time}")

    #     input = {
    #         'obs': [policy_input],
    #         'k' : k,
    #         'temperature' : 1.0,
    #     }
    #     start_time = time.time()
    #     out = client.infer(input)
    #     end_time = time.time()
    #     print(f"Re-using KV cache time with k={k}: {end_time - start_time}")

    #     # batch_in = [policy_input.copy() for _ in range(k)]
    #     # start_time = time.time()
    #     # action_chunks_parallel = client.infer(batch_in)
    #     # end_time = time.time()
    #     # action_chunks_parallel = [action_chunks_parallel[i]['actions'] for i in range(k)]

    #     # print(f"Parallel time with k={k}: {end_time - start_time}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(run_sample_actions_test)
