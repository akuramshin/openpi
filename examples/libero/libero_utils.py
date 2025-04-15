import math
import numpy as np
import pathlib
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def get_imgs_from_obs(obs, img_resize_size):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, img_resize_size, img_resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, img_resize_size, img_resize_size)
    )
    return img, wrist_img


def construct_policy_input(obs, task_description, img_resize_size=224):
    """
    Construct a policy input element from a LIBERO observation and task description.

    Inputs
    ------
    obs: dict
        A LIBERO observation.
    task_description: str
        The task description.
    img_resize_size: int
        The size to resize the images to.

    Returns
    -------
    element: dict
        A policy input element.
    """
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, img_resize_size, img_resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, img_resize_size, img_resize_size)
    )

    element  = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(task_description),
        }
    return element

def _get_libero_env(
        task,
        resolution : int = LIBERO_ENV_RESOLUTION, 
        seed : int = 42,
    ):
    """
    Initializes and returns the LIBERO environment, along with the task description.
    """
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def build_libero_env_and_task_suite(
        task_suite_name, 
        task_id, 
        resolution : int = LIBERO_ENV_RESOLUTION, 
        seed : int = 42,
        video_out_path : str = 'data/libero/videos',
    ):
    """
    Initializes and returns the LIBERO environment, along with the task suite, task, and number of tasks in suite.

    Args:
        task_suite_name: Name of the task suite to use.
        task_id: ID of the task to use.
        resolution: Resolution of the images.
        seed: Seed for the environment.
        video_out_path: Path to save the videos.

    Returns:
        env: LIBERO environment.
        task_description: Description of the task.
        task: Task to use.
        task_suite: Task suite to use.
        num_tasks_in_suite: Number of tasks in the suite.
        max_steps: Maximum number of steps in the task.
    """
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    if task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    
    task = task_suite.get_task(task_id)

    env, task_description = _get_libero_env(task, resolution, seed)
    
    return env, task_description, task, task_suite, num_tasks_in_suite, max_steps

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def simulate_action_chunk_in_env(env, state, action_chunk, steps_to_simulate=10):
    """
    Simulate the action chunk in the environment to generate a new node.
    """
    
    obs = env.set_init_state(state)

    total_reward = 0
    for action in action_chunk[:steps_to_simulate, :]:
        obs, reward, done, info = env.step(action.tolist())
        total_reward += reward
        if done:
            break
    new_state = env.get_sim_state()

    return new_state, obs, total_reward