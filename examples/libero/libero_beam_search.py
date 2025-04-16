import collections
import dataclasses
import logging

from datetime import datetime
from pathlib import Path

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import tyro
from PIL import Image

import time

from libero_utils import \
    construct_policy_input, \
    build_libero_env_and_task_suite, \
    get_imgs_from_obs, \
    simulate_action_chunk_in_env


class BeamSearchNode:
    def __init__(
            self, 
            state, 
            obs, 
            node_reward, 
            node_logprob, 
            parent=None
        ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state = state
        self.obs = obs
        self.node_reward = node_reward
        self.node_logprob = node_logprob
        self.parent = parent
        self.children = {}
        self.sampled_action_chunks = {}
        self.sampled_chunk_logprobs = {}
        self.expanded = False
        
class BeamSearchAgent:
    def __init__(
            self, 
            args, 
            env_copy, 
            task_description,
            beam_width : int = 3,
            search_depth : int = 5,
            num_expansions_per_node : int = 10,
            generation_temperature : float = 1.0,
            chunk_steps_to_simulate : int = 10,
        ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.args = args
        self.client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        self.env_copy = env_copy
        self.task_description = task_description

        self.beam_width = beam_width
        self.search_depth = search_depth

        self.num_expansions_per_node = num_expansions_per_node
        self.generation_temperature = generation_temperature
        self.chunk_steps_to_simulate = chunk_steps_to_simulate
        
    def beam_search(
            self, 
            init_state,
            init_obs,
        ):
        """
        Perform beam search on the Libero environment.
        
        Args:
            init_state: Initial state to start the search from.
            init_obs: Initial observation to start the search from.
            beam_width: Number of trajectories to maintain at each step
            search_depth: Depth of the search tree
        
        Returns:
            best_actions: List of actions for the best trajectory
            best_reward: Reward of the best trajectory
        """

        initial_node = BeamSearchNode(
            state=init_state,
            obs=init_obs,
            node_reward=0,
            node_logprob=0,
            parent=None,
        )

        beam_search_steps = 0
        nodes_to_expand = [initial_node]

        self.logger.info("Starting beam search with initial node: {}".format(initial_node))

        while beam_search_steps < self.search_depth:
            self.logger.info("Beam search step: {}".format(beam_search_steps))

            # Expand the nodes in the beam
            nodes_to_expand = self.beam_search_layer_step(nodes_to_expand)

            beam_search_steps += 1

        # Get the best node
        best_node = nodes_to_expand[0]

        # Get the action chunks from the best node
        bs_action_chunks, bs_chunk_logprobs, bs_states, bs_observations = self.get_action_chunks_from_leaf_node(best_node)

        self.logger.info("Best node: {}".format(best_node))

        self.save_observation_list_as_images(
            obs_list=bs_observations,
            out_dir=f"{self.args.video_out_path}/beam_search",
        )

    def get_action_chunks_from_leaf_node(self, node):
        """
        Get the action chunks from the leaf node.
        This is used to get the action chunks from the best node after the beam search is complete.

        Args:
            node: The leaf node to get the action chunks from.
        
        Returns:
            action_chunks: array of action chunks from root down to leaf
            chunk_logprobs: array of logprobs for each action chunk
            states: array of states from root down to leaf
            obs: array of observations from root down to leaf
        """
        action_chunks = []
        chunk_logprobs = []
        states = []
        obs = []

        this_node = node
        states.append(this_node.state)
        obs.append(this_node.obs)

        while this_node.parent is not None:
            # Get the parent of the current node
            parent = this_node.parent

            # Get the key of the best node in the parent's children
            this_node_key = list(parent.children.keys())[list(parent.children.values()).index(this_node)]
            action_chunks.append(parent.sampled_action_chunks[this_node_key])
            chunk_logprobs.append(parent.sampled_chunk_logprobs[this_node_key])

            this_node = parent
            states.append(this_node.state)
            obs.append(this_node.obs)

        # Reverse the lists to get the action chunks from root down to leaf
        action_chunks.reverse()
        chunk_logprobs.reverse()
        states.reverse()
        obs.reverse()

        return action_chunks, chunk_logprobs, states, obs

    def beam_search_layer_step(self, nodes_to_expand):
        """
        Process 1 layer of the beam search.
        Uses the logprob of the entire path to reach each node in order to rank the nodes.

        Args:
            nodes_to_expand: List of nodes to expand
        Returns:
            expanded_node_list: Pruned list of expanded nodes
        """
        expanded_node_list = []
        with logging_redirect_tqdm():
            for node in tqdm.tqdm(nodes_to_expand, desc="Expanding nodes", total=len(nodes_to_expand)):
                start_time = time.time()
                self.expand_node(node)
                end_time = time.time()
                self.logger.info("Node expanded in {} seconds".format(end_time - start_time))

                for k,v in node.children.items():
                    expanded_node_list.append(v)
        
        expanded_node_list.sort(key=lambda x: x.node_logprob, reverse=True)

        return expanded_node_list[:self.beam_width]

    def expand_node(self, node):
        """
        Expand the node by sampling VLA policy for action chunks, and simulating them to generate child nodes.
        """

        # Construct policy input
        observation = node.obs
        policy_input = construct_policy_input(
            observation, 
            self.task_description, 
            self.args.resize_size
        )
        input_dict = {
            'obs': [policy_input],
            'k': self.num_expansions_per_node,
            'temperature': self.generation_temperature,
        }

        # Query policy for action chunks
        output = self.client.infer(input_dict)
        action_chunks = output.get('actions', [])
        logprobs = output.get('logprobs', [])

        num_chunks = action_chunks.shape[1]

        # Simulate action chunks and generate child nodes
        for i in range(num_chunks):
            action_chunk = action_chunks[0, i]
            chunk_logprob = logprobs[0, i]
            self.env_copy.env.timestep = 0 # Hack to prevent the env running out of timesteps
            new_state, new_obs, new_reward = simulate_action_chunk_in_env(
                self.env_copy, 
                node.state, 
                action_chunk,
                steps_to_simulate=self.chunk_steps_to_simulate,
            )
            new_node = BeamSearchNode(
                new_state, 
                new_obs, 
                node.node_reward + new_reward, 
                node.node_logprob + chunk_logprob, 
                parent=node
            )
            node.children[i] = new_node
            node.sampled_action_chunks[i] = action_chunk
            node.sampled_chunk_logprobs[i] = chunk_logprob

        node.expanded = True

    def save_observation_list_as_images(self, obs_list, out_dir):
        """
        Save the observations as images in the output directory.
        """
        for i, obs in enumerate(obs_list):
            img, wrist_img = get_imgs_from_obs(obs, self.args.resize_size)
            Image.fromarray(img).save(f"{out_dir}/img_{i}.png")
            Image.fromarray(wrist_img).save(f"{out_dir}/wrist_img_{i}.png")

if __name__ == "__main__":
    
    # Set up loggin
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"libero_beam_search_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode='w'),
        ],
    )

    @dataclasses.dataclass
    class Args:
        task_suite_name: str = "libero_spatial"
        task_id: int = 0
        episode_id: int = 0
        video_out_path: str = "data/libero/videos"
        seed: int = 42
        resize_size: int = 224
        host: str = "0.0.0.0"
        port: int = 8000

    # Make a structured args object
    args = Args()

    env, task_description, task, task_suite, num_tasks_in_suite, max_steps = \
        build_libero_env_and_task_suite(
            args.task_suite_name, 
            args.task_id, 
            256, 
            args.seed, 
            args.video_out_path
        )
    
    env_copy, task_description_copy, _, _, _, _ = \
        build_libero_env_and_task_suite(
            args.task_suite_name, 
            args.task_id, 
            256, 
            args.seed, 
            args.video_out_path
        )
    
    env.reset()
    env_copy.reset()
    
    beam_search_agent = BeamSearchAgent(
        args, 
        env_copy, 
        task_description,
        beam_width = 3,
        search_depth = 7,
        num_expansions_per_node = 10,
        generation_temperature = 1.0,
        chunk_steps_to_simulate = 10,
    )

    init_states = task_suite.get_task_init_states(args.task_id)
    obs = env.set_init_state(init_states[0])

    beam_search_agent.beam_search(
        init_state=init_states[0],
        init_obs=obs
    )

    # node = BeamSearchNode(
    #     state=init_states[0],
    #     obs=obs,
    #     node_reward=0,
    #     node_logprob=0,
    #     parent=None,
    # )

    # beam_search_agent.expand_node(
    #     node, 
    #     num_expansions=10, 
    #     temperature=1.0, 
    #     steps_to_simulate=10
    # )

    # for key, child in node.children.items():
    #     img, wrist_img = get_imgs_from_obs(child.obs, args.resize_size)
    #     Image.fromarray(img).save(f"{args.video_out_path}/img_child_{key}.png")
    #     Image.fromarray(wrist_img).save(f"{args.video_out_path}/wrist_img_child_{key}.png")
