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
        ):
        self.args = args
        self.client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        self.env_copy = env_copy
        self.task_description = task_description
        
    def beam_search(self, initial_state, num_expansions=10, beam_width=3, search_depth=5):
        """
        Perform beam search on the Libero environment.
        
        Args:
            initial_state: Initial state to start the search from
            beam_width: Number of trajectories to maintain at each step
            search_depth: Depth of the search tree
        
        Returns:
            best_actions: List of actions for the best trajectory
            best_reward: Reward of the best trajectory
        """

        # Initialize beam with the initial state
        beam = [(initial_state, [], 0)]  # (state, actions_so_far, cumulative_reward)
        
        for depth in range(search_depth):
            # Collect all candidate next states
            pass
    
    def expand_node(self, node, num_expansions=10, temperature=1.0, steps_to_simulate=10):
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
            'k': num_expansions,
            'temperature': temperature,
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
            new_state, new_obs, new_reward = simulate_action_chunk_in_env(
                self.env_copy, 
                node.state, 
                action_chunk,
                steps_to_simulate=steps_to_simulate,
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

    # def simulate_action_chunk(self, node, action_chunk, chunk_logprob, steps_to_simulate=10):
    #     """
    #     Simulate the action chunk in the environment to generate a new node.
    #     """
    #     state = node.state
    #     obs = self.env_copy.set_init_state(state)

    #     total_reward = 0
    #     for action in action_chunk[:steps_to_simulate, :]:
    #         obs, reward, done, info = self.env_copy.step(action.tolist())
    #         total_reward += reward
    #     new_state = self.env_copy.get_sim_state()

    #     new_node = BeamSearchNode(
    #             new_state, 
    #             obs, 
    #             node.node_reward + total_reward, 
    #             node.node_logprob + chunk_logprob, 
    #             parent=node
    #         )
    #     return new_node

    def prune_children(self, node, num_children=3):
        """
        Prune the children of the node based on the reward.
        """
        
        sorted_children = sorted(node.children.items(), key=lambda x: x[1].node_logprob, reverse=True)[:num_children]
        sorted_action_chunks = [node.sampled_action_chunks[key] for key, child in sorted_children]
        sorted_chunk_logprobs = [node.sampled_chunk_logprobs[key] for key, child in sorted_children]

        return sorted_action_chunks, sorted_chunk_logprobs
        return sorted(node.children.items(), key=lambda x: x[1].node_logprob, reverse=True)[:num_children]

    def rank_node_list(self, node_list):
        """
        Rank the node list based on the reward.
        """
        return sorted(node_list, key=lambda x: x.score, reverse=True)
    

def beam_search_agent(
        args, 
        initial_state, 
        env, 
        env_copy, 
        client,
        beam_width=3, 
        search_depth=5,
    ):
    """
    A simple beam search agent for the Libero environment.
    
    Args:
        args: Environment arguments
        initial_state: Initial state to start the search from
        env: Main environment instance
        env_copy: Copy of the environment for rollouts
        client: WebSocket client for action prediction
        beam_width: Number of trajectories to maintain at each step
        search_depth: Depth of the search tree
        
    Returns:
        best_actions: List of actions for the best trajectory
        best_reward: Reward of the best trajectory
    """
    # Initialize beam with the initial state
    beam = [(initial_state, [], 0)]  # (state, actions_so_far, cumulative_reward)
    
    for depth in range(search_depth):
        # Collect all candidate next states
        candidates = []
        
        for state, actions_so_far, reward_so_far in beam:
            # Reset copy environment to the current state
            obs = env_copy.set_initial_state(state)
            
            # Construct policy input
            policy_input = construct_policy_input(obs, args.task_description, args.resize_size)
            
            # Get k different action chunks from the policy
            input_dict = {
                'obs': [policy_input],
                'k': 5,  # Generate 5 different action chunks to explore
                'temperature': 1.0,  # Higher temperature for more diversity
            }
            
            # Get action chunks from the policy
            output = client.infer(input_dict)
            action_chunks = output.get('actions', [])
            
            # For each action chunk, rollout the environment
            for i, action_chunk in enumerate(action_chunks):
                # Reset to the current state
                env_copy.set_state(state)
                
                # Initialize variables for this rollout
                rollout_reward = 0
                rollout_actions = list(actions_so_far)  # Copy the actions so far
                rollout_states = []
                done = False
                
                # Execute the action chunk
                for action in action_chunk:
                    # Step the environment
                    obs, reward, done, info = env_copy.step(action)
                    
                    # Store the action
                    rollout_actions.append(action)
                    
                    # Get the current state
                    current_state = env_copy.sim.get_state().flatten()
                    rollout_states.append(current_state)
                    
                    # Accumulate reward
                    rollout_reward += reward
                    
                    # Check if the episode is done
                    if done:
                        break
                
                # Add the rollout to candidates
                if rollout_states:  # Make sure we have at least one state
                    final_state = rollout_states[-1]
                    candidates.append((final_state, rollout_actions, reward_so_far + rollout_reward))
        
        # If we have no candidates, break
        if not candidates:
            break
        
        # Sort candidates by cumulative reward (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Keep only the top beam_width candidates
        beam = candidates[:beam_width]
    
    # Return the best trajectory found
    if beam:
        best_state, best_actions, best_reward = beam[0]
        return best_actions, best_reward
    else:
        return [], 0

if __name__ == "__main__":
    
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
    
    beam_search_agent = BeamSearchAgent(args, env_copy, task_description)

    init_states = task_suite.get_task_init_states(args.task_id)
    obs = env.set_init_state(init_states[0])

    node = BeamSearchNode(
        state=init_states[0],
        obs=obs,
        node_reward=0,
        node_logprob=0,
        parent=None,
    )

    beam_search_agent.expand_node(
        node, 
        num_expansions=10, 
        temperature=1.0, 
        steps_to_simulate=10
    )

    for key, child in node.children.items():
        img, wrist_img = get_imgs_from_obs(child.obs, args.resize_size)
        Image.fromarray(img).save(f"{args.video_out_path}/img_child_{key}.png")
        Image.fromarray(wrist_img).save(f"{args.video_out_path}/wrist_img_child_{key}.png")

    print(node.children)
    pruned_children = beam_search_agent.prune_children(node, num_children=3)

    # policy_input = construct_policy_input(
    #         obs, 
    #         task_description, 
    #         args.resize_size
    #     )
    # input_dict = {
    #     'obs': [policy_input],
    #     'k': 10,
    #     'temperature': 1.0,
    # }
    # output = beam_search_agent.client.infer(input_dict)
    # action_chunks = output.get('actions', [])
    # logprobs = output.get('logprobs', [])

    # action_chunk = action_chunks[0, 0]
    # logprob = logprobs[0, 0]

    # node = BeamSearchNode(
    #     state=init_states[0],
    #     obs=obs,
    #     node_reward=0,
    #     node_logprob=0,
    #     parent=None,
    # )

    # new_state, new_obs, new_reward = simulate_action_chunk_in_env(env_copy, init_states[0], action_chunk)
    # new_state2, new_obs2, new_reward2 = simulate_action_chunk_in_env(env_copy, init_states[0], action_chunk)

    # img, wrist_img = get_imgs_from_obs(new_obs, args.resize_size)
    # img2, wrist_img2 = get_imgs_from_obs(new_obs2, args.resize_size)

    # Image.fromarray(img).save(f"{args.video_out_path}/img.png")
    # Image.fromarray(wrist_img).save(f"{args.video_out_path}/wrist_img.png")
    # Image.fromarray(img2).save(f"{args.video_out_path}/img2.png")
    # Image.fromarray(wrist_img2).save(f"{args.video_out_path}/wrist_img2.png")
    
    # print(new_state == new_state2)