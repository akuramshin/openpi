from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        # self._sample_k_action_chunks_and_logits = nnx_utils.module_jit(model.sample_k_action_chunks_and_logits) 
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        # check if the model has sample_k_action_chunks_and_logits
        if not hasattr(model, "sample_k_action_chunks_and_logits"):
            self._sample_k_action_chunks_and_logits = None
        else:
            self._sample_k_action_chunks_and_logits = nnx_utils.module_jit(model.sample_k_action_chunks_and_logits)

        self.build_probability_calculator()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        tokens = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)
        outputs = {
            "state": inputs["state"],
            "actions": tokens,
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs) #, {'actions': tokens, 'state': inputs["state"]}
    
    def infer_k_action_chunks_and_logprobs(
            self, 
            observations: list[dict], 
            k: int = 1, 
            temperature: float = 0.0,
            logprob_calc_temp = 1.0,
        ):
        """
        Run inference on a batch of observations.
        For each observation, sample k action chunks and the sequences of logits for each sampled action chunk.
        """
        # 1. Transform each observation individually (like in training)
        transformed_inputs = []
        for obs in observations:
            # Make a copy and transform (same as in single infer)
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self._input_transform(inputs)
            transformed_inputs.append(inputs)
            
        # 2. Stack into batch (like the collate_fn in training)
        batched_inputs = jax.tree.map(
            lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), 
            *transformed_inputs
        )

        # 3. Run model inference on batch
        self._rng, rng = jax.random.split(self._rng)
        sample_rngs = jax.random.split(rng, k)
        tokens, logits = self._sample_k_action_chunks_and_logits(
            sample_rngs, 
            _model.Observation.from_dict(batched_inputs), 
            temperature=temperature,
        )
        
        # 4. Split batch back into individual examples to apply FAST decoding
        batch_outputs = []
        individual_outputs = []
        batch_size = len(observations)
        for i in range(batch_size):
            for _k in range(k):
                curr_output = {
                    "state": batched_inputs["state"][i],
                    "actions": tokens[i, _k],
                }
                # Transform output (same as in single infer)
                single_output = self._output_transform(curr_output)['actions']
                individual_outputs.append(single_output)
            batch_outputs.append(individual_outputs)
            
        # 5. Get the logprobs for the generated tokens
        logprobs = self.get_probs_from_action_chunks_and_logits(tokens, logits, logprob_calc_temp)

        batch_outputs = {
            'actions' : np.array(batch_outputs),
            'tokens': np.array(tokens, dtype=np.int32),
            # 'logits': np.array(logits, dtype=np.float32),
            'logprobs': np.array(logprobs, dtype=np.float32),
        }

        return batch_outputs
    
    def get_probs_from_action_chunks_and_logits(
            self, 
            tokens: jnp.ndarray,
            logits: jnp.ndarray,
            temperature: float = 1.0
        ):
        """
        Get the probabilities of the action chunks from the corresponding sequences of logits.

        Args:
            tokens: jnp.ndarray, shape (batch_size, k, max_decoding_steps)
            logits: jnp.ndarray, shape (batch_size, k, max_decoding_steps, vocab_size)
            temperature: float, temperature for the softmax

        Returns:
            jnp.ndarray, shape (batch_size, k, max_decoding_steps)
        """
        batch_size = tokens.shape[0]
        num_samples = tokens.shape[1]
        
        if temperature == 0.0:
            # Return probabilities of 1 for all selected tokens
            chunk_logprobs = jnp.zeros((batch_size, num_samples))
        else:
            logits = logits / temperature
            logprobs = jax.nn.log_softmax(logits, axis=-1)
    
            chunk_logprobs = self._compute_logprobs_for_token_chunks(tokens, logprobs)

        return chunk_logprobs

    def infer_batch(self, observations: list[dict]) -> list[dict]:
        """
        Run inference on a batch of observations.
        
        Args:
            observations: List of observation dictionaries
            
        Returns:
            List of action dictionaries
        """
        # 1. Transform each observation individually (like in training)
        transformed_inputs = []
        for obs in observations:
            # Make a copy and transform (same as in single infer)
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self._input_transform(inputs)
            transformed_inputs.append(inputs)
            
        # 2. Stack into batch (like the collate_fn in training)
        batched_inputs = jax.tree.map(
            lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), 
            *transformed_inputs
        )
        
        # 3. Run model inference on batch
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": batched_inputs["state"],
            "actions": self._sample_actions(
                sample_rng, 
                _model.Observation.from_dict(batched_inputs), 
                **self._sample_kwargs
            ),
        }
        
        # 4. Split batch back into individual examples
        individual_outputs = []
        batch_size = len(observations)
        for i in range(batch_size):
            single_output = jax.tree.map(
                lambda x: np.asarray(x[i]),
                outputs
            )
            # Transform output (same as in single infer)
            single_output = self._output_transform(single_output)
            individual_outputs.append(single_output)
            
        return individual_outputs

    def build_probability_calculator(self):
        def get_logprobs_for_action_chunk(token_chunk, logprobs):
            """
            Get the log probability for a single action chunk.

            Args:
                action_chunk: jnp.ndarray, shape (max_decoding_steps,)
                logprobs: jnp.ndarray, shape (max_decoding_steps, vocab_size)
            Returns:
                float, log probability of the action chunk
            """
            # Get the indeces where the action chunk is non-zero
            indices = jnp.where(token_chunk != 0, size=token_chunk.shape[0])[0]
            vals = logprobs[indices, token_chunk[indices]]
            return jnp.sum(vals)

        vmapped_over_batch = jax.vmap(get_logprobs_for_action_chunk, in_axes=(0, 0))
        vmapped_over_samples = jax.vmap(vmapped_over_batch, in_axes=(0, 0))
        compute_logprobs_for_token_chunk = jax.jit(vmapped_over_samples)
        self._compute_logprobs_for_token_chunks = compute_logprobs_for_token_chunk

class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
