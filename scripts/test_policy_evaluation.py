from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

import pickle, time
import copy

import numpy as np

# # Pi0Fast Libero
# config = config.get_config("pi0_fast_libero")
# checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_libero")

# # Pi0Fast Base
# config = config.get_config("pi0_fast_libero")
# checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_base")

# Pi0 Libero
config = config.get_config("pi0_libero")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")

# Pi0 Base

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Load the policy input from the pickle file
with open('policy_input.pkl', 'rb') as f:
    policy_input = pickle.load(f)

# Run once to jit compile
single_outputs = policy.infer(policy_input)
batch_outputs = policy.infer_k_action_chunks_and_logprobs([policy_input], k=5, temperature=1.0, logprob_calc_temp = 1.0)

timings_default_inference = []
timings_k_inference = []
max_action_diff = []
for i in range(10):
    start_time = time.time()
    single_outputs = policy.infer(policy_input)
    end_time = time.time()
    timings_default_inference.append(end_time - start_time)

    start_time = time.time()
    batch_outputs = policy.infer_k_action_chunks_and_logprobs([policy_input], k=1, temperature=0.0)
    end_time = time.time()
    timings_k_inference.append(end_time - start_time)
    print(i)

    max_diff = np.max(np.abs(single_outputs['actions'] - batch_outputs['actions'][0,0,:,:]))
    max_action_diff.append(max_diff)

print(f"default timings: {timings_default_inference}")
print(f"k timings: {timings_k_inference}")

print(f"Average time for default inference: {sum(timings_default_inference) / len(timings_default_inference)}")
print(f"Average time for k inference: {sum(timings_k_inference) / len(timings_k_inference)}")
print(f"Max action difference: {max(max_action_diff)}")

# k_list = [2, 5, 10, 20, 32]
# for k in k_list:

#     start_time = time.time()
#     action_chunks_sequential = []
#     for i in range(k):
#         single_outputs = policy.infer(policy_input)
#         action_chunks_sequential.append(single_outputs['actions'])
#     end_time = time.time()

#     print(f"Sequential time with k={k}: {end_time - start_time}")

#     start_time = time.time()
#     batch_outputs = policy.infer_k_action_chunks_and_logits([policy_input], k=k, temperature=0.0)
#     end_time = time.time()
#     print(f"Batch time with k={k}: {end_time - start_time}")


    # # batch_in = [copy.deepcopy(policy_input) for _ in range(k)]
    # # # batch_in = [policy_input] * k
    # # start_time = time.time()
    # # action_chunks_parallel, inputs_batched = policy.infer_batch(batch_in)
    # # end_time = time.time()
    # # action_chunks_parallel = [action_chunks_parallel[i]['actions'] for i in range(k)]

    # print(f"Parallel time with k={k}: {end_time - start_time}")