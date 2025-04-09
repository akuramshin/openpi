from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

import pickle, time
import copy

config = config.get_config("pi0_fast_libero")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_libero")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Load the policy input from the pickle file
with open('policy_input.pkl', 'rb') as f:
    policy_input = pickle.load(f)

# Run once to jit compile
# single_outputs = policy.infer(policy_input)
batch_outputs = policy.infer_k_action_chunks_and_logprobs([policy_input, policy_input], k=5, temperature=0.0)

timings = []
for i in range(10):
    start_time = time.time()
    batch_outputs = policy.infer_k_action_chunks_and_logprobs([policy_input, policy_input], k=5, temperature=0.0)
    end_time = time.time()
    timings.append(end_time - start_time)
    print(i)

print(f"Average time: {sum(timings) / len(timings)}")

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