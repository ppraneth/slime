import random
from typing import List

from slime.rollout.sglang_rollout import sglang_rollout
from slime.utils.types import Sample


def ragged_generate_rollout(args, rollout_id, data_source, evaluation=False) -> List[List[Sample]]:
    """
    A custom rollout function that generates a variable number of samples per prompt.
    This triggers the "ragged rollout" handling in the training pipeline.
    """
    # 1. First, generate a flat list of samples using the standard function.
    # We generate n_samples_per_prompt for each prompt in the batch.
    flat_samples = sglang_rollout(args, rollout_id, data_source, evaluation)

    if not flat_samples:
        return []

    # 2. Now, restructure the flat list into a ragged list (list of lists).
    ragged_samples = []
    samples_consumed = 0

    # We know the original batch had args.rollout_batch_size prompts.
    for i in range(args.rollout_batch_size):
        if samples_consumed >= len(flat_samples):
            break

        # For each prompt, decide randomly how many samples to use (between 1 and n_samples_per_prompt).
        # This creates the variable-length ("ragged") structure.
        num_samples_for_this_prompt = random.randint(1, args.n_samples_per_prompt)

        # Ensure we don't take more samples than are available.
        num_to_take = min(num_samples_for_this_prompt, len(flat_samples) - samples_consumed)

        if num_to_take == 0:
            continue

        # Take the slice of samples for the current prompt.
        start_index = samples_consumed
        end_index = samples_consumed + num_to_take

        group = flat_samples[start_index:end_index]
        ragged_samples.append(group)

        samples_consumed += num_to_take

    print(f"Ragged Rollout: Generated {len(ragged_samples)} groups with sizes {[len(g) for g in ragged_samples]}")
    return ragged_samples
