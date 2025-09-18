# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the outputs of HF and vLLM when using greedy sampling.

It tests automated prefix caching (APC). APC can be enabled by
enable_prefix_caching=True.

Run `pytest tests/basic_correctness/test_APC_hybrid_models.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.models.utils import check_logprobs_close

if TYPE_CHECKING:
    from ...conftest import HfRunner, VllmRunner

MODELS = [
    "ibm-granite/granite-4.0-tiny-preview",
]


def _get_vllm_runner_params(model, mamba_ssm_cache_dtype, enforce_eager,
                            max_model_len, dtype, tensor_parallel_size):
    return {
        'model_name': model,
        'mamba_ssm_cache_dtype': mamba_ssm_cache_dtype,
        'enable_prefix_caching': False,
        'enforce_eager': enforce_eager,
        'max_model_len': max_model_len,
        'dtype': dtype,
        'tensor_parallel_size': tensor_parallel_size,
        'disable_cascade_attn': True,  ## not verified yet
        'disable_log_stats': False,  ## collect APC stats
        'gpu_memory_utilization': 0.4
    }


def _get_vLLM_output_logprobs(vllm_runner,
                              kwargs,
                              prompts,
                              max_tokens,
                              num_logprobs,
                              num_repetitions=1):
    outs = []
    with vllm_runner(**kwargs) as vllm_model:
        for _ in range(num_repetitions):
            outs.append(
                vllm_model.generate_greedy_logprobs(prompts, max_tokens,
                                                    num_logprobs))

    return outs


def _get_vLLM_output(vllm_runner,
                     kwargs,
                     prompts,
                     max_tokens,
                     num_repetitions=1):
    outs = []
    with vllm_runner(**kwargs) as vllm_model:
        for _ in range(num_repetitions):
            outs.append(vllm_model.generate_greedy(prompts, max_tokens))

    return outs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("n_repetitions", [2])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("mamba_ssm_cache_dtype", ['auto', 'float32'])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("num_logprobs", [5])
def test_single_prompt(
    hf_runner: HfRunner,
    vllm_runner: VllmRunner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    n_repetitions: int,
    enforce_eager: bool,
    mamba_ssm_cache_dtype: str,
    tensor_parallel_size: int,
    num_logprobs: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Checks exact match decode vllm runner with and without prefix caching
    """
    MULTIPLE = 120

    # Common prefix.
    prefix = MULTIPLE * example_prompts[0]

    # Sample prompts.
    generated_prompts = [prefix + prompt for prompt in example_prompts[1:]]

    with monkeypatch.context() as m:
        # Ensure that the testcase is using V1
        m.setenv("VLLM_USE_V1", "1")

        max_model_len = max(
            len(prompt) + max_tokens for prompt in generated_prompts)
        vllm_runner_kwargs = _get_vllm_runner_params(model,
                                                     mamba_ssm_cache_dtype,
                                                     enforce_eager,
                                                     max_model_len, dtype,
                                                     tensor_parallel_size)
        vllm_outputs_no_cache = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, [generated_prompts[0]],
            max_tokens, num_logprobs)[0]

        vllm_runner_kwargs['enable_prefix_caching'] = True
        vllm_outputs_cache_rep = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, [generated_prompts[0]],
            max_tokens, num_logprobs, n_repetitions)

        for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
            # In the first repetition, the caches are filled
            # In the second repetition, these caches are reused

            check_logprobs_close(
                outputs_0_lst=vllm_outputs_no_cache,
                outputs_1_lst=vllm_outputs_cache_itn,
                name_0="vllm_no_cache",
                name_1=f"vllm_cache_it_{r_idx + 1}",
            )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("n_repetitions", [2])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("mamba_ssm_cache_dtype", ['auto', 'float32'])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("num_logprobs", [5])
def test_single_prompt_mamba_size_alignment(
    hf_runner: HfRunner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    n_repetitions: int,
    enforce_eager: bool,
    mamba_ssm_cache_dtype: str,
    tensor_parallel_size: int,
    num_logprobs: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Checks exact match decode vllm runner with and without prefix caching
    """
    MULTIPLE = 120

    # Common prefix.
    prefix = MULTIPLE * example_prompts[0]

    # Sample prompts.
    generated_prompts = [prefix + prompt for prompt in example_prompts[1:]]

    with monkeypatch.context() as m:
        # Ensure that the testcase is using V1
        m.setenv("VLLM_USE_V1", "1")

        max_model_len = max(
            len(prompt) + max_tokens for prompt in generated_prompts)
        vllm_runner_kwargs = _get_vllm_runner_params(model,
                                                     mamba_ssm_cache_dtype,
                                                     enforce_eager,
                                                     max_model_len, dtype,
                                                     tensor_parallel_size)
        vllm_outputs_no_cache = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, [generated_prompts[0]],
            max_tokens, num_logprobs)[0]

        vllm_runner_kwargs['enable_prefix_caching'] = True
        with vllm_runner(**vllm_runner_kwargs) as vllm_model:
            # Retrieve mamba state block size
            mamba_block_size = vllm_model.llm.llm_engine.cache_config. \
                mamba_block_size

        vllm_outputs_cache_rep = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, [generated_prompts[0]],
            max_tokens, num_logprobs, n_repetitions)

        for multiple in [1, 3, 7]:
            for offsets in [
                    3, mamba_block_size // 4 - 3, mamba_block_size // 4,
                    mamba_block_size // 4 + 3, mamba_block_size // 2 - 3,
                    mamba_block_size // 2, mamba_block_size // 2 + 3,
                    mamba_block_size - 3
            ]:

                vllm_runner_kwargs[
                    'max_num_batched_tokens'] = multiple * mamba_block_size - \
                                                offsets
                vllm_outputs_cache_rep = _get_vLLM_output_logprobs(
                    vllm_runner, vllm_runner_kwargs, [generated_prompts[0]],
                    max_tokens, num_logprobs, n_repetitions)

                # Check alignment of the output logits when using APC
                for r_idx, vllm_outputs_cache_itn in enumerate(
                        vllm_outputs_cache_rep):
                    # In the first repetition, the caches are filled
                    # In the second repetition, these caches are reused

                    check_logprobs_close(
                        outputs_0_lst=vllm_outputs_no_cache,
                        outputs_1_lst=vllm_outputs_cache_itn,
                        name_0="vllm_no_cache",
                        name_1=f"vllm_cache_it_{r_idx + 1}",
                    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("n_repetitions", [2])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("mamba_ssm_cache_dtype", ['auto', 'float32'])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("num_logprobs", [5])
def test_multiple_prompts_all_cached_output_logprobs(
    hf_runner: HfRunner,
    vllm_runner: VllmRunner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    n_repetitions: int,
    enforce_eager: bool,
    mamba_ssm_cache_dtype: str,
    tensor_parallel_size: int,
    num_logprobs: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Checks exact match decode vllm runner with and without prefix caching
    """
    MULTIPLE = 120

    # Common prefix.
    prefix = MULTIPLE * example_prompts[0]

    # Sample prompts.
    generated_prompts = [prefix + prompt for prompt in example_prompts[1:]]

    with monkeypatch.context() as m:
        # Ensure that the testcase is using V1
        m.setenv("VLLM_USE_V1", "1")

        max_model_len = max(
            len(prompt) + max_tokens for prompt in generated_prompts)
        vllm_runner_kwargs = _get_vllm_runner_params(model,
                                                     mamba_ssm_cache_dtype,
                                                     enforce_eager,
                                                     max_model_len, dtype,
                                                     tensor_parallel_size)
        vllm_outputs_no_cache = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens,
            num_logprobs)[0]

        vllm_runner_kwargs['enable_prefix_caching'] = True
        vllm_outputs_cache_rep = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens,
            num_logprobs, n_repetitions)

        for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
            # In the first repetition, the caches are filled
            # In the second repetition, these caches are reused

            check_logprobs_close(
                outputs_0_lst=vllm_outputs_no_cache,
                outputs_1_lst=vllm_outputs_cache_itn,
                name_0="vllm_no_cache",
                name_1=f"vllm_cache_it_{r_idx + 1}",
            )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("n_repetitions", [2])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("mamba_ssm_cache_dtype", ['auto', 'float32'])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("num_logprobs", [5])
def test_multiple_prompts_partial_cached_output_logprobs(
    hf_runner: HfRunner,
    vllm_runner: VllmRunner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    n_repetitions: int,
    enforce_eager: bool,
    mamba_ssm_cache_dtype: str,
    tensor_parallel_size: int,
    num_logprobs: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Checks exact match decode vllm runner with and without prefix caching
    """
    MULTIPLE = 120

    # Common prefix.
    prefix = MULTIPLE * example_prompts[0]

    # Sample prompts.
    generated_prompts = [prefix + prompt for prompt in example_prompts[1:]]

    with monkeypatch.context() as m:
        # Ensure that the testcase is using V1
        m.setenv("VLLM_USE_V1", "1")

        max_model_len = max(
            len(prompt) + max_tokens for prompt in generated_prompts)
        vllm_runner_kwargs = _get_vllm_runner_params(model,
                                                     mamba_ssm_cache_dtype,
                                                     enforce_eager,
                                                     max_model_len, dtype,
                                                     tensor_parallel_size)
        vllm_outputs_no_cache = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens,
            num_logprobs)[0]

        # Cache only part of all the prompts
        vllm_runner_kwargs['enable_prefix_caching'] = True
        vllm_outputs_partial_cache = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, generated_prompts[:3], max_tokens,
            num_logprobs)[0]

        check_logprobs_close(
            outputs_0_lst=vllm_outputs_no_cache[:3],
            outputs_1_lst=vllm_outputs_partial_cache,
            name_0="vllm_no_cache",
            name_1="vllm_partial_cache",
        )

        vllm_outputs_cache_rep = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens,
            num_logprobs, n_repetitions)

        for r_idx, vllm_outputs_cache_itn in enumerate(vllm_outputs_cache_rep):
            # In the first repetition, the caches are filled
            # In the second repetition, these caches are reused

            check_logprobs_close(
                outputs_0_lst=vllm_outputs_no_cache,
                outputs_1_lst=vllm_outputs_cache_itn,
                name_0="vllm_no_cache",
                name_1=f"vllm_cache_it_{r_idx + 1}",
            )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("n_repetitions", [2])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("mamba_ssm_cache_dtype", ['auto', 'float32'])
# NOTE: Increasing this in this suite will fail CI because we currently cannot
# reset distributed env properly. Use a value > 1 just when you test.
@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("num_logprobs", [5])
def test_specific_prompts_output_logprobs(
    hf_runner: HfRunner,
    vllm_runner: VllmRunner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    n_repetitions: int,
    enforce_eager: bool,
    mamba_ssm_cache_dtype: str,
    tensor_parallel_size: int,
    num_logprobs: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Checks exact match decode vllm runner with and without prefix caching
    """

    generated_prompts = [
        "Hello, my name is John Smith and I work at " * 100,
        "The president of the United States is " * 200,
        "The capital of France is something like" * 200,
        "The future of AI is " * 300,
    ]

    with monkeypatch.context() as m:
        # Ensure that the testcase is using V1
        m.setenv("VLLM_USE_V1", "1")

        max_model_len = max(
            len(prompt) + max_tokens for prompt in generated_prompts)
        vllm_runner_kwargs = _get_vllm_runner_params(model,
                                                     mamba_ssm_cache_dtype,
                                                     enforce_eager,
                                                     max_model_len, dtype,
                                                     tensor_parallel_size)
        vllm_outputs_logprobs_no_cache = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens,
            num_logprobs)[0]
        # vllm_outputs_no_cache = _get_vLLM_output(
        #   vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens)[0]

        # Cache only part of all the prompts
        vllm_runner_kwargs['enable_prefix_caching'] = True
        vllm_outputs_logprobs_cache_rep = _get_vLLM_output_logprobs(
            vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens,
            num_logprobs, n_repetitions)
        # vllm_outputs_cache_rep = _get_vLLM_output(
        #   vllm_runner, vllm_runner_kwargs, generated_prompts, max_tokens,
        #   n_repetitions)

        # for r_idx, (vllm_outputs_logprobs_cache_itn, vllm_outputs_cache_itn)
        #   in enumerate(zip(
        #       vllm_outputs_logprobs_cache_rep, vllm_outputs_cache_rep)):
        for r_idx, vllm_outputs_logprobs_cache_itn in enumerate(
                vllm_outputs_logprobs_cache_rep):
            # In the first repetition, the caches are filled
            # In the second repetition, these caches are reused

            check_logprobs_close(
                outputs_0_lst=vllm_outputs_logprobs_no_cache,
                outputs_1_lst=vllm_outputs_logprobs_cache_itn,
                name_0="vllm_no_cache",
                name_1=f"vllm_cache_it_{r_idx + 1}",
            )

            # check_outputs_equal(
            #     outputs_0_lst=vllm_outputs_no_cache,
            #     outputs_1_lst=vllm_outputs_cache_itn,
            #     name_0="vllm_no_cache",
            #     name_1=f"vllm_cache_it_{r_idx + 1}",
            # )
