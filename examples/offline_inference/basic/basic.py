# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
import torch 

# Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# MAX_LEN=512
# Create a sampling params object.
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


prompts = [
    "The president of the United States is",
]
#greedy sampling
#sampling_params = SamplingParams(temperature=0.0, top_p=1.0)
sampling_params = SamplingParams(max_tokens=200, temperature=0.0)

# Create an LLM.
#llm = LLM(model="facebook/opt-125m")
DIR='/block/granite/granite-hybridmoe-7b-a1b-base-pipecleaner-hf'
llm = LLM(model=DIR, gpu_memory_utilization=0.5, max_model_len=512, dtype=torch.bfloat16
          #num_gpu_blocks_override=16 #TODO: Why do we get zero?
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {generated_text!r}")
    print("-" * 60)