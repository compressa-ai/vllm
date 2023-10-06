"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM
from vllm.transformers_utils.tokenizer import get_tokenizer

from transformers import AutoConfig
from awq.quantize.quantizer import real_quantize_model_weight
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map, load_checkpoint_in_model

def main(args: argparse.Namespace):
    if args.tokenizer is None:
        args.tokenizer = args.model
    print(args)

    # Process all the requests in a single batch if possible.
    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    if not args.hf:
        llm = LLM(
            model=args.model,
            tokenizer=args.tokenizer,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=args.batch_size * args.input_len,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
        )
    else:
        tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
        if args.awq is not None:
            config = AutoConfig.from_pretrained(args.model)

            q_group_size = 64 if "g64" in args.awq else 128
            q_config = {"zero_point": not False, "q_group_size": q_group_size}  # TODO
            print("Loading pre-computed quantized weights...")
            with init_empty_weights():
                llm = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16, trust_remote_code=True)
            real_quantize_model_weight(llm, w_bit=4, q_config=q_config, init_only=True)

            llm.tie_weights()

            # Infer device map
            max_memory = f'{40000}MB'
            n_gpus = torch.cuda.device_count()
            max_memory = {i: max_memory for i in range(n_gpus)}
            kwargs = {"max_memory": max_memory} if len(max_memory) else {}
            device_map = infer_auto_device_map(
                llm,
                no_split_module_classes=[
                    "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"],
                **kwargs
            )
            # Load checkpoint in the model 
            load_checkpoint_in_model(
                llm,
                checkpoint=args.awq,
                device_map=device_map,
                offload_state_dict=True,
            )
        else:
            llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=args.trust_remote_code)
        if llm.config.model_type == "llama":
            # To enable padding in the HF backend.
            tokenizer.pad_token = tokenizer.eos_token
        llm = llm.cuda()

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = [[1] * args.input_len] * args.batch_size
    input_ids = torch.Tensor(dummy_prompt_token_ids).to(torch.int).cuda()

    def run_to_completion(profile: bool = False, hf=False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        if not hf:
            llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                         sampling_params=sampling_params,
                         use_tqdm=False)
        else:
            llm_outputs = llm.generate(
                input_ids=input_ids,
                do_sample=not args.use_beam_search,
                num_return_sequences=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=args.output_len,
            )
            # Include the decoding time.
            tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)

        end_time = time.perf_counter()
        latency = end_time - start_time
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("Warming up...")
    run_to_completion(profile=False, hf=args.hf)

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile=False, hf=args.hf))
    print(f'Avg latency: {np.mean(latencies)} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters',
                        type=int,
                        default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--hf', type=bool, default=False)
    parser.add_argument("--awq",
                    type=str,
                    default=None)
    args = parser.parse_args()
    main(args)
