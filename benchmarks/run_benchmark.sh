CUDA_VISIBLE_DEVICES=7 python benchmark_throughput.py --backend vllm \
--dataset /home/prutko_a/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--model /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b-awq  \
--num-prompts 1000 -q awq --tokenizer hf-internal-testing/llama-tokenizer

CUDA_VISIBLE_DEVICES=7 python benchmark_throughput.py --backend vllm \
--dataset /home/prutko_a/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--model /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b  \
--num-prompts 1000 --tokenizer hf-internal-testing/llama-tokenizer

CUDA_VISIBLE_DEVICES=7 python benchmark_throughput.py --backend hf \
--dataset /home/prutko_a/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--model /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b \
--num-prompts 100 --awq /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/quant_cache/llama-7b-w4-g128-awq.pt \
--hf-max-batch-size 1 --tokenizer hf-internal-testing/llama-tokenizer

CUDA_VISIBLE_DEVICES=7 python benchmark_throughput.py --backend hf \
--dataset /home/prutko_a/data/ShareGPT_V3_unfiltered_cleaned_split.json \
--model /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b \
--num-prompts 100 --hf-max-batch-size 1 \
--tokenizer hf-internal-testing/llama-tokenizer

CUDA_VISIBLE_DEVICES=7 python benchmark_latency.py  \
--model /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b \
--batch-size 8 --input-len 256  \
--tokenizer hf-internal-testing/llama-tokenizer

CUDA_VISIBLE_DEVICES=7 python benchmark_latency.py  \
--model /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b-awq \
--batch-size 8 --input-len 256  -q awq\
--tokenizer hf-internal-testing/llama-tokenizer

CUDA_VISIBLE_DEVICES=7 python benchmark_latency.py  \
--model /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b \
--batch-size 8 --input-len 256  --hf true \
--tokenizer hf-internal-testing/llama-tokenizer

CUDA_VISIBLE_DEVICES=7 python benchmark_latency.py  \
--model /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/llama-7b \
--batch-size 8 --input-len 256  --hf true -q awq \
--awq /data/shared/CompressaAI/LLaMA/llama-1_2-7_13b/quant_cache/llama-7b-w4-g128-awq.pt \
--tokenizer hf-internal-testing/llama-tokenizer