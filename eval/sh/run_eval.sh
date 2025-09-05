#!/usr/bin/env bash
set -euo pipefail

# 用法 A（老）：sh/eval.sh /.../global_step_80/actor/huggingface
# 用法 B（新）：sh/eval.sh /.../math_grpo_phi_advprm_0.8_sigmiod_lr1e6_0813 80

if [[ $# -eq 2 ]]; then
  BASE="$1"
  STEP="$2"
  MODEL_NAME_OR_PATH="${BASE%/}/global_step_${STEP}/actor/huggingface"
else
  MODEL_NAME_OR_PATH="${1:?need MODEL_NAME_OR_PATH or BASE+STEP}"
  # 尝试从路径里提取 step（global_step_XX）
  if [[ "$MODEL_NAME_OR_PATH" =~ global_step_([0-9]+) ]]; then
    STEP="${BASH_REMATCH[1]}"
  else
    STEP="NA"
  fi
fi

echo "Using MODEL: $MODEL_NAME_OR_PATH  (step=$STEP)"

export CUDA_VISIBLE_DEVICES=0

# 你的采样设置（pass@1：贪心）
if [[ "${MODEL_NAME_OR_PATH,,}" =~ "deepseek" ]]; then
    PROMPT_TYPE="deepseek-distill-cot-ft"
    temperature=0.0
    top_p=1.0
    max_gen_len=32768
else
    PROMPT_TYPE="qwen25-math-cot-ft"
    temperature=0.0 #0.6
    top_p=1.0 #0.95
    max_gen_len=3000
fi

SPLIT="test"
NUM_TEST_SAMPLE=-1
RUN_TAG="verl_grpo_phi_advprm_lookahead_prmreward_rawgain_0904_pass1"
# <<< 关键：在输出目录里带上 step >>>
OUTPUT_DIR="./output_$(basename "${MODEL_NAME_OR_PATH}")_${RUN_TAG}_step${STEP}"
DATA_NAME="gsm8k,math500,minerva_math,olympiadbench,amc23"

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --max_tokens_per_call "$max_gen_len" \
    --temperature "$temperature" \
    --top_p "$top_p" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_names "${DATA_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --seed 42 \
    --n_sampling 1 \
    --start 0 \
    --end -1 \
    --use_vllm

echo "Model CKPT: $MODEL_NAME_OR_PATH"
echo "Outputs -> $OUTPUT_DIR"

'''BASE=/home/vault/b273dd/b273dd15/checkpoints/verl_grpo_phi_advprm_lookahead_prmreward_rawgain_0904
for s in {10..110..10}; do
  sh/run_eval.sh "$BASE" "$s"
done
'''

'''
BASE=/home/vault/b273dd/b273dd15/checkpoints/verl_grpo_phi_advprm_lookahead_prmreward_rawgain_0904/math_grpo_phi_advprm_lookahead_prmreward_rawgain_0904; for s in {10..110..10}; do bash sh/run_eval.sh "$BASE" "$s"; done
'''

'''
LOG=run_eval_$(date +%Y%m%d_%H%M%S).log; nohup bash -c 'BASE=/home/vault/b273dd/b273dd15/checkpoints/verl_grpo_phi_advprm_lookahead_prmreward_rawgain_0904/math_grpo_phi_advprm_lookahead_prmreward_rawgain_0904; for s in $(seq 10 10 110); do bash sh/run_eval.sh "$BASE" "$s"; done' > "$LOG" 2>&1 & tail -f "$LOG"

'''