set -ex

# for Qwen2.5 7B Instruct-1M
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# single GPU for others
export CUDA_VISIBLE_DEVICES=0
MODEL_NAME_OR_PATH=$1

if [[ "${MODEL_NAME_OR_PATH,,}" =~ "deepseek" ]]; then
    PROMPT_TYPE="deepseek-distill-cot-ft"
    temperature=0.6
    top_p=0.95
    max_gen_len=32768
else
    PROMPT_TYPE="qwen25-math-cot-ft"
    temperature=0.0 #0.6 #0.0
    top_p=1 #0.95 #1
    max_gen_len=3000
fi

SPLIT="test"
NUM_TEST_SAMPLE=-1
OUTPUT_DIR="./output_$(basename ${MODEL_NAME_OR_PATH})_verl_grpo_phi_advprm_lookahead_prmreward_rawgain_0904_pass1_10"
DATA_NAME="gsm8k,math500,minerva_math,olympiadbench,amc23"

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --max_tokens_per_call $max_gen_len \
    --temperature $temperature \
    --top_p $top_p \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_names ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 42 \
    --n_sampling 1 \
    --start 0 \
    --end -1 \
    --use_vllm

echo "Model CKPT: "$MODEL_NAME_OR_PATH


'''ROOT=/home/vault/b273dd/b273dd15/checkpoints/verl_grpo_phi_advprm_lookahead_prmreward_rawgain_0904/math_grpo_phi_advprm_lookahead_prmreward_rawgain_0904
for s in {10..110..10}; do
  sh/eval.sh "$ROOT/global_step_${s}/actor/huggingface"
done'''
