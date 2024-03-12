export HF_HOME=/fs/nexus-projects/brain_project/ck_icml/cache

python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"

module load cuda/11.8.0

# #   "econ_test": {
# #     "file_name": "econometrics_test.json"
# #   },
# #   "elec_test": {
# #     "file_name": "electrical_engineering_test.json"
# #   },
# #   "formal_logic_test": {
# #     "file_name": "formal_logic_test.json"
# #   },
# #   "global_facts_test": {
# #     "file_name": "global_facts_test.json"
# #   },
# #   "high_school_biology_test": {
# #     "file_name": "high_school_biology_test.json"
# #   },
# #   "high_school_geography_test": {
# #     "file_name": "high_school_geography_test.json"
# #   },
# #   "math_test": {
# #     "file_name": "high_school_microeconomics_test.json"
# #   },

# accelerate launch --config_file /nfshomes/sreyang/.cache/huggingface/accelerate/default_config.yaml src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path openai-community/gpt2-large \
#     --dataset alpaca_gpt4_en \
#     --template vanilla \
#     --finetuning_type lora \
#     --lora_target c_attn \
#     --output_dir /fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/output_dir/gpt2_umd/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 2 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 100 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 5.0 \
#     --plot_loss \
#     --report_to none \
#     --fp16

# declare -A tests=(
#     ["econ_test"]="econometrics_test.json"
#     ["elec_test"]="electrical_engineering_test.json"
#     ["formal_logic_test"]="formal_logic_test.json"
#     ["global_facts_test"]="global_facts_test.json"
#     ["high_school_biology_test"]="high_school_biology_test.json"
#     ["high_school_geography_test"]="high_school_geography_test.json"
#     ["math_test"]="high_school_mathematics_test.json"
# )

# for key in "${!tests[@]}"
# do
#     echo "Running for dataset: $key"
#     python src/evaluate.py \
#         --model_name_or_path openai-community/gpt2-large \
#         --finetuning_type lora \
#         --adapter_name_or_path /fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/output_dir/gpt2_umd/$key \
#         --template vanilla \
#         --task mmlu \
#         --split test \
#         --lang en \
#         --n_shot 0 \
#         --batch_size 4 \
#         --save_dir mmlu_eval_{$key}_2
# done

python src/evaluate.py \
    --model_name_or_path openai-community/gpt2-large \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 0 \
    --batch_size 4 \
    --save_dir mmlu_eval_vanilla

# for key in "${!tests[@]}"; do
#     echo "First loop running for dataset: $key"
#     for inner_key in "${!tests[@]}"; do
#         if [ "$key" != "$inner_key" ]; then
#             echo "Second loop, running for dataset: $inner_key with exclusion of $key"
#             echo ${inner_key/test/dev}
#             python src/evaluate.py \
#                 --model_name_or_path openai-community/gpt2-large \
#                 --finetuning_type lora \
#                 --adapter_name_or_path /fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/output_dir/gpt2_umd/${key}/${key}_vs_${inner_key} \
#                 --template vanilla \
#                 --task mmlu \
#                 --split test \
#                 --lang en \
#                 --n_shot 0 \
#                 --batch_size 4 \
#                 --save_dir mmlu_eval_${key}_vs_${inner_key}
#         fi
#     done
# done


# --adapter_name_or_path path_to_checkpoint \
# --finetuning_type lora \
# --save_dir mmlu_eval


# declare -A tests=(
#     ["econ_test"]="econometrics_test.json"
#     ["elec_test"]="electrical_engineering_test.json"
#     ["formal_logic_test"]="formal_logic_test.json"
#     ["global_facts_test"]="global_facts_test.json"
#     ["high_school_biology_test"]="high_school_biology_test.json"
#     ["high_school_geography_test"]="high_school_geography_test.json"
#     ["math_test"]="high_school_mathematics_test.json"
# )

# for key in "${!tests[@]}"
# do
#     echo "Running for dataset: $key"
#     accelerate launch --config_file /nfshomes/sreyang/.cache/huggingface/accelerate/default_config.yaml src/train_bash.py \
#         --stage sft \
#         --do_train \
#         --model_name_or_path openai-community/gpt2-large \
#         --adapter_name_or_path /fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/output_dir/gpt2_umd/high_school_geography_test \
#         --dataset $key \
#         --template vanilla \
#         --finetuning_type lora \
#         --lora_target c_attn \
#         --output_dir /fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/output_dir/gpt2_umd/$key \
#         --overwrite_cache \
#         --overwrite_output_dir \
#         --per_device_train_batch_size 2 \
#         --gradient_accumulation_steps 2 \
#         --lr_scheduler_type cosine \
#         --logging_steps 10 \
#         --save_steps 100 \
#         --learning_rate 5e-5 \
#         --num_train_epochs 25 \
#         --plot_loss \
#         --report_to none \
#         --fp16
# done

# declare -A tests=(
#     ["econ_test"]="econometrics_test.json"
#     ["elec_test"]="electrical_engineering_test.json"
#     ["formal_logic_test"]="formal_logic_test.json"
#     ["global_facts_test"]="global_facts_test.json"
#     ["high_school_biology_test"]="high_school_biology_test.json"
#     ["high_school_geography_test"]="high_school_geography_test.json"
#     ["math_test"]="high_school_mathematics_test.json"
# )

# for key in "${!tests[@]}"; do
#     echo "First loop running for dataset: $key"
#     for inner_key in "${!tests[@]}"; do
#         if [ "$key" != "$inner_key" ]; then
#             echo "Second loop, running for dataset: $inner_key with exclusion of $key"
#             echo ${inner_key/test/dev}
#             accelerate launch --config_file /nfshomes/sreyang/.cache/huggingface/accelerate/default_config.yaml src/train_bash.py \
#                 --stage sft \
#                 --do_train \
#                 --model_name_or_path openai-community/gpt2-large \
#                 --adapter_name_or_path /fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/output_dir/gpt2_umd/${key} \
#                 --dataset ${inner_key/test/dev} \
#                 --template vanilla \
#                 --finetuning_type lora \
#                 --lora_target c_attn \
#                 --output_dir /fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/output_dir/gpt2_umd/${key}/${key}_vs_${inner_key} \
#                 --overwrite_cache \
#                 --overwrite_output_dir \
#                 --per_device_train_batch_size 2 \
#                 --gradient_accumulation_steps 2 \
#                 --lr_scheduler_type cosine \
#                 --logging_steps 10 \
#                 --save_steps 100 \
#                 --learning_rate 5e-5 \
#                 --num_train_epochs 25 \
#                 --plot_loss \
#                 --report_to none \
#                 --fp16
#         fi
#     done
# done

# export HF_HOME=/fs/nexus-projects/brain_project/ck_icml/cache

# python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"

# module load cuda/11.8.0

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path openai-community/gpt2-large \
#     --dataset temp_gpt \
#     --template vanilla \
#     --finetuning_type freeze \
#     --output_dir /fs/nexus-projects/brain_project/ck_icml/ck_llm/LLaMA-Factory/output_dir/gpt_icl_out/ \
#     --per_device_eval_batch_size 4 \
#     --predict_with_generate \
#     --top_p 0.0 \
#     --temperature 0.0 \
#     --do_sample False \
#     --top_k 0 \
#     --fp16


