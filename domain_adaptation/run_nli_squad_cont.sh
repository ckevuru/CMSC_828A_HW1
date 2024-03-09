export HF_HOME=/fs/nexus-projects/audio-visual_dereverberation/ck_838C/cache

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_glue_no_trainer.py \
--model_name_or_path /fs/nexus-projects/audio-visual_dereverberation/ck_838C/hw1/ckpts/squad_train \
--output_dir /fs/nexus-projects/audio-visual_dereverberation/ck_838C/hw1/ckpts/nli_squad_cont/ \
--checkpointing_steps epoch \
--domain_adaptation 0 \
--domain_adaptation_continue 0 \
--squad_continue 1 \
--ner_continue 0 \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--task_name mnli \
--ignore_mismatched_sizes