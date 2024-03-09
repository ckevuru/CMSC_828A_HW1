export HF_HOME=/fs/nexus-projects/audio-visual_dereverberation/ck_838C/cache

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=4 run_glue_no_trainer.py \
--model_name_or_path google-bert/bert-large-uncased \
--output_dir /fs/nexus-projects/audio-visual_dereverberation/ck_838C/hw1/ckpts/nli_dom_train_step_1/ \
--checkpointing_steps epoch \
--domain_adaptation 1 \
--domain_adaptation_continue 0 \
--squad_continue 0 \
--ner_continue 0 \
--num_train_epochs 5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--task_name mnli \
--learning_rate 2e-5