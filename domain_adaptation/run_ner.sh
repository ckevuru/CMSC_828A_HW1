export HF_HOME=/fs/nexus-projects/audio-visual_dereverberation/ck_838C/cache

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=4 run_ner_no_trainer.py \
--dataset_name Babelscape/wikineural \
--model_name_or_path google-bert/bert-large-uncased \
--tokenizer_name google-bert/bert-large-uncased \
--output_dir /fs/nexus-projects/audio-visual_dereverberation/ck_838C/hw1/ckpts/ner_train/ \
--checkpointing_steps epoch \
--num_train_epochs 5 \
--learning_rate 2e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16