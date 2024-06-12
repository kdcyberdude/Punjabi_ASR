deepspeed run_speech_recognition_seq2seq_streaming.py \
	--deepspeed="ds_config.json" \
	--model_name_or_path="openai/whisper-large-v3" \
	--language="punjabi" \
	--train_split_name="train" \
	--eval_split_name="test" \
	--model_index_name="Whisper Large V3 Punjabi" \
	--output_dir="./checkpoints/whisper/whisper_large_v3_pa_v0.1" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="16" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="1500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="3000" \
	--max_duration_in_seconds="16" \
	--text_column_name="text" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
	--gradient_checkpointing \
	--fp16 \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--use_auth_token \
    --do_remove_punctuation \
    --lr_scheduler_type="cosine" \
    --dataloader_num_workers=4 \
    --dataloader_prefetch_factor=4 \
    --dataloader_pin_memory=True \
    --save_total_limit=4 \
	--streaming="False" \
    --overwrite_output_dir \
    --epochs="6" \
	--push_to_hub \




	# --dataset_name="/mnt/sea/speech/punjabi_asr_datasets/all_except_stt_yt,/mnt/pi/datasets/speech/yt_dataset" \
	# --generation_max_length="225" \
	# --max_steps="5000" \
    # --max_train_samples="10000" \

