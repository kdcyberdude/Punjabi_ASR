import faulthandler
faulthandler.enable()


from datasets import load_from_disk, concatenate_datasets, DatasetDict, load_dataset, Audio
import IPython.display as ipd
import speech_utils as su
import random
import numpy as np
from transformers import Wav2Vec2CTCTokenizer
from transformers import SeamlessM4TFeatureExtractor
from transformers import Wav2Vec2BertProcessor
from transformers import Wav2Vec2BertForCTC
from transformers import TrainingArguments
from transformers import Trainer
from functools import partial

import numpy as np
import random
import torch
import os
from datasets import load_metric

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from  data_collator import DataCollatorCTCWithPadding
import json
import sys
sys.append('..')
import creds

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

hf_token = creds.hf_c_token
os.environ["HF_TOKEN"]  = hf_token

USE_WANDB = False
CKPT_PATH = None
BASE_CHECKPOINT = 'facebook/w2v-bert-2.0'

if USE_WANDB:
    import os
    os.environ['WANDB_NOTEBOOK_NAME'] = 'train_w2v-bert_v2.ipynb'

# for trying training on single dataset source
ds1 = load_dataset(creds.yt_ds_hub, cache_dir='./asr_ds_yt_cache')
filtered_ds1_indexes_file = 'selected_indexes_yt_dataset_58.53_percent.json'
ds1_selected_indexes = []
with open(filtered_ds1_indexes_file) as f:
    ds1_selected_indexes = json.load(f)
ds1['train'] = ds1['train'].select(ds1_selected_indexes)
ds1 = ds1['train'].train_test_split(test_size=1500, seed=42)
ds1 = ds1.remove_columns(['id', 'chunk_index', 'start_sr', 'end_sr', 'sample_rate', 'source'])
ds1 = ds1.cast_column('audio', Audio(sampling_rate = 16000))
print(ds1)
ds2 = load_dataset('kdcyberdude/punjabi_asr_dataset_v2', cache_dir='./asr_ds_cache')
filtered_ds2_indexes_file = 'selected_indexes_punjabi_asr_dataset_v2_92.23_percent.json'
with open(filtered_ds2_indexes_file) as f:
    ds2_selected_indexes = json.load(f)
ds2['train'] = ds2['train'].select(ds2_selected_indexes)
ds2 = ds2.remove_columns(['source', 'speaker_id'])
ds2 = ds2.cast_column('audio', Audio(sampling_rate = 16000))
print(ds2)
ds = DatasetDict({
    'train': concatenate_datasets([ds1['train'], ds2['train']]),
    'test': concatenate_datasets([ds1['test'], ds2['test']])
})
print('Summary after preprocessing')
su.get_summary(ds)

ds['train'] = su.remove_audio_samples(ds['train'])
ds['test'] = su.remove_audio_samples(ds['test'])
ds = su.normalize_text_ds(ds, remove_digits=True)
ds['train'] = su.remove_text_samples(ds['train'], column_name='normalized_text', min_text_length=1)
ds['test'] = su.remove_text_samples(ds['test'], column_name='normalized_text')
su.get_summary(ds, text_column='normalized_text')

update_vocab = True
if update_vocab:
  def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

  vocab_train = ds['train'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=ds['train'].column_names)
  vocab_test = ds['test'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=ds['test'].column_names)

  vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

  vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
  vocab_dict["|"] = vocab_dict[" "]
  del vocab_dict[" "]
  vocab_dict["[UNK]"] = len(vocab_dict)
  vocab_dict["[PAD]"] = len(vocab_dict)
  len(vocab_dict)

  import json
  with open('vocab.json', 'w') as vocab_file:
      json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(BASE_CHECKPOINT)
processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):
    # batch is single row 
    audio = batch["audio"]
    batch["input_features"] = processor(audio['array'], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"][0])
    batch["labels"] = processor(text=batch["normalized_text"]).input_ids

    return batch

ds = ds.map(prepare_dataset, remove_columns=ds['train'].column_names, batch_size=1, num_proc=1)
ds['train'] = ds['train'].shuffle(seed=42)
print(ds)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

model = Wav2Vec2BertForCTC.from_pretrained(
    BASE_CHECKPOINT, ignore_mismatched_sizes=True,
    apply_spec_augment = True, 
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    mask_time_prob=0.0, # default 0.05
    layerdrop=0.0, # default 0.1
    add_adapter=True,
    use_intermediate_ffn_before_adapter=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

batch_size = 16
accumulation_steps = 1 
effective_batch_size = batch_size * accumulation_steps
if USE_WANDB:
   report_to = ['tensorboard', 'wandb']
else:
    report_to = ['tensorboard']

epochs = 6
training_args = TrainingArguments(
  output_dir='./checkpoints/w2v-pa-v2',
  # group_by_length=True,
  per_device_train_batch_size=batch_size,
  gradient_accumulation_steps=accumulation_steps,
  evaluation_strategy="steps",
  num_train_epochs=epochs,
  adam_beta1=0.9,
  adam_beta2=0.999,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=900,
  eval_steps=300,
  logging_steps=10,  
  weight_decay=0.01,
  learning_rate=5e-5,
  lr_scheduler_type="cosine",
  load_best_model_at_end=True,
  metric_for_best_model="wer",
  greater_is_better=False,
  warmup_ratio=1/(epochs*2),
  dataloader_num_workers=8,
  dataloader_prefetch_factor=16,
  dataloader_pin_memory=True,
  dataloader_persistent_workers=False,
  run_name='wav2vec2-pa-v2',
  #   ignore_data_skip=True,
  save_total_limit=4,
  push_to_hub=True,
  report_to= report_to,
  resume_from_checkpoint=CKPT_PATH,
  hub_strategy="all_checkpoints"

)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=partial(su.compute_wer_metrics, processor=processor),
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    tokenizer=processor.feature_extractor,
)


if CKPT_PATH is None:
    trainer.train()
else:   
    trainer_stats = trainer.train(resume_from_checkpoint=CKPT_PATH)


model.save_pretrained(creds.w2v_hf_model) 
tokenizer.save_pretrained(creds.w2v_hf_model)
processor.save_pretrained(creds.w2v_hf_model)