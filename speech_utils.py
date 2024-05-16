from datasets import concatenate_datasets, DatasetDict
import pylab as plt
import re
import execjs
import json
from num_to_words import num_to_word
import numpy as np
from datasets import load_metric

wer_metric = load_metric("wer")

# Punjabi font converter
with open('converter.js', 'r') as f:
    js_code = f.read()
ctx = execjs.compile(js_code)

replacements = {
    'ਆ': 'ਅਾ',
    'ਇ': 'ਿੲ',
    'ਈ': 'ੲੀ',
    'ਉ': 'ੳੁ',
    'ਊ': 'ੳੂ',
    'ਏ': 'ੲੇ',
    'ਐ': 'ਅੈ',
    'ਔ': 'ਅੌ',
}

def merge_train_valid_splits(ds):
    train = concatenate_datasets([ds["train"], ds["valid"]])
    return DatasetDict({"train": train, "test": ds["test"]})

# for w2v-bert testing
def num_to_words(input_string, lang='en'):
    numbers = re.findall(r'\d+', input_string)
    for number in numbers:
        word = num_to_word(int(number), lang=lang)
        input_string = input_string.replace(number, word)
    return input_string


def remove_audio_samples(ds, max_duration=30.0, min_duration=0.1, column_name='duration'):
    durations = ds[column_name]
    indices = [i for i, d in enumerate(durations) if (
        d < max_duration and d > min_duration)]
    print(f'Removed {len(durations) - len(indices)} audio samples')
    return ds.select(indices)


def remove_text_samples(ds, max_text_length=5000, min_text_length=1, column_name='text', debug=False):
    texts = ds[column_name]
    indices = [i for i, t in enumerate(texts) if (
        len(t) < max_text_length and len(t) > min_text_length)]
    print(f'Removed {len(texts) - len(indices)} text samples')

    if debug:
        normal_text = ds['text']
        texts_removing = [normal_text[i]
                          for i in range(len(texts)) if i not in indices]
        texts_removing_normalized = [texts[i]
                                     for i in range(len(texts)) if i not in indices]
        for i in range(len(texts_removing)):
            print(f'{texts_removing[i]} -> {texts_removing_normalized[i]}')

    return ds.select(indices)


def get_split_duration(ds_split):
    duration_in_hours = sum(ds_split['duration']) / 3600
    return duration_in_hours


def normalize_transcript(ds, replace_vowels, text_column='text', remove_digits=True):
    # Manual analysis of characters in the dataset
    # these things are required but since there are very few samples of them in the dataset... we are removing them - '\u200b', '\u200c', 'ਁ', 'ੑ'
    remove_sents = ['ઘ', 'ˆ', '%', 'Â', 'a', 'b', 'c', 'e', 'f', 'g', 'h', 'i', 'l', 'n', 'o', 'r', 's', 't', 'x', '\u200b', '\u200c', 'ਁ', 'ੑ']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if remove_digits:
        remove_sents.extend(numbers)
    replace_with_space = ['-', '.', '/', '+', '…', 'ੴ',  '–', '—', '‘', '’', "”", '।', '|',  '¥',
                          '°', '·',  'à', '~', '¤', '½', '¾', 'ਃ', '!', '"', "'", '$', ':', ';', '?', 'I', '[', ']', 'Œ']
    # remove chars with space if not surrounded by digits
    replace_with_space_if = [',', ]
    
    texts = ds[text_column]
    updated_texts = []
    for text in texts:
        for c in replace_with_space:
            text = text.replace(c, ' ')
        for c in replace_with_space_if:
            if c in text:
                if text[text.index(c)-1].isdigit() and text[text.index(c)+1].isdigit():
                    text = text.replace(c, '')
                else:
                    text = text.replace(c, ' ')

        if replace_vowels:
            for c in replacements.keys():
                text = text.replace(c, replacements[c])

        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        updated_texts.append(text)

    ds = ds.add_column('normalized_text', updated_texts)

    remove_indexes = [i for i in range(len(texts)) if any(
        c in texts[i] for c in remove_sents)]
    unremoved_indexes = [i for i in range(
        len(texts)) if i not in remove_indexes]
    ds = ds.select(unremoved_indexes)
    print(f'Removed {len(remove_indexes)} sentences')
    return ds


def normalize_text_for_inference(text, vocab_chars, replace_vowels=False, strategy='remove'):
    if ',' in text:
        if text[text.index(',')-1].isdigit() and text[text.index(',')+1].isdigit():
            text = text.replace(',', '')

    if strategy == 'num2word':
        text = num_to_words(text, lang='pa')
        text = text.replace(',', '')

    if replace_vowels:
        for c in replacements.keys():
            text = text.replace(c, replacements[c])

    for c in text:
        if c not in vocab_chars:
            text = text.replace(c, ' ')

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def normalize_texts_for_inference(ds, vocab_chars, text_column='text', strategy='remove'):
    strategies = ['remove', 'num2word', 'nothing']
    assert strategy in strategies, f'Invalid strategy. Choose from {strategies}'

    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    texts = ds[text_column]
    updated_texts = []
    for text in texts:
        text = normalize_text_for_inference(
            text, vocab_chars, strategy=strategy)
        updated_texts.append(text)

    ds = ds.add_column('normalized_text', updated_texts)

    if strategy == 'remove':
        remove_indexes = [i for i in range(len(texts)) if any(
            c in texts[i] for c in numbers)]
        if len(remove_indexes) > 0:
            print(f'Removed {len(remove_indexes)} sentences')
    else:
        remove_indexes = []

    unremoved_indexes = [i for i in range(
        len(texts)) if i not in remove_indexes]
    ds = ds.select(unremoved_indexes)

    return ds


def check_sents_count(chars_list, ds, text_column='normalized_text'):
    texts = ds[text_column]
    for c in chars_list:
        count = 0
        for text in texts:
            if c in text:
                count += 1
        print(f'{c}: {count}')


def check_sents_count_in_list(chars_list, ds, text_column='text'):
    texts = ds[text_column]
    counts = 0
    for text in texts:
        if any(c in text for c in chars_list):
            counts += 1
    print(f'Count: {counts}')


def pbprint(txt):
    res = ctx.eval(f'PunjabiFontConvertor.convert("{txt}", "AnmolUni", "Arial Unicode MS")')
    print(res)
    return res


def print_red(text):
    print(f"\x1b[31m\"{text}\"\x1b[0m")


def normalize_text_ds(ds, replace_vowels=False):
    ds['train'] = normalize_transcript(ds['train'], replace_vowels)
    ds['test'] = normalize_transcript(ds['test'], replace_vowels)
    return ds


def add_silence(ds):
    import numpy as np

    def add_silence_to_audio_array(batch):
        audios = batch['audio']
        audios_res = []
        for i in range(len(audios)):
            audio = audios[i]['array']
            # with initial small training... add checking shrutilipi kind of ds - using 1sec of silence after the audio gives less loss... but I want model to learn this without 1 sec of silence
            audio = np.concatenate([np.zeros(256), audio, np.zeros(8000)])
            audios[i]['array'] = audio
            audios_res.append(audios[i])

        batch['audio'] = audios_res
        return batch

    ds = ds.map(add_silence_to_audio_array,
                batched=True, num_proc=12, batch_size=16)
    return ds


def get_summary(ds, text_column='text', duration_column='duration'):
    print('Dataset summary: ')
    print(ds)
    # get the duration of each split whichever is present
    for split in ds.keys():
        duration_in_hours = get_split_duration(ds[split])
        print(f'{split} duration in hours: {duration_in_hours.__round__(2)}')

    # max duration, min duration
    all_durations = []
    for split in ds.keys():
        all_durations.extend(ds[split][duration_column])
    print(f'Max duration: {max(all_durations)} seconds')
    print(f'Min duration: {min(all_durations)} seconds')
    print(f'Avg duration: {sum(all_durations) / len(all_durations)} seconds')
    print()
    # max text length, min text length
    all_texts = []
    for split in ds.keys():
        all_texts.extend(ds[split][text_column])
    print(f'Max text length: {max([len(t) for t in all_texts])} characters')
    print(f'Min text length: {min([len(t) for t in all_texts])} characters')
    print(f'Avg text length: {sum([len(t) for t in all_texts]) / len(all_texts)} characters')
    print()

    for split in ds.keys():
        plt.hist(ds[split][duration_column])


def dataset_filtering_based_on_sample_loss(train_file, test_file):
    with open(test_file, 'r') as f:
        test_losses = json.load(f)
    with open(train_file, 'r') as f:
        train_losses = json.load(f)

    # get the keys where loss is <= 1
    train_losses_s = {k: v for k, v in train_losses.items() if v <= 1.1}
    test_losses_s = {k: v for k, v in test_losses.items() if v <= 1.1}

    train_losses_s_keys = [int(k) for k in train_losses_s.keys()]
    test_losses_s_keys = [int(k) for k in test_losses_s.keys()]
    train_losses_keys = [int(k) for k in train_losses.keys()]
    test_losses_keys = [int(k) for k in test_losses.keys()]

    train_loss_accept_indexes = train_losses_s_keys
    test_loss_accept_indexes = test_losses_s_keys

    train_loss_reject_indexes = [
        i for i in train_losses_keys if i not in train_losses_s_keys]
    test_loss_reject_indexes = [
        i for i in test_losses_keys if i not in test_losses_s_keys]

    # print len summary
    print(len(train_loss_accept_indexes), len(train_loss_reject_indexes))
    print(len(test_loss_accept_indexes), len(test_loss_reject_indexes))
    return train_loss_accept_indexes, train_loss_reject_indexes, test_loss_accept_indexes, test_loss_reject_indexes


def filter_and_save(ds, train_loss_accept_indexes, train_loss_reject_indexes, test_loss_accept_indexes, test_loss_reject_indexes, save_ds_name):
    dsx = ds.remove_columns('normalized_text')
    dsx_select = dsx['train'].select(train_loss_accept_indexes)
    dsx_reject = dsx['train'].select(train_loss_reject_indexes)

    dsx_select_test = dsx['test'].select(test_loss_accept_indexes)
    dsx_reject_test = dsx['test'].select(test_loss_reject_indexes)

    ds_filtered = DatasetDict({
        'train': dsx_select,
        'test': dsx_select_test,
    })

    ds_filtered_reject = DatasetDict({
        'train': dsx_reject,
        'test': dsx_reject_test,
    })

    ds_filtered.save_to_disk(
        f'/mnt/sea/speech/ds_filtered/{save_ds_name}_pa_ASR_filtered')
    ds_filtered_reject.save_to_disk(
        f'/mnt/sea/speech/ds_filtered/{save_ds_name}_pa_ASR_filtered_reject')


def shrutilipi_ds_filtering(ds):
    test_file = '/home/kd/Desktop/proj/apr/speech_pa/shrutilipi_ test_losses.json'
    train_file = '/home/kd/Desktop/proj/apr/speech_pa/shrutilipi_ train_losses.json'
    train_a_idx, train_r_idx, test_a_idx, test_r_idx = dataset_filtering_based_on_sample_loss(
        train_file, test_file)
    filter_and_save(ds, train_a_idx, train_r_idx,
                    test_a_idx, test_r_idx, 'shrutilipi')


def indicsuperb_ds_filtering(ds):
    test_file = '/home/kd/Desktop/proj/apr/speech_pa/indicsuperb_test_losses.json'
    train_file = '/home/kd/Desktop/proj/apr/speech_pa/indicsuperb_train_losses.json'
    train_a_idx, train_r_idx, test_a_idx, test_r_idx = dataset_filtering_based_on_sample_loss(
        train_file, test_file)
    filter_and_save(ds, train_a_idx, train_r_idx,
                    test_a_idx, test_r_idx, 'indicsuperb')


def process_dataset(batch, processor):
    # batch is single row
    audio = batch["audio"]

    clean_audio_arr = audio["array"]
    noised_audio = clean_audio_arr

    batch["input_features"] = processor(
        noised_audio, sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])

    batch["labels"] = processor(text=batch["normalized_text"]).input_ids
    return batch


def compute_wer_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
