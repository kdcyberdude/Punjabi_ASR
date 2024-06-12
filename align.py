

import sys
sys.path.append('..')
from utils.alignment import align, load_align_model
from speech_utils import load_audio, SAMPLE_RATE
import IPython.display as ipd
from tqdm import tqdm 
import torch
from utils.segment_types import SingleSegment, TranscriptionResult
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from datasets import load_from_disk, load_dataset, Audio
import matplotlib.pyplot as plt
import torchaudio.functional as F
import re
import numpy as np
import speech_utils as su
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ds_all = load_from_disk('/mnt/pi/datasets/speech/yt_dataset')
# ds_all = load_dataset('kdcyberdude/punjabi_asr_dataset_v2', cache_dir='./asr_ds_cache')
ds_all = ds_all.cast_column('audio', Audio(sampling_rate = 16000))

alignment_save_result_file_name = 'alignment_probs_for_yt_dataset.pkl'
# torch.cuda.set_device(1)

ds = ds_all['train']

print(ds)

def show_example(i):
    print(ds[i]['text'])
    audio = ds[i]['audio']['array']
    ipd.display(ipd.Audio(audio, rate=SAMPLE_RATE))

print(ds[0])

device = torch.device('cuda')

align_language = 'pa'
align_model_base = 'kdcyberdude/w2v-bert-punjabi'
align_model_verbatim = '/home/kd/Desktop/proj/apr/Punjabi_ASR/checkpoints/wav2vec2-bert-pa_indicvoice_verbatim_2/checkpoint-3500'
results = []
model_v, align_metadata_v, processor_v = load_align_model(align_language, device, model_name=align_model_verbatim)
model_b, align_metadata_b, processor_b = load_align_model(align_language, device, model_name=align_model_base)

# BOTH processors should be same
assert processor_v.tokenizer.get_vocab() == processor_b.tokenizer.get_vocab()
processor  = processor_v

model_b.eval()
model_v.eval()
font_path = './AnmolUni.ttf'  
from pathlib import Path
font_path = Path(font_path)
prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = ["Ubuntu", prop.get_name()]

sample = None
for i in range(1000):
    if ds[i]['duration'] < 5:
        print(i)
        sample = ds[i]
        break

print(sample)

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

def align(emission, tokens, transcript):
    try:
        targets = torch.tensor([tokens], dtype=torch.int32, device=device)
        if emission.shape[1] <= targets.shape[1]:
            targets = targets[:, :-1] # remove leading * token
            transcript = transcript[:-1]
            if emission.shape[1] != targets.shape[1]:
                targets = targets[:, 1:] # remove trailing * token
                transcript = transcript[1:]
        alignments, scores = F.forced_align(emission, targets, blank=0)

        alignments, scores = alignments[0], scores[0]  # remove batch dimension 
        scores = scores.exp()  # convert back to probability
    except RuntimeError as e:
        print(f'Alignment failed for {len(tokens)} tokens and {emission.shape[1]} frames')
        print(f'Alignment failed this will happen either there is no speech in the audio or the speech is very fast that even google STT cant transcribe it (manually checked and confirmed these cases)')
        print(e)
        raise ValueError('Alignment failed')
    return alignments, scores, transcript, targets.shape[1]

def compute_alignments(emission, transcript, dictionary):
    # tokens = processor.tokenizer.encode(''.join(transcript))
    tokens = [dictionary[char] for word in transcript for char in word]
    alignment, scores, transcript, targets_shape = align(emission, tokens, transcript)
    token_spans = F.merge_tokens(alignment, scores)
    word_spans = unflatten(token_spans, [len(word) for word in transcript])
    return word_spans, targets_shape


# add a special '*' token to detect unaligned words or sequences in the transcript
dictionary = processor.tokenizer.get_vocab()
dictionary["*"] = len(dictionary)
vocab_chars = list(processor.tokenizer.get_vocab().keys())[3:-2]

# Create a reverse dictionary to map numerical IDs back to their corresponding tokens.
dic_rev = {v: k for k, v in dictionary.items()}

# Define character replacements for specific Punjabi characters to their modified forms.
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

normalize_chars = {
    'ਭ': 'ਪ',
    'ਫ਼': 'ਫ',
    'ਜ਼': 'ਜ',
    'ਗ਼': 'ਗ',
    'ਖ਼': 'ਖ',
    'ਸ਼': 'ਸ਼',
    'ਲ਼': 'ਲ',
    'ਣ': 'ਨ',
    'ਓਂ': 'ਓ',
}

# List of Punjabi vowel signs (matras).
matras = [ 'ਂ', '਼', 'ਾ', 'ਿ', 'ੀ', 'ੁ', 'ੂ', 'ੇ', 'ੈ', 'ੋ', 'ੌ', '੍', 'ੰ', 'ੱ']

def process_star_transcript(text):
    """Process transcript by inserting '*' between words and at boundaries."""
    text = re.sub(r'\s+', ' ', text.strip())
    return f'* {text.replace(" ", " * ")} *'

def get_word_spans(model, text, audio, debug=False):
    """Extract word spans from audio using a w2v-bert trained model on clean dataset."""
    if debug:
        print(text)
    transcript_words = text.split()
    audio = torch.from_numpy(audio)

    with torch.no_grad():
        input_features = processor(audio, sampling_rate=SAMPLE_RATE).input_features[0]
        input_features = torch.tensor(input_features).to(device).unsqueeze(0)
        emissions = model(input_features).logits
        emissions = torch.log_softmax(emissions, dim=-1)
    
    pred_ids = torch.argmax(emissions, dim=-1)[0]
    star_dim = torch.zeros((1, emissions.size(1), 1), device=emissions.device, dtype=emissions.dtype)
    emissions = torch.cat((emissions, star_dim), 2)

    word_spans, targets_shape = compute_alignments(emissions, transcript_words, dictionary)
    return {'word_spans': word_spans, 'pred_ids': pred_ids, 'text_tokens_shape': targets_shape}


def evaluate_predictions(words_map, deleted_words, replacements, matras, debug_infos, debug):
    prev_next_word_overlap = {}
    for i in range(len(words_map)):
        current, current_type, current_score = words_map[i]
        # Check phonetic context to potentially adjust scores
        if current_type == 'word': #  and i > 0 and i < len(words_map) - 1
            if i==0:
                prev_word = ''
                next_word = words_map[i+1][0]
            elif i == len(words_map) - 1:
                prev_word = words_map[i-1][0]
                next_word = ''
            else:
                prev_word, next_word = words_map[i-1][0], words_map[i+1][0]
            p_nc, p_nv, n_nc, n_nv = adjust_prob_based_on_context(current, prev_word, next_word, words_map, i, matras, debug_infos)
            prev_next_word_overlap[i] = (p_nc, p_nv, n_nc, n_nv)
        if current_type == 'deleted_word':
            handle_deleted_word_cases(current, words_map, i, replacements, deleted_words, debug)

    for d, idx in deleted_words:
        if idx == 0:
            pv = ''
            nv = words_map[idx + 1][0]
        elif idx == len(words_map) - 1:
            pv = words_map[idx - 1][0]
            nv = ''
        else:
            pv = words_map[idx - 1][0]
            nv = words_map[idx + 1][0]
        if ' ' in d and d.count(' ') == 1: # skip long deleted words
            p,n = d.split()
            if n == nv[:len(n)] or p == pv[-len(p):]:
                deleted_words.remove((d, idx))
        elif d in pv or d in nv:
            deleted_words.remove((d, idx))

def count_consonents_and_vowels(word):
    consonents = 0
    vowels = 0
    for char in word:
        if char in matras:
            vowels += 1
        else:
            consonents += 1
    return consonents, vowels

def calculate_cons_vow_probs(nc, nv, p_nc, p_nv, n_nc, n_nv):
    # Weightages
    wc = 0.7  # weightage for consonants
    wv = 0.3  # weightage for vowels
    
    # Calculate probabilities
    if nc > 0 and nv > 0:
        p_c = wc / nc
        p_v = wv / nv
    elif nc > 0:
        p_c = (wc + wv) / nc
        p_v = 0
    elif nv > 0:
        p_c = 0
        p_v = (wc + wv) / nv
    else:
        p_c = 0
        p_v = 0

    prev_word_prob = p_nc * p_c + p_nv * p_v
    nex_word_prob = n_nc * p_c + n_nv * p_v
    
    return prev_word_prob, nex_word_prob

def normalize(word):
    for k, v in normalize_chars.items():
        word = word.replace(k, v)
    return word

def adjust_prob_based_on_context(current, prev_word, next_word, words_map, index, matras, debug_infos):
    current = normalize(current)
    prev_word = normalize(prev_word)
    next_word = normalize(next_word)

    p_nc, p_nv, n_nc, n_nv = 0, 0, 0, 0
    # Punjabi specific nuances: understood by manually analyzing the results of the alignment
    last_end_char = prev_word[-1] if prev_word else ''
    next_first_char = next_word[0] if next_word else ''
    star_prev_word = prev_word.split()[-1] if prev_word else ''
    star_next_word = next_word.split()[0] if next_word else ''
    if next_first_char in matras and len(next_word) > 1:
        next_first_char = next_word[1]
    nc, nv = count_consonents_and_vowels(current)
        
    matched_chars = ''
    if star_prev_word != '' and star_prev_word in current:
        matched_chars = star_prev_word
    elif last_end_char == current[0]:
        matched_chars = last_end_char
    
    p_nc, p_nv = count_consonents_and_vowels(matched_chars)
    
    matched_chars = ''
    if star_next_word != '' and star_next_word in current:
        matched_chars = star_next_word
    elif next_first_char == current[-1]:
        matched_chars = next_first_char
    n_nc, n_nv = count_consonents_and_vowels(matched_chars)

    prev_word_prob_adjustment, next_word_prob_adjustment = calculate_cons_vow_probs(nc, nv, p_nc, p_nv, n_nc, n_nv)
    current_prob_adjustment = prev_word_prob_adjustment + next_word_prob_adjustment

    _, type_, score_ = words_map[index]
    # special case # ਆ (ਾ) => 0.25 * 2 = 0.5
    if len(current) == 1:
        words_map[index] = (current, type_, score_ * 2) 
        debug_infos[index] = debug_infos[index].replace('=>      Token Scores', f'=> [x*2][Single word] -> {(score_ * 2):.2f}     Token Scores')
    if current_prob_adjustment > 0:
        words_map[index] = (current, type_, score_ + current_prob_adjustment)
        di_scores = ''
        if prev_word_prob_adjustment > 0:
            di_scores += f' + P:{prev_word_prob_adjustment:.2f}'
        if next_word_prob_adjustment > 0:
            di_scores += f' + N:{next_word_prob_adjustment:.2f}'
        debug_infos[index] = debug_infos[index].replace('=>      Token Scores', f'=> [S:{score_:.2f} {di_scores}] -> {(score_ + current_prob_adjustment):.2f}     Token Scores')

    return p_nc, p_nv, n_nc, n_nv

def apply_replacements(word, replacements):
    for k, v in replacements.items():
        word = word.replace(k, v)
    return word

def handle_deleted_word_cases(current, words_map, index, replacements, deleted_words, debug):
    # You can tweek these conditions to even detect a single extra phoneme - good for TTS... 
    if current.count(' ') == 2: # ਰ ਤੀ ਗ
        current = current.split()[1]
    if len(current) <= 2:
        return # This is the case where * word is associated with at max 2 characters. It could be a phonetic component, fumble or a neighboring word
    if ' '  in current and len(current) < 5:
        return # Generally if a space is predicted in a star token word. Which basically signifies left and right side of space either belongs to neighouring words or new word is inserted; Example: 'ਹ ਹੈ' len:4 
    if current in matras:
        return # This is the case where a matra is predicted as a deleted word. This is not a valid case

    if index == 0:
        pv = ''
        nv = words_map[index + 1][0]
    elif index == len(words_map) - 1:
        pv = words_map[index - 1][0]
        nv = ''
    else:
        pv = words_map[index - 1][0]
        nv = words_map[index + 1][0]

    current = normalize(current)
    pv = normalize(pv)
    nv = normalize(nv)

    # Further evaluation for deleted words involves checking if the deleted word fits well with surrounding context
    # if index > 0 and index < len(words_map) - 1: # TODO: remove this
    next_replaced = apply_replacements(nv, replacements)
    prev_replaced = apply_replacements(pv, replacements)
    if current not in pv and current not in prev_replaced and current not in nv and current not in next_replaced:
        deleted_words.append((current, index))
        if debug:
            print(f'Deleted word is: {current}')
   
def process_word_scores(word, pred_ids, words_map, word_scores, star_token, debug):
    scores = []
    score_weights = []
    tokens = []

    debug_info = ""

    if len(word) == 1 and word[0].token == star_token:  # frames predicted with star token
        star_tokens = pred_ids[word[0].start:word[0].end]
        words_map.append((processor.decode(star_tokens), 'deleted_word', word[0].score))
        scores.append(word[0].score)
        score_weights.append(1)
        if debug:
            # su.print_red(f'{dic_rev[word[0].token]} ({processor.decode(star_tokens)})', end='\n')
            debug_info += su.get_red(f'{dic_rev[word[0].token]} ({processor.decode(star_tokens)})')
            
    else:
        for span in word:
            if debug:
                # su.print_green(f'{dic_rev[span.token]}', end='')
                debug_info += su.get_green(f'{dic_rev[span.token]}')
            # Weightage score - Consonants will have .7 weightage and vowel signs will have .3 weightage
            token = processor.decode(span.token)
            if token in matras:
                scores.append(span.score * 0.3)
                score_weights.append(0.3)
            else:
                scores.append(span.score * 0.7)
                score_weights.append(0.7)
            tokens.append(span.token)
        words_map.append((processor.decode(tokens), 'word', sum(scores)/sum(score_weights)))
        predicted_tokens = pred_ids[word[0].start:word[-1].end]

        if debug and len(scores) > 0:
            # su.print_blue(f' ({processor.decode(predicted_tokens)})', end='')
            debug_info += su.get_blue(f' ({processor.decode(predicted_tokens)})')
            diff = word[0].end - word[0].start
            template = " => {:<4} Token Scores: [{}] => {:.2f} / {:.1f} = {}"
            average_score = sum(scores) / sum(score_weights) if sum(score_weights) != 0 else 0  # Safe division
            scores_2f = [f"{score:.2f}" for score in scores]
            formatted_scores = ", ".join(f"{su.get_green(score) if weight == 0.7 else su.get_blue(score)}" for score, weight in zip(scores_2f, score_weights))
            # print(template.format("", formatted_scores, sum(scores), sum(score_weights), su.get_red(f"{average_score:.2f}")))
            debug_info += template.format("", formatted_scores, sum(scores), sum(score_weights), su.get_red(f"{average_score:.2f}"))

    if scores:
        word_scores.append((sum(scores)/sum(score_weights), processor.decode(tokens)))

    return debug_info

def is_correct(word_spans, text_tokens_shape, pred_ids, debug=False):
    """
    Evaluates the accuracy of word spans extracted from predictions.
    Determines if there are any deleted or incorrectly inserted words.
    
    Args:
    word_spans : List[Tuple]
        A list of word spans, each span being a tuple representing a word and its properties.
    pred_ids : torch.Tensor
        The predicted indices from the model, representing each audio frame's predicted token.
    debug : bool, optional
        Flag to enable detailed debug output. Default is False.

    Returns:
    dict
        Dictionary containing information about deleted words, extra words, and whether an extra word should be removed.
    """
    words_map = []
    word_scores = []
    star_token = len(dictionary) - 1  # star_token is the last token in the dictionary
    deleted_words = []
    debug_infos = []

    # Process each word span for score and token analysis
    for index, word in enumerate(word_spans):
        di = process_word_scores(word, pred_ids, words_map, word_scores, star_token, debug)
        debug_infos.append(di)

    # Evaluate the predictions for contextual accuracy and potential errors
    evaluate_predictions(words_map, deleted_words, replacements, matras, debug_infos, debug)

    if debug:
        print('\n'.join(debug_infos))

    # Determine the word with the minimum score and decide if it's an erroneously inserted word
    if words_map:
        min_word_score = min(words_map, key=lambda x: x[2])  # Find the word with the lowest probability score
        threshold = 0.4  # Define a threshold for determining if a word is incorrectly inserted
        remove_extra_word = min_word_score[2] < threshold
        if debug:
            print(f'Min Score word (Word Inserted): {min_word_score[0]} | score: {min_word_score[2]:.3f}')

    else:
        min_word_score = ("", 0)
        remove_extra_word = False

    return {
        'has_deleted_word': deleted_words,
        'remove_extra_word': remove_extra_word,
        'min_prob_word': (min_word_score[0], min_word_score[2]),
        'words_map': words_map,
        'text_tokens_shape': text_tokens_shape
    }

print('Aligning...')
import pickle

def save_checkpoint(results, index, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump({'results': results, 'last_index': index}, f)

# Function to load state if exists
def load_checkpoint(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            return data['results'], data['last_index']
    return [], -1

# Initialize or load from checkpoint
checkpoint_file_name = 'alignment_checkpoints/alignment_checkpoint.pkl'
if not os.path.exists('alignment_checkpoints'):
    os.makedirs('alignment_checkpoints')
results, last_processed_index = load_checkpoint(checkpoint_file_name)

for i in tqdm(range(last_processed_index + 1, len(ds))):
    s = ds[i]
    test_text = f'{s["text"]}'
    source = s.get('source', 'yt_stt' if s['source'] is None else 'indicsuperb')
    norm_text = su.normalized_text(test_text, vocab_chars)
    if norm_text == '':
        results.append({'indicvoice_verbatim': None, 'all_ds': None, 'text': test_text, 'norm_text': norm_text, 'index': i, 'source': source, 'duration': s['duration']})
        continue

    test_audio = s['audio']['array']
    test_audio = np.concatenate([np.zeros(1425), test_audio, np.zeros(1425)]) # Pad audio
    norm_text = process_star_transcript(norm_text)
    try:
        result_v = is_correct(**get_word_spans(model_v, norm_text, test_audio), debug=False)
        result_b = is_correct(**get_word_spans(model_b, norm_text, test_audio), debug=False)
    except Exception as e:
        print(f'FAST OR EMTPY SPEECH: {i} - {e}\n-------------------------------------------------')
        result_v = {'has_deleted_word': [], 'remove_extra_word': True, 'min_prob_word': ('FAST OR EMTPY SPEECH', 0.0)}
        result_b = {'has_deleted_word': [], 'remove_extra_word': True, 'min_prob_word': ('FAST OR EMTPY SPEECH', 0.0)}
    results.append({'indicvoice_verbatim': result_v, 'all_ds': result_b, 'text': test_text, 'norm_text': norm_text, 'index': i, 'source': source, 'duration': s['duration']})
    
    # Checkpoint every 100 iterations
    if i % 10000 == 0:
        save_checkpoint(results, i, checkpoint_file_name)
    torch.cuda.empty_cache()

# Final save
save_checkpoint(results, len(ds) - 1, checkpoint_file_name)

print('DONE')
print(f'Total results processed: {len(results)}')

# Save final results
with open(alignment_save_result_file_name, 'wb') as f:
    pickle.dump(results, f)