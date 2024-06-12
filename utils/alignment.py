from dataclasses import dataclass
from typing import Iterable, Union, List

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

from speech_utils import SAMPLE_RATE, load_audio, interpolate_nans

PUNKT_ABBREVIATIONS = ['dr', 'vs', 'mr', 'mrs', 'prof']

def load_align_model(language_code, device, model_name=None, model_dir=None):
    processor = Wav2Vec2BertProcessor.from_pretrained(model_name)
    align_model = Wav2Vec2BertForCTC.from_pretrained(model_name)

    pipeline_type = "huggingface"
    align_model = align_model.to(device)
    labels = processor.tokenizer.get_vocab()
    align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata, processor


# TODO: I don't think we do need this... we have used the pytorch implementation API for alignment. Discard this.
# def align(
#     transcript: Iterable[SingleSegment],
#     model: torch.nn.Module,
#     processor: Wav2Vec2BertProcessor,
#     align_model_metadata: dict,
#     audio: Union[str, np.ndarray, torch.Tensor],
#     device: str,
#     interpolate_method: str = "nearest",
#     return_char_alignments: bool = False,
#     print_progress: bool = False,
#     combined_progress: bool = False,
# ) -> AlignedTranscriptionResult:
#     """
#     Align phoneme recognition predictions to known transcription.
#     """
    
#     if not torch.is_tensor(audio):
#         if isinstance(audio, str):
#             audio = load_audio(audio)
#         audio = torch.from_numpy(audio)
    
#     MAX_DURATION = audio.shape[0] / SAMPLE_RATE

#     model_dictionary = align_model_metadata["dictionary"]
#     model_lang = align_model_metadata["language"]
#     model_type = align_model_metadata["type"]

#     # 1. Preprocess to keep only characters in dictionary
#     total_segments = len(transcript)
#     for sdx, segment in enumerate(transcript):
#         # strip spaces at beginning / end, but keep track of the amount.
#         if print_progress:
#             base_progress = ((sdx + 1) / total_segments) * 100
#             percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
#             print(f"Progress: {percent_complete:.2f}%...")
            
#         num_leading = len(segment["text"]) - len(segment["text"].lstrip())
#         num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
#         text = segment["text"]

#         per_word = text.split(" ")
        
#         clean_char, clean_cdx = [], []
#         for cdx, char in enumerate(text):
#             char_ = char.lower()
#             # wav2vec2 models use "|" character to represent spaces
#             char_ = char_.replace(" ", "|")
            
#             # ignore whitespace at beginning and end of transcript
#             if cdx < num_leading:
#                 pass
#             elif cdx > len(text) - num_trailing - 1:
#                 pass
#             elif char_ in model_dictionary.keys():
#                 clean_char.append(char_)
#                 clean_cdx.append(cdx)

#         clean_wdx = []
#         for wdx, wrd in enumerate(per_word):
#             if any([c in model_dictionary.keys() for c in wrd]):
#                 clean_wdx.append(wdx)

                
#         punkt_param = PunktParameters()
#         punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
#         sentence_splitter = PunktSentenceTokenizer(punkt_param)
#         sentence_spans = list(sentence_splitter.span_tokenize(text))

#         segment["clean_char"] = clean_char
#         segment["clean_cdx"] = clean_cdx
#         segment["clean_wdx"] = clean_wdx
#         segment["sentence_spans"] = sentence_spans
    
#     aligned_segments: List[SingleAlignedSegment] = []
    
#     # 2. Get prediction matrix from alignment model & align
#     for sdx, segment in enumerate(transcript):
        
#         t1 = segment["start"]
#         t2 = segment["end"]
#         text = segment["text"]

#         aligned_seg: SingleAlignedSegment = {
#             "start": t1,
#             "end": t2,
#             "text": text,
#             "words": [],
#         }

#         if return_char_alignments:
#             aligned_seg["chars"] = []

#         # check we can align
#         if len(segment["clean_char"]) == 0:
#             print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
#             aligned_segments.append(aligned_seg)
#             continue

#         if t1 >= MAX_DURATION:
#             print(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping...')
#             aligned_segments.append(aligned_seg)
#             continue

#         text_clean = "".join(segment["clean_char"])
#         tokens = [model_dictionary[c] for c in text_clean]

#         f1 = int(t1 * SAMPLE_RATE)
#         f2 = int(t2 * SAMPLE_RATE)

#         # using full audio segment for alignment
#         waveform_segment = audio[f1:]

#         with torch.inference_mode():
#             if model_type == "huggingface":
#                 input_features = processor(waveform_segment, sampling_rate=16000).input_features[0]
#                 input_features = torch.tensor(input_features).to(device).unsqueeze(0)
#                 emissions = model(input_features).logits
#             else:
#                 raise NotImplementedError(f"Align model of type {model_type} not supported.")
#             emissions = torch.log_softmax(emissions, dim=-1)

#         emission = emissions[0].cpu().detach()

#         blank_id = 0
#         for char, code in model_dictionary.items():
#             if char == '[pad]' or char == '<pad>':
#                 blank_id = code

#         trellis = get_trellis(emission, tokens, blank_id)
#         path = backtrack(trellis, emission, tokens, blank_id)

#         if path is None:
#             print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
#             aligned_segments.append(aligned_seg)
#             continue

#         char_segments = merge_repeats(path, text_clean)

#         duration = t2 -t1
#         ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

#         # assign timestamps to aligned characters
#         char_segments_arr = []
#         word_idx = 0
#         for cdx, char in enumerate(text):
#             start, end, score = None, None, None
#             if cdx in segment["clean_cdx"]:
#                 char_seg = char_segments[segment["clean_cdx"].index(cdx)]
#                 start = round(char_seg.start * ratio + t1, 3)
#                 end = round(char_seg.end * ratio + t1, 3)
#                 score = round(char_seg.score, 3)

#             char_segments_arr.append(
#                 {
#                     "char": char,
#                     "start": start,
#                     "end": end,
#                     "score": score,
#                     "word-idx": word_idx,
#                 }
#             )

#             # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
#             if cdx == len(text) - 1 or text[cdx+1] == " ":
#                 word_idx += 1
            
#         char_segments_arr = pd.DataFrame(char_segments_arr)

#         aligned_subsegments = []
#         # assign sentence_idx to each character index
#         char_segments_arr["sentence-idx"] = None
#         for sdx, (sstart, send) in enumerate(segment["sentence_spans"]):
#             curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
#             char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx
        
#             sentence_text = text[sstart:send]
#             sentence_start = curr_chars["start"].min()
#             end_chars = curr_chars[curr_chars["char"] != ' ']
#             sentence_end = end_chars["end"].max()
#             sentence_words = []

#             for word_idx in curr_chars["word-idx"].unique():
#                 word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
#                 word_text = "".join(word_chars["char"].tolist()).strip()
#                 if len(word_text) == 0:
#                     continue

#                 # dont use space character for alignment
#                 word_chars = word_chars[word_chars["char"] != " "]

#                 word_start = word_chars["start"].min()
#                 word_end = word_chars["end"].max()
#                 word_score = round(word_chars["score"].mean(), 3)

#                 # -1 indicates unalignable 
#                 word_segment = {"word": word_text}

#                 if not np.isnan(word_start):
#                     word_segment["start"] = word_start
#                 if not np.isnan(word_end):
#                     word_segment["end"] = word_end
#                 if not np.isnan(word_score):
#                     word_segment["score"] = word_score

#                 sentence_words.append(word_segment)
            
#             aligned_subsegments.append({
#                 "text": sentence_text,
#                 "start": sentence_start,
#                 "end": sentence_end,
#                 "words": sentence_words,
#             })

#             if return_char_alignments:
#                 curr_chars = curr_chars[["char", "start", "end", "score"]]
#                 curr_chars.fillna(-1, inplace=True)
#                 curr_chars = curr_chars.to_dict("records")
#                 curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
#                 aligned_subsegments[-1]["chars"] = curr_chars

#         aligned_subsegments = pd.DataFrame(aligned_subsegments)
#         aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
#         aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
#         # concatenate sentences with same timestamps
#         agg_dict = {"text": " ".join, "words": "sum"}
#         if return_char_alignments:
#             agg_dict["chars"] = "sum"
#         aligned_subsegments= aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
#         aligned_subsegments = aligned_subsegments.to_dict('records')
#         aligned_segments += aligned_subsegments

#     # create word_segments list
#     word_segments: List[SingleWordSegment] = []
#     for segment in aligned_segments:
#         word_segments += segment["words"]

#     return {"segments": aligned_segments, "word_segments": word_segments}

# """
# source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
# """
# def get_trellis(emission, tokens, blank_id=0):
#     num_frame = emission.size(0)
#     num_tokens = len(tokens)

#     # Trellis has extra diemsions for both time axis and tokens.
#     # The extra dim for tokens represents <SoS> (start-of-sentence)
#     # The extra dim for time axis is for simplification of the code.
#     trellis = torch.empty((num_frame + 1, num_tokens + 1))
#     trellis[0, 0] = 0
#     trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
#     trellis[0, -num_tokens:] = -float("inf")
#     trellis[-num_tokens:, 0] = float("inf")

#     for t in range(num_frame):
#         trellis[t + 1, 1:] = torch.maximum(
#             # Score for staying at the same token
#             trellis[t, 1:] + emission[t, blank_id],
#             # Score for changing to the next token
#             trellis[t, :-1] + emission[t, tokens],
#         )
#     return trellis

# @dataclass
# class Point:
#     token_index: int
#     time_index: int
#     score: float

# def backtrack(trellis, emission, tokens, blank_id=0):
#     # Note:
#     # j and t are indices for trellis, which has extra dimensions
#     # for time and tokens at the beginning.
#     # When referring to time frame index `T` in trellis,
#     # the corresponding index in emission is `T-1`.
#     # Similarly, when referring to token index `J` in trellis,
#     # the corresponding index in transcript is `J-1`.
#     j = trellis.size(1) - 1
#     t_start = torch.argmax(trellis[:, j]).item()

#     path = []
#     for t in range(t_start, 0, -1):
#         # 1. Figure out if the current position was stay or change
#         # Note (again):
#         # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
#         # Score for token staying the same from time frame J-1 to T.
#         stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
#         # Score for token changing from C-1 at T-1 to J at T.
#         changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

#         # 2. Store the path with frame-wise probability.
#         prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
#         # Return token index and time index in non-trellis coordinate.
#         path.append(Point(j - 1, t - 1, prob))

#         # 3. Update the token
#         if changed > stayed:
#             j -= 1
#             if j == 0:
#                 break
#     else:
#         # failed
#         return None
#     return path[::-1]

# # Merge the labels
# @dataclass
# class Segment:
#     label: str
#     start: int
#     end: int
#     score: float

#     def __repr__(self):
#         return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

#     @property
#     def length(self):
#         return self.end - self.start

# def merge_repeats(path, transcript):
#     i1, i2 = 0, 0
#     segments = []
#     while i1 < len(path):
#         while i2 < len(path) and path[i1].token_index == path[i2].token_index:
#             i2 += 1
#         score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
#         segments.append(
#             Segment(
#                 transcript[path[i1].token_index],
#                 path[i1].time_index,
#                 path[i2 - 1].time_index + 1,
#                 score,
#             )
#         )
#         i1 = i2
#     return segments

# def merge_words(segments, separator="|"):
#     words = []
#     i1, i2 = 0, 0
#     while i1 < len(segments):
#         if i2 >= len(segments) or segments[i2].label == separator:
#             if i1 != i2:
#                 segs = segments[i1:i2]
#                 word = "".join([seg.label for seg in segs])
#                 score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
#                 words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
#             i1 = i2 + 1
#             i2 = i1
#         else:
#             i2 += 1
#     return words