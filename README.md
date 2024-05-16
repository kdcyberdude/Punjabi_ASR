# Punjabi_ASR

## Introduction
The `Punjabi_ASR` project is dedicated to advancing Automatic Speech Recognition (ASR) for the Punjabi language, using various datasets to benchmark and improve performance. Our goal is to refine ASR technology to make it more accessible and efficient for speakers of Punjabi.

## Performance
We have benchmarked the ASR model using the IndicSuperb - [AI4Bharat/IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB) ASR benchmark with the following results:

- **Common Voice:** 10.18%
- **Fleurs:** 6.96%
- **Kathbath:** 8.30%
- **Kathbath Noisy:** 9.31%

These Word Error Rates (WERs) demonstrate the current capabilities and focus areas for improvement in our models.

## Example Usage
To use the `w2v-bert-punjabi` model for speech recognition, follow the steps below. This example demonstrates loading the model and processing an audio file for speech-to-text conversion.

### Code
```python
import speech_utils as su
from m4t_processor_with_lm import M4TProcessorWithLM
from transformers import Wav2Vec2BertForCTC, pipeline

# Load the model and processor
model_id = 'kdcyberdude/w2v-bert-punjabi'
processor = M4TProcessorWithLM.from_pretrained(model_id)
model = Wav2Vec2BertForCTC.from_pretrained(model_id)

# Set up the pipeline
pipe = pipeline('automatic-speech-recognition', model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, decoder=processor.decoder, return_timestamps='word', device='cuda:0')

# Process the audio file
output = pipe("example.wav", chunk_length_s=20, stride_length_s=(4, 4))
su.pbprint(output['text'])
```

https://github.com/kdcyberdude/Punjabi_ASR/assets/34835322/88515c45-3212-4457-8d72-a35de0060d65

**Transcription:**
ਉਹ ਕਹਿੰਦੇ ਸਾਡਾ ਸੁਨੇਹਾ ਹੁਣ ਜਾ ਕੇ ਅਹਿਮਦ ਸ਼ਾਹ ਬਦਾਲੀ ਨੂੰ ਦੇ ਦਿਓ ਉਹਨੇ ਸਾਨੂੰ ਪੇਸ਼ਕਸ਼ ਭੇਜੀ ਸੀ ਤਾਜ ਉਸ ਦਾ ਤੇ ਰਾਜ ਸਾਡਾ ਉਹਨੇ ਕਿਹਾ ਸੀ ਕਣਕ ਕੋਰਾ ਮੱਕੀ ਬਾਜਰਾ ਜਵਾਰ ਦੇ ਦਿਆ ਕਰੋ ਤੇ ਜ਼ਿੰਦਗੀ ਜੀ ਸਕਦੇ ਓ ਹੁਣ ਸਾਡਾ ਜਵਾਬ ਉਹਨੂੰ ਦੇ ਦਿਓ ਕਿ ਸਾਡੀ ਜੰਗ ਕੇਸ ਗੱਲ ਦੀ ਐ ਸਾਡੇ ਵੱਲੋਂ ਸ਼ਾਹ ਨੂੰ ਕਹਿ ਦੇਣਾ ਜਾ ਕੇ ਮਿਲਾਂਗੇ ਉਸ ਨੂੰ ਰਣ ਵਿੱਚ ਹੱਥ ਤੇਗ ਉਠਾ ਕੇ ਸ਼ਰਤਾਂ ਲਿਖਾਂਗੇ ਰੱਤ ਨਾਲ ਖੈਬਰ ਕੋਲ ਜਾ ਕੇ ਸ਼ਾਹ ਨਜ਼ਰਾਨੇ ਸਾਥੋਂ ਭਾਲਦਾ ਇਉਂ ਈਨ ਮਨਾ ਕੇ ਪਰ ਸ਼ੇਰ ਨਾ ਜਿਉਂਦੇ ਸੀਤਲਾ ਨੱਕ ਨੱਥ ਪਾ ਕੇ ਇਹ ਸੀ ਉਸ ਵੇਲੇ ਸਾਡੇ ਇਹਨਾਂ ਜਰਨੈਲਾਂ ਦਾ ਕਿਰਦਾਰ ਬਹੁਤ ਵੱਡਾ ਜੀਵਨ ਹੈ ਜਿਹਦੇ ਚ ਰਾਜਨੀਤੀ ਕੂਟਨੀਤੀ ਯੁੱਧ ਨੀਤੀ ਧਰਮਨੀਤੀ ਸਭ ਕੁਝ ਭਰਿਆ ਪਿਆ ਹੈ

## Datasets
The training and testing data used in this project are available on Hugging Face:
- [Punjabi ASR Datasets](https://huggingface.co/datasets/kdcyberdude/Punjabi_ASR_datasets)

## Model
Our current model is hosted on Hugging Face, and you can explore its capabilities through the demo:
- **Model:** [w2v-bert-punjabi](https://huggingface.co/kdcyberdude/w2v-bert-punjabi)
- **Demo:** [Try the model](https://huggingface.co/spaces/kdcyberdude/w2v-bert-punjabi)

## Next Steps
Here are the key areas we're focusing on to advance our Punjabi ASR project:

- [ ] **Training Whisper:** Implement and train the Whisper model to compare its performance against our current models.
- [ ] **Filtering Pipeline:** Develop a robust filtering pipeline to enhance dataset quality by addressing transcription inaccuracies found in datasets like Shrutilipi, IndicSuperb, and IndicTTS.
- [ ] **Building a Custom Dataset:** Compile approximately 500 hours of high-quality Punjabi audio data to support diverse and comprehensive training.
- [ ] **Multilingual Training:** Utilize the linguistic similarities between Punjabi and Hindi to improve model training through multilingual datasets.
- [ ] **Data Augmentation:** Apply techniques such as speed variation and background noise addition to training to bolster the ASR system's robustness.
- [ ] **Iterative Training:** Continuously retrain models like w2v-bert or Whisper based on experimental outcomes and enhanced data insights.

## Collaboration and Support
We are actively seeking collaborators and sponsors to expand our efforts on the Punjabi ASR project. Contributions can be in the form of coding, dataset provision, or compute resources sponsorship. Your support will be crucial in making this practically beneficial for real-life applications.

- **Issues and Contributions:** Encounter an issue or want to help? Create a [GitHub issue](https://github.com/kdcyberdude/Punjabi_ASR/issues) or submit a pull request to contribute directly.
- **Sponsorship:** If you are interested in sponsoring, especially in terms of compute resources, please email us at kdsingh.cyberdude@gmail.com to discuss collaboration opportunities.
