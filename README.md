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
