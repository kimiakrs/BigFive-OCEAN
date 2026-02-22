# Project: Real-time Multimodal Personality Analysis (OCEAN)
This project implements a multimodal deep learning pipeline for real-time personality trait prediction based on 
the Big Five (OCEAN) model, integrating audio and visual cues extracted from short video clips (15 seconds) using Transformer-based attention mechanisms.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)
[![Deep Learning](https://img.shields.io/badge/Field-Deep%20Learning-black.svg)](https://en.wikipedia.org/wiki/Deep_learning)
[![Multimodal Learning](https://img.shields.io/badge/Paradigm-Multimodal%20Learning-purple.svg)](https://en.wikipedia.org/wiki/Multimodal_learning)
[![Transformer](https://img.shields.io/badge/Architecture-Transformer-red.svg)](https://www.tensorflow.org/text/tutorials/transformer)
[![Visual Backbone](https://img.shields.io/badge/Visual-EfficientNetB0-yellow.svg)](https://keras.io/api/applications/efficientnet/)
[![LSTM](https://img.shields.io/badge/Temporal-LSTM-green.svg)](https://keras.io/api/layers/recurrent_layers/lstm/)
[![AST](https://img.shields.io/badge/Audio-AST%20Transformer-brown.svg)](https://arxiv.org/abs/2104.01778)
[![Cross-Modal Fusion](https://img.shields.io/badge/Fusion-Cross--Attention-pink.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-ChaLearn%20FI%20V2-darkgreen.svg)](https://chalearnlap.cvc.uab.es/)
[![Task](https://img.shields.io/badge/Task-Personality%20Regression-brightgreen.svg)](#)
[![CVPR](https://img.shields.io/badge/Benchmark-CVPR%202017-lightgrey.svg)](https://openaccess.thecvf.com/)
[![Real-Time](https://img.shields.io/badge/Inference-Real--Time-blue.svg)](#)
[![Logging](https://img.shields.io/badge/Output-CSV%20Logging-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Quick Start (Kaggle)

1. **Open the notebook**: In Google Colab or Locally in your computer
2. **Add datasets** (see [Dataset section](#dataset) for details):
3. **Enable GPU** in notebook settings (T4 x2 or higher)
4. **Run all cells**


---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Performance](#performance)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [EXecution Instructions](#execution-instructions)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [Disclaimer](#disclaimer)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project presents a real-time multimodal deep learning framework for predicting Big Five (OCEAN) personality traits from short video clips (15 seconds). 
The system integrates audio and visual features using a Transformer-based attention architecture to model temporal dynamics and cross-modal interactions.

The pipeline consists of four major components:

1. *Multimodal Feature Extraction*

   * Visual features extracted from sampled video frames

   * Audio features derived from Mel-spectrogram representations

2. *Cross-Modal Fusion*
   * Audio and visual embeddings are fused using attention mechanisms to capture inter-modal correlations.

3. *Temporal Modeling*
   * Transformer-based sequence learning captures behavioral dynamics over time.

4. *Real-Time Inference & Logging*
   * The trained model continuously monitors a directory for new video uploads, performs inference, and automatically logs predicted OCEAN scores into a structured CSV file.

The system is evaluated using the ChaLearn First Impressions V2 dataset (CVPR 2017) and is designed to support real-time personality assessment scenarios such as human-computer interaction, 
behavioral analytics, and AI-driven profiling applications.

---

## Dataset

### Raw Dataset (Original ChaLearn Data)

If you want to work with the original raw videos and annotations — without pre-generated feature files or processed frame splits — you can download the official ChaLearn dataset here:

**Because of limited resources first part of the dataset was downloaded for each training, validation, and testing**

[ChaLearn First Impressions V2 (CVPR'17)](https://chalearnlap.cvc.uab.cat/dataset/24/description/#)



### Preprocessed ChaLearn Dataset

All datasets, pre-processed files, trained models, and intermediate outputs used in this project are stored in the following Google Drive folder:

[Google Drive Folder](https://drive.google.com/drive/folders/1f7PbXNuwy5xxPK3mY4rh5SuphaLHwajv?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto)

#### Main Directory
`/Machine Learning-ProjectAK/`

```
Root: /Machine Learning-ProjectAK/

├── PersonalityBig5.ipynb
├── readme.txt
├── train-1.zip
├── val-1.zip
├── test-1e.zip
├── train-annotation-e.zip
├── val-annotation-e.zip
├── test-annotation-e.zip
│
├── shared_folder/
│   ├── transformer_attention_v1.keras
│   ├── personality_logs.csv
│   ├── annotations/
│   ├── train_audio_wav/
│   ├── val_audio_wav/
│   ├── test_audio_wav/
│   ├── train_frames_1p5s/
│   ├── val_frames_1p5s/
│   ├── test_frames_1p5s/
│   ├── audio_features_AST/
│   │   ├── training/
│   │   ├── validation/
│   │   └── test/
│   ├── videos_features_extraction/
│   │   ├── train_features_npy/
│   │   ├── val_features_npy/
│   │   └── test_features_npy/
│   ├── self_recorded_clips/
│   └── different_videos/
```
---

## Architecture

### System Pipeline

| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| 1 | **Video Input (15s clip)** | Raw MP4 Video | Frames + Audio |
| 2 | **Frame Sampling (1.5 FPS)** | Raw Video | Sampled RGB Frames |
| 3 | **Visual Feature Extraction (EfficientNetB0)** | Frames | Deep Visual Embeddings |
| 4 | **Visual Temporal Modeling (LSTM)** | Visual Embeddings | Visual Personality Representation |
| 5 | **Audio Extraction (WAV)** | Raw Video | .wav Audio |
| 6 | **Audio Feature Extraction (Mel-Spectrogram)** | WAV Audio | Spectrogram |
| 7 | **Audio Modeling (AST)** | Spectrogram | Audio Personality Representation |
| 8| **Multimodal Fusion (Transformer)** | Visual + Audio Embeddings | Joint Representation |
| 9 | **Regression Head** | Fused Representation | OCEAN Scores |
| 10 | **Real-Time Testing** | Predictions | personality_logs.csv / Visual personality traits |


## Model Architecture

The proposed framework follows a three-stage multimodal learning strategy:  
(1) Visual modeling, (2) Audio modeling, and (3) Cross-modal fusion.

### 1.Visual Stream

#### EfficientNetB0 Feature Extraction

- **Backbone**: `EfficientNetB0` (ImageNet pretrained)
- **Configuration**:
  - `include_top=False`
  - `pooling='avg'`
- **Input**: Sampled RGB frames (1.5 FPS from 15-second clips)
- **Output**: Global pooled deep feature vector per frame

EfficientNetB0 extracts high-level semantic and facial behavior features while reducing spatial dimensionality.  
Each frame is converted into a compact embedding suitable for temporal modeling.


#### LSTM Temporal Modeling

- **Input**: Sequence of EfficientNet embeddings
- **Architecture**: LSTM-based temporal network
- **Purpose**: Capture behavioral dynamics across time
- **Output**: Visual personality embedding

The LSTM models transitions in facial expressions, posture, and micro-movements to learn temporal behavioral patterns related to personality traits.


### 2.Audio Stream

#### Audio Preprocessing

- Extract `.wav` audio from video
- Convert audio into Mel-spectrogram representation using `librosa`

#### Audio Spectrogram Transformer (AST)

- **Input**: Mel-spectrogram
- **Architecture**: Transformer-based Audio Spectrogram Transformer
- **Purpose**: Learn temporal acoustic representations
- **Output**: Audio personality embedding

The AST captures vocal characteristics such as:
- Tone
- Pitch variation
- Speech rhythm
- Energy patterns

### 3.Multimodal Fusion

After unimodal training, the learned embeddings are fused.

- **Inputs**:
  - Visual embedding (LSTM output)
  - Audio embedding (AST output)
- **Fusion Mechanism**: Transformer-based cross-modal attention
- **Function**:
  - Align temporal representations
  - Learn inter-modal correlations
  - Dynamically weight modality importance
- **Output**: Unified multimodal representation

---

### Final Prediction Layer

The fused representation is passed to a regression head producing continuous predictions for the Big Five traits:

| Trait | Description |
|--------|------------|
| O | Openness |
| C | Conscientiousness |
| E | Extraversion |
| A | Agreeableness |
| N | Neuroticism |

Outputs are continuous values in the range [0, 1].


### Architectural Highlights

- Pretrained CNN backbone (transfer learning)
- Temporal sequence modeling via LSTM
- Transformer-based audio representation (AST)
- Cross-modal attention fusion
- Real-time inference and logging capability
---
## Performance

This section reports the quantitative performance of the three main stages of the personality analysis pipeline.
All metrics are computed on the final test set or validation set, depending on the stage objective.

#### Stage1: Video-Based Temporal Modeling (Visual Modality)

| Metric                        | Value      | Interpretation                                         |
| ----------------------------- | ---------- | ------------------------------------------------------ |
| **Mean Squared Error (MSE)**  | **0.0192** | Low overall regression error across personality traits |
| **Mean Absolute Error (MAE)** | **0.1103** | Average prediction deviation from ground truth         |


#### Stage1: Training Details

- **Input Shape**: `(10, 1280)`
- **Temporal Model**: LSTM (128 units)
- **Dense Layers**: 64 units (ReLU)
- **Regularization**:
  - Dropout: `0.3`
  - Recurrent Dropout: `0.2`
- **Batch Normalization**: Applied to input features
- **Optimizer**: Adam  `learning_rate = 1e-4`
- - **Epochs** : 50
- **Loss Function**: Mean Squared Error (MSE)


#### Stage2: Audio-Based Personality Estimation (AST)

| Metric                        | Value      | Interpretation                            |
| ----------------------------- | ---------- | ----------------------------------------- |
| **Mean Squared Error (MSE)**  | **0.0190** | Comparable performance to visual modality |
| **Mean Absolute Error (MAE)** | **0.1109** | Stable regression accuracy across traits  |


#### Stage2: Training Details

- **Input Shape**: `(128, 1500, 1)`
- **Patch Embedding**: Conv2D `(16x16)` with stride `(16x16)`
- **Attention Mechanism:**: Multi-Head Attention (4 heads, key_dim = 64)
- **Pooling**: GlobalAveragePooling1D
- **Dense Layers**: 128 units (ReLU)
- **Regularization**:
  - Dropout: `0.3`
  - SpecAugment (frequency + time masking)
- **Batch Normalization**: Applied to input features
- **Optimizer**: Adam  `learning_rate = 1e-4`
- - **Epochs** : 50
- **Loss Function**: Mean Squared Error (MSE)

#### Stage3: Multimodal Integration Strategy Selection (Audio + Video)

| Model               | Validation MAE |
| ------------------- | -------------- |
| **Transformer V1**  | **0.1055**     |
| Baseline            | 0.1066         |
| Full Transformer    | 0.1069         |
| Refined Transformer | 0.1076         |

#### Stage3: Training Details per Strategy

##### Baseline
* **Epoch** : 12
* **Dropout**: 0.3
* **Optimizer**: Adam  `learning_rate = 1e-4`

##### Transformer V1 (**Selected Model**)
* **Epoch** : 15
* **Dropout**: 0.4
* **Optimizer**: Adam  `learning_rate = 1e-4`

##### Refined Transformer
* **Epoch** : 25
* **Optimizer**: Adam  `learning_rate = 5e-5`
* **Dropout**: 0.5
* **L2Regularization**:  kernel_regularizer `l2(0.01)`

##### Full Transfromer (Self Attention + CrossModal Attention)
* **Epoch** : 20
* **Dropout**: 0.3
* **Optimizer**: Adam  `learning_rate = 5e-5`
---

## Requirements


### LIBRARY ROLES & PROJECT IMPACT

- **os & glob**  
  Automated the management of thousands of files across the shared folder system.

- **cv2 (OpenCV)**  
  Handled frame extraction such as `train_frames_1p5s`, allowing for visual behavior analysis.

- **librosa**  
  Transformed raw audio into Mel-spectrograms to identify vocal personality cues.

- **tensorflow & keras**  
  Used to build the Transformer-Attention architecture and perform real-time inference.

- **tqdm & gc**  
  Added progress tracking and managed memory to prevent crashes during large-scale feature extraction.

- **scipy & sklearn**  
  Used to calculate the Pearson Correlation and MAE to validate model accuracy.


### ENVIRONMENT SETUP
All required libraries were saved in requirements.txt

Copy and paste the following command into your terminal to install all required libraries:

```bash
pip install tensorflow pandas numpy opencv-python librosa matplotlib seaborn scikit-learn plotly tqdm scipy
```

---

## Project Structure

To ensure the code executes correctly, please maintain the following hierarchy in the root directory:

```
Root: /Machine Learning-ProjectAK/

# Main Jupyter Notebook
PersonalityBig5.ipynb

# Instruction file
readme.txt

# First Part of Dataset
train-1.zip
val-1.zip
test-1e.zip
train-annotation-e.zip
val-annotation-e.zip
test-annotation-e.zip

# Central processing hub
/shared_folder/

# The final trained model
/shared_folder/transformer_attention_v1.keras

# Output file for real-time predictions
/shared_folder/personality_logs.csv

# Contains .csv and .pkl sorted baseline files
/shared_folder/annotations/

# Extracted raw audio
/shared_folder/train_audio_wav/
 /val_audio_wav/
 /test_audio_wav/

# Sampled video frames 1.5 per seconds
/shared_folder/train_frames_1p5s/
 /val_frames_1p5s/
 /test_frames_1p5s/

# Extracted audio features
/shared_folder/audio_features_AST/
(Subfolders: /training, /validation, /test)

# Extracted video features
/shared_folder/videos_features_extraction/
(Subfolders: /train_features_npy, /val_features_npy, /test_features_npy)

# 15-second self-recorded videos for checking pretrained model
/shared_folder/self_recorded_clips/

/shared_folder/different_videos/
(Folder for real-time video testing / unseen data)
```

The project operates in four distinct stages:

### 4.1 Extraction
Raw videos are unzipped and stored in:
- `train_videos_extracted`
- `val_videos_extracted`
- `test_videos_extracted`

### 4.2 Preprocessing

**Visual:**  
Videos are sampled into frames stored in:
- `train_frames_1p5s`
- `val_frames_1p5s`
- `test_frames_1p5s`

**Audio:**  
Audio is stripped from videos and saved as `.wav` files in:
- `train_audio_wav`
- `val_audio_wav`
- `test_audio_wav`

### 4.3 Feature Extraction

**Audio:**  
Features are processed into the `audio_features_AST` directory (stored as `.npy` files for train, val, and test).

**Visual:**  
Visual features are saved as `.npy` files in the `videos_features_extraction` directory.

### 4.4 Real-Time Inference & Logging

- The model loads `transformer_attention_v1.keras`.
- **Testing:** Upload a 15-second video clip to `/shared_folder/different_videos/`.
- The system monitors this folder; once a video is detected, it predicts the OCEAN traits and automatically appends the results to `personality_logs.csv`.

---

## EXECUTION INSTRUCTIONS

1. **Open the Notebook**  
   Launch `PersonalityBig5.ipynb`.

2. **Configure Path**  
   Locate the `BASE_DIR` or `base_path` variable and update it to:

   ```python
   '/content/drive/MyDrive/Machine Learning-ProjectAK/shared_folder'
   ```

3. **Real-Time Test**  
   Upload a clip (ideally 15 seconds in duration) to:

   ```
   shared_folder/different_videos/
   ```

   The system will automatically process the clip and update the `personality_logs.csv` file.

---

## Future Work

- [x] EfficientNetB0 + LSTM visual modeling
- [x] Audio Spectrogram Transformer (AST) implementation
- [x] Transformer-based cross-modal fusion
- [x] Experiment with Late Fusion vs cross-modal strategies
- [x] Real-time inference and CSV logging
- [ ] Replace EfficientNetB0 with Vision Transformer (ViT) for spatial modeling
- [ ] Use full ChaLearn First Impressions V2
- [ ] Improve robustness against background noise
- [ ] Domain adaptation for real-world webcam videos
- [ ] Deployment using TensorRT optimization
- [ ] Edge deployment (e.g., NVIDIA Jetson)
- [ ] Hyperparameter search (Optuna / Bayesian Optimization)

---


## Contributors

- **Kimiakarbasi** - *Co-author* - [Github Profile](https://github.com/kimiakrs)
- **Andres Cabilon** - *Co-author* - [Github Profile](https://www.kaggle.com/andressabillon)
- **Civan Arda Ozel** - *Co-author* - [Github Profile](https://github.com/cardaozel)
  
---

## Disclaimer

This project is for research and educational purposes. The models are trained on specific datasets and may not generalize to all scenarios. For safety-critical applications, thorough testing and validation are required.

**If you find this project useful, please consider giving it a star!**

---

## Citation

If you use this project or dataset in your research, please cite:

```bibtex
@inproceedings{perez2017first,
  title={First Impressions: A Dataset for Apparent Personality Analysis},
  author={P{\'e}rez-Rosas, Ver{\'o}nica and Mihalcea, Rada and Narvaez, Alexis and Burzo, Mihai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}

```
----

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


