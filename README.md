# Project: Real-time Multimodal Personality Analysis (Big Five - OCEAN)

## Dataset
**ChaLearn First Impressions V2 (CVPR'17)**  
https://chalearnlap.cvc.uab.cat/dataset/24/description/#

---

## Main Directory
`/Machine Learning-ProjectAK/`

---

## 1. PROJECT STRUCTURE & DIRECTORY MAP

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

---

## 2. ENVIRONMENT SETUP

Copy and paste the following command into your terminal to install all required libraries:

```bash
pip install tensorflow pandas numpy opencv-python librosa matplotlib seaborn scikit-learn plotly tqdm scipy
```

---

## 3. LIBRARY ROLES & PROJECT IMPACT

- **os & glob**  
  Automated the management of thousands of files across the shared folder system.

- **cv2 (OpenCV)**  
  Handled frame extraction for `train_frames_1p5s`, allowing for visual behavior analysis.

- **librosa**  
  Transformed raw audio into Mel-spectrograms to identify vocal personality cues.

- **tensorflow & keras**  
  Used to build the Transformer-Attention architecture and perform real-time inference.

- **tqdm & gc**  
  Added progress tracking and managed memory to prevent crashes during large-scale feature extraction.

- **scipy & sklearn**  
  Used to calculate the Pearson Correlation and MAE to validate model accuracy.

---

## 4. DATA PIPELINE PROCESS

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

## 5. EXECUTION INSTRUCTIONS

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
