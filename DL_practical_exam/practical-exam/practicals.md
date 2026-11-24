# Deep Learning Lab

## 📊 Training Time Comparison Table

| PS # | Task | Model | Dataset | Epochs | Time/Epoch (GPU) | Total GPU Time | Total CPU Time | Params |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | Handwritten Digit Classification | CNN (Conv→Pool→Conv→Pool→Dense) | MNIST (60k images, 28×28) | 15 | ~2–3 s | **~30–45 s** | ~2–3 min | ~225K |
| **2** | Transfer Learning (Fine-tuning) | MobileNetV2 (frozen → unfrozen top 30) | Flower Photos (3,670 images) | 5 + 5 | ~17 s + ~19 s | **~3 min** | ~10–15 min | ~2.26M (transfer) |
| **3** | Text Generation (RNN) | SimpleRNN (256) + Embedding(64) | Shakespeare (200k chars) | 15 | ~11 s | **~2.5–3 min** | ~5–8 min | ~94K |
| **4** | Text Generation (LSTM) | LSTM (256) + Embedding(64) | Shakespeare (200k chars) | 15 | ~36 s | **~9 min** | ~20–30 min | ~340K |
| **5** | Facial Emotion Recognition | CNN (Conv→BN→Pool×3→Dense) | FER-2013 (28,709 train, 48×48) | 25 | ~27 s | **~11 min** | ~25–35 min | ~1.28M |
| **6** | Facial Emotion Recognition | CNN (Conv→Pool×3→Dense) | CK+48 (981 images, 48×48) | 40 | ~0.5 s | **~30 s** | ~2–3 min | ~683K |
| **7** | YOLO Classification (Flowers) | YOLOv8n-cls | Flowers (2,746 images, 5 classes) | 10 | ~7 s | **~1.5 min** | ~7 min (CPU) | ~1.44M |
| **8** | YOLO Object Detection (Fruits) | YOLOv8n (detect) | Fruits (320 train, 10 classes) | 30 | ~6–8 s | **~4–5 min** | ~30–45 min | ~3.2M |
| **9** | YOLO Classification (Fruits & Veg) | YOLOv8n-cls | Fruits & Veg (3,115 train, 36 classes) | 15 | ~5–7 s | **~2–3 min** | ~15–20 min | ~1.48M |
| **10** | Music Genre Classification (RNN) | SimpleRNN (128→64) | GTZAN (8,000 segments, MFCC) | 40 | ~5 s | **~3–4 min** | ~8–12 min | ~35K |
| **11** | Music Genre Classification (LSTM) | LSTM (128→64) | GTZAN (8,000 segments, MFCC) | 40 | ~6–7 s | **~4–5 min** | ~10–15 min | ~130K |

> **Note:** GPU times are approximate for NVIDIA T4 (15 GB). CPU times are for Intel i7-11800H. Actual times vary with batch size, image resolution, and system load.

---

## 📌 Key Points Per PS

### PS 1: Handwritten Digit Classification (MNIST CNN)
- **Task:** 10-class image classification on 28×28 grayscale digits.
- **Architecture:** Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(10, softmax).
- **Key Metric:** Test accuracy ~99.0–99.3%.
- **Important:** Use `sparse_categorical_crossentropy` with integer labels. Normalize pixels to [0,1].
- **Common Confusions:** 4↔9, 3↔5, 7↔2.

### PS 2: Transfer Learning with Fine-Tuning (MobileNetV2)
- **Task:** 5-class flower classification using pre-trained ImageNet weights.
- **Two-Stage Strategy:**
  - **Stage 1 (Feature Extraction):** Freeze entire MobileNetV2, train only the new Dense head (5 epochs, lr=1e-3).
  - **Stage 2 (Fine-Tuning):** Unfreeze top 30 layers, train with very small lr=1e-5 (5 epochs).
- **Key Metric:** Validation accuracy ~87–90%.
- **Important:** Use `keras.applications.mobilenet_v2.preprocess_input()` to scale pixels to [-1,1]. Use `GlobalAveragePooling2D` instead of Flatten to reduce parameters.
- **Why Two Stages?** Random head with large gradients would destroy pre-trained features if base is unfrozen immediately.

### PS 3: Text Generation using RNN (SimpleRNN)
- **Task:** Character-level text generation from Shakespeare corpus.
- **Architecture:** Embedding(64) → SimpleRNN(256) → Dense(vocab_size, softmax).
- **Input:** 40-character sliding windows (stride 3). Predict the 41st character.
- **Key Metric:** Validation accuracy ~46%, Perplexity ~6.7.
- **Important:** SimpleRNN suffers from vanishing gradients over long sequences. Temperature sampling (0.2, 0.5, 1.0) controls creativity vs. coherence.
- **Generation:** Autoregressive loop — feed predicted character back as input for next step.

### PS 4: Text Generation using LSTM
- **Task:** Same as PS3 but with LSTM to handle longer dependencies.
- **Architecture:** Embedding(64) → LSTM(256) → Dense(vocab_size, softmax).
- **Key Metric:** Validation accuracy ~47%, Perplexity ~6.8.
- **Important:** LSTM's gated cell state (forget, input, output gates) mitigates vanishing gradient. Training is ~3× slower than SimpleRNN due to 4× more parameters per unit.
- **Comparison with PS3:** LSTM produces slightly more coherent long-range text but trains slower.

### PS 5: Facial Emotion Recognition (FER-2013 CNN)
- **Task:** 7-class emotion classification (angry, disgust, fear, happy, neutral, sad, surprise) on 48×48 grayscale faces.
- **Architecture:** Conv2D(32→64→128) with BatchNormalization, MaxPooling, Dropout → Flatten → Dense(256) → Dense(7, softmax).
- **Key Metric:** Test accuracy ~54–55% (human accuracy is ~65%, SOTA ~76%).
- **Important:** Dataset is severely imbalanced (disgust: 436 vs happy: 7,215). Use `class_weight='balanced'` or augmentation. BatchNorm before ReLU stabilizes training.
- **Common Confusions:** fear↔sad, angry↔disgust.

### PS 6: Facial Emotion Recognition (CK+ CNN)
- **Task:** Same 7-class emotion classification but on CK+48 dataset (981 images).
- **Architecture:** Conv2D(32→64→128) → MaxPooling → Flatten → Dense(128) → Dense(7, softmax).
- **Key Metric:** Test accuracy ~91% (CK+ is lab-controlled, posed apex expressions).
- **Important:** CK+ is much easier than FER-2013 (controlled lighting, frontal faces, clear expressions). Only 981 images → use 80/20 split. Subject-independent split is ideal but not enforced here.
- **Why Higher Accuracy than PS5?** CK+ has clean, posed, apex-frame expressions vs. FER-2013's in-the-wild noisy images.

### PS 7: YOLO Classification on Flowers Dataset
- **Task:** 5-class flower classification using YOLOv8 in classification mode.
- **Dataset Catch:** Flowers dataset has NO bounding boxes → use `yolov8n-cls` (classification head), not detection.
- **Key Metric:** Top-1 accuracy ~95%, Top-5 = 100% (only 5 classes).
- **Important:** YOLO is a multi-task framework. With folder labels and no boxes, classification is the only valid task. Use `yolov8n-cls.pt` (ImageNet-pretrained).
- **Architecture:** C2f/CSP backbone → 1×1 conv to 1280 → Global Average Pooling → Linear(1280, 5).

### PS 8: YOLO Object Detection on Fruits Dataset
- **Task:** 10-class fruit detection with bounding boxes.
- **Dataset Bug:** All `.txt` label files use class ID `0`. True class is in filename (e.g., `01_Banana.jpg`). Notebook rewrites labels with correct class IDs.
- **Key Metric:** mAP@0.5 ~0.85–0.95, Precision/Recall ~0.90+.
- **Important:** This is TRUE object detection (boxes + classes). Use `yolov8n.pt` (COCO-pretrained). Transfer learning from COCO is essential with only 320 training images.
- **Common Confusions:** hog plum ↔ mango ↔ guava (similar greenish-yellow oval shapes).

### PS 9: YOLO Classification on Fruits & Vegetables Dataset
- **Task:** 36-class fruit/vegetable classification using YOLOv8-cls.
- **Dataset Catch:** 36 class folders, NO bounding boxes → classification only.
- **Key Metric:** Top-1 accuracy ~92–96%, Top-5 ~98–99%.
- **Important:** On Kaggle, dataset is mounted read-only. Copy to `/kaggle/working/` to avoid `[Errno 30]` errors. Use `cache='ram'` and `batch=128` to avoid GPU starvation from OneDrive/slow I/O.
- **Common Confusions:** Pepper family (bell pepper, capsicum, chilli, paprika) and leafy greens.

### PS 10: Music Genre Classification using RNN
- **Task:** 10-class audio genre classification using SimpleRNN on MFCC features.
- **Preprocessing:** Each 30s track → 10 segments of 3s → 13 MFCCs per frame → shape (130, 13) per segment.
- **Key Metric:** Segment-level accuracy ~55–65%, Song-level accuracy ~65–75%.
- **Important:** Split by SONG (not segment) to prevent data leakage. Normalize MFCCs with StandardScaler. Song-level prediction averages probabilities across 10 segments.
- **Common Confusions:** rock↔metal, country↔blues.

### PS 11: Music Genre Classification using LSTM
- **Task:** Same as PS10 but with LSTM for better long-range temporal modeling.
- **Architecture:** LSTM(128, return_sequences=True) → LSTM(64) → Dense(64) → Dense(10, softmax).
- **Key Metric:** Segment-level accuracy ~68–78%, Song-level accuracy ~78–88%.
- **Important:** LSTM outperforms SimpleRNN by ~10–13% due to gated cell state preserving rhythmic context over 130 time steps. Same song-level split and MFCC preprocessing as PS10.
- **Comparison with PS10:** LSTM is ~2× slower per epoch but significantly more accurate.

---

## 🎓 Viva Questions & Answers

### PS 1: MNIST CNN
**Q1. Why use CNN instead of Dense network for images?**
CNNs exploit spatial locality (nearby pixels are related) and weight sharing (same filter detects edges anywhere), reducing parameters and capturing hierarchical features (edges → shapes → digits).

**Q2. What is the purpose of MaxPooling?**
Reduces spatial dimensions (2×2 → half size), cuts computation, provides small translation invariance, and increases receptive field.

**Q3. Why use Dropout?**
Randomly zeroes neurons during training to prevent co-adaptation and overfitting. Disabled at inference.

**Q4. Why is test accuracy ~99% but not 100%?**
Some digits are genuinely ambiguous (4 vs 9, 3 vs 5, 7 vs 2) due to handwriting variation. ~99.2% is near the practical ceiling.

**Q5. What is sparse_categorical_crossentropy?**
Loss function for integer-labeled multi-class classification. Computes `-log(p_true_class)`. "Sparse" means labels are integers (0–9), not one-hot vectors.

---

### PS 2: Transfer Learning
**Q1. What is transfer learning and why use it here?**
Reusing a model trained on a large source task (ImageNet, 1.2M images) as a starting point for a related target task (flowers). With only 3,670 images, training from scratch would overfit.

**Q2. Why two stages (feature extraction → fine-tuning)?**
Stage 1: Head is randomly initialized. If base were trainable, large error gradients would destroy pre-trained weights. Stage 2: Once head is sensible, unfreeze top layers with tiny lr (1e-5) to adapt high-level features without catastrophic forgetting.

**Q3. Why use GlobalAveragePooling2D instead of Flatten?**
GAP averages each feature map to one number (zero parameters), is spatially robust, and prevents overfitting. Flatten would create 25,088 features → Dense layer needs 6.4M parameters.

**Q4. What does `base.trainable = False` do?**
Freezes all weights in the base model. Gradients do not flow through it, so only the new head trains.

**Q5. Why use `preprocess_input()` for MobileNetV2?**
Scales pixels from [0,255] to [-1,1], matching the normalization used during ImageNet pre-training. Skipping this drastically reduces accuracy.

---

### PS 3: RNN Text Generation
**Q1. What is the vanishing gradient problem in SimpleRNN?**
During BPTT, gradients are multiplied by the recurrent weight matrix at every time step. If eigenvalues < 1, gradients decay exponentially, making long-range dependencies unlearnable.

**Q2. What is temperature sampling?**
Rescaling probabilities as `p_i^(1/T)`. T < 1 sharpens (conservative, repetitive). T > 1 flattens (creative, error-prone). T → 0 becomes greedy argmax.

**Q3. What is the shape of the input tensor?**
(batch_size, 40, 64) after embedding. 40 time steps (characters), 64-dim embedding per character.

**Q4. Why use character-level instead of word-level?**
Tiny vocabulary (~37 chars vs 10k+ words), no OOV problem, learns spelling. Trade-off: longer sequences, weaker semantics.

**Q5. What is teacher forcing?**
During training, the true previous character is always fed as input (not the model's own prediction). Speeds up convergence but creates train/inference mismatch (exposure bias).

---

### PS 4: LSTM Text Generation
**Q1. How does LSTM solve the vanishing gradient problem?**
LSTM adds a cell state `c_t` with additive updates: `c_t = f_t * c_{t-1} + i_t * c_tilde`. This creates a "gradient highway" where gradients flow unchanged across many steps, unlike SimpleRNN's multiplicative decay.

**Q2. Name the three LSTM gates and their roles.**
- **Forget gate (f_t):** Decides how much old memory to keep.
- **Input gate (i_t):** Decides how much new content to write.
- **Output gate (o_t):** Decides how much memory to expose as hidden state.

**Q3. Why is LSTM slower than SimpleRNN?**
LSTM has 4× the parameters per unit (4 gate weight matrices vs 1). Training takes ~3× longer but captures longer dependencies.

**Q4. What is the difference between cell state and hidden state?**
Cell state (`c_t`) is the internal long-term memory highway. Hidden state (`h_t = o_t * tanh(c_t)`) is the gated, externally visible summary passed to the next step/layer.

**Q5. Why does LSTM produce more coherent text than SimpleRNN?**
LSTM's gated memory preserves context over 40 characters, allowing it to maintain speaker names, line structure, and grammatical patterns better than SimpleRNN.

---

### PS 5: FER-2013 Emotion CNN
**Q1. Why is accuracy only ~55% on FER-2013?**
FER-2013 labels are noisy (crowd-sourced), expressions are ambiguous, and human accuracy is only ~65%. SOTA is ~76%. A well-trained CNN at 55% is reasonable.

**Q2. Why is the dataset imbalanced and how do you handle it?**
Disgust has 436 images vs happy's 7,215 (16.5× ratio). Use `class_weight='balanced'` to give rare classes higher loss weight, or oversample/augment minority classes.

**Q3. Why use BatchNormalization?**
Normalizes activations per channel, stabilizes training, allows higher learning rates, and acts as mild regularization. Placed BEFORE ReLU (Conv → BN → ReLU).

**Q4. Which emotions are most confused and why?**
fear↔sad (both involve downturned mouth, lowered brows), angry↔disgust (both involve scrunched nose/brows). Happy and surprise are easiest (distinctive open mouth/raised brows).

**Q5. Why is FER-2013 harder than CK+?**
FER-2013 is in-the-wild (variable lighting, pose, occlusion, noisy labels). CK+ is lab-controlled (frontal, posed, apex frames, clean labels).

---

### PS 6: CK+ Emotion CNN
**Q1. Why is CK+ accuracy ~91% vs FER-2013's ~55%?**
CK+ is lab-controlled: frontal faces, posed apex expressions, clean expert labels, aligned crops. FER-2013 is in-the-wild with noisy crowd labels.

**Q2. What is subject leakage and why does it matter?**
CK+ has 3 frames per sequence from the same person. Random image-level split puts near-duplicate frames of the SAME person in train and test, inflating accuracy. Subject-independent split is the correct protocol.

**Q3. Why is contempt the hardest class?**
Only 54 images (vs 249 for surprise). Subtle expression (one-sided mouth raise), easily confused with neutral or angry.

**Q4. What is the CK+48 dataset?**
The last 3 frames of each of 327 labeled sequences → 981 images, already cropped/aligned to 48×48 grayscale.

**Q5. Why use only 40 epochs with such a small dataset?**
With 785 training images, the model converges quickly. More epochs risk overfitting. Early stopping (patience=5) prevents this.

---

### PS 7: YOLO Flowers (Classification)
**Q1. The PS says "object detection" — why classification?**
The flowers dataset has only folder labels, NO bounding boxes. Detection cannot be supervised without boxes. YOLO is multi-task; we use its classification head (`yolov8n-cls`).

**Q2. Is YOLO only a detector?**
No. Ultralytics YOLOv8 supports detect, segment, classify, pose, and obb tasks with a shared backbone. The weights file and data format select the task.

**Q3. What is the architecture of YOLOv8n-cls?**
C2f/CSP nano backbone → 1×1 conv to 1280 channels → Global Average Pooling → Dropout → Linear(1280, num_classes). ~1.44M parameters.

**Q4. Why is top-5 accuracy exactly 1.0?**
With only 5 classes, the true class is ALWAYS among the top 5 predictions. Top-1 is the meaningful metric.

**Q5. What is Global Average Pooling and why use it?**
Averages each feature map to one number. Zero parameters, spatially robust, strong regularizer. Prevents overfitting on small datasets.

---

### PS 8: YOLO Fruit Detection
**Q1. What was wrong with the original dataset labels?**
Every `.txt` file used class ID `0`. Training on them would produce a 1-class "generic fruit" detector. The true class was in the filename (e.g., `01_Banana.jpg`).

**Q2. How does this notebook fix it?**
Scans all filenames, extracts class name with regex, builds a 10-class vocabulary, and rewrites every label file replacing `0` with the correct class ID.

**Q3. Define IoU, mAP50, and mAP50-95.**
- **IoU:** Intersection/Union of predicted and GT boxes.
- **mAP50:** Mean Average Precision at IoU ≥ 0.5.
- **mAP50-95:** Mean AP over IoU thresholds 0.50 to 0.95 in 0.05 steps (stricter, COCO primary metric).

**Q4. What is NMS and why is it needed?**
Non-Maximum Suppression removes duplicate overlapping predictions for the same object. Keeps the highest-confidence box, drops same-class boxes with IoU above a threshold.

**Q5. Which classes confuse most and why?**
Hog plum, mango, and guava — similar greenish-yellow color and oval silhouette at this resolution.

---

### PS 9: YOLO Fruits & Vegetables (Classification)
**Q1. Why classification and not detection?**
Dataset has 36 class folders but NO bounding boxes. Detection requires boxes. Classification is the only valid supervision.

**Q2. How do you fix the Kaggle "Read-only file system" error?**
Kaggle mounts `/kaggle/input/` as read-only. Ultralytics tries to write cache files there. Fix: copy dataset to writable `/kaggle/working/` before training.

**Q3. Why was training slow on your local machine?**
Project was in a OneDrive folder. Cloud sync locks files, destroying I/O and starving the GPU. Fix: `cache='ram'` loads dataset into RAM, bypassing disk I/O.

**Q4. What does `cache='ram'` do?**
Loads all images (~150MB) into System RAM on Epoch 1. Subsequent epochs read directly from RAM, completely bypassing slow disk/cloud sync.

**Q5. Why increase batch size to 128?**
A batch of 16 is too small for a 15GB GPU. Batch 128 saturates VRAM, enabling larger parallel matrix multiplications and faster epochs.

---

### PS 10: RNN Music Genre Classification
**Q1. Why split by song instead of segment?**
To prevent data leakage. Consecutive 3-second segments of the same song are highly correlated. Splitting by segment would put them in both train and test, inflating accuracy.

**Q2. What are MFCCs and why use them?**
Mel-Frequency Cepstral Coefficients represent the short-term power spectrum on a perceptual Mel scale. They capture timbre/instrumentation (the "texture") while ignoring pitch — ideal for genre classification.

**Q3. Why use 13 MFCCs?**
The first 13 coefficients capture the overall spectral envelope (timbre). Higher coefficients represent fine spectral details/noise, less relevant for broad genre classification.

**Q4. How does song-level prediction work?**
Extract 10 segments from the 30s track, predict probabilities for each, average them (mean pooling). The genre with the highest average probability is chosen.

**Q5. Which genres confuse most and why?**
Rock↔Metal (similar distorted guitars/drums), Country↔Blues (similar acoustic instrumentation/chord progressions). Classical is rarely confused (no percussion/electronic elements).

---

### PS 11: LSTM Music Genre Classification
**Q1. Why use LSTM instead of SimpleRNN for audio?**
130 time steps is long enough to cause vanishing gradients in SimpleRNN. LSTM's gated cell state preserves rhythmic and temporal context over the entire 3-second clip.

**Q2. What is the shape of the input tensor?**
(batch_size, 130, 13). 130 time steps (frames) per 3-second segment, 13 MFCC features per frame.

**Q3. Why is the first LSTM set to `return_sequences=True`?**
Outputs hidden state at every time step (130 vectors), allowing the second LSTM to process temporal evolution at a higher level of abstraction.

**Q4. How much better is LSTM than SimpleRNN here?**
LSTM achieves ~68–78% segment-level accuracy vs SimpleRNN's ~55–65% — a ~10–13% improvement due to better long-range dependency modeling.

**Q5. What is the computational complexity of LSTM vs SimpleRNN?**
LSTM has ~4× the parameters (4 gate matrices vs 1). Slower to train but significantly more accurate on sequences > 50 steps.

---

Yes, to make your reference material a **complete "Survival Guide"** for the lab exam and viva, you should add sections on **Core Mathematical Formulas**, **Troubleshooting Common Errors**, and **Hyperparameter Tuning**. Examiners love asking about the math behind the layers and how to debug a failing model.

Here is the additional content. You can append this directly to the bottom of your existing Markdown file:

---

## 🧮 Key Mathematical Formulas (Viva Favorites)

**1. Convolution Output Size**
Used to calculate the spatial dimensions after a Conv2D or MaxPool layer:
$$Output = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
*Where: $W$ = Input size, $K$ = Kernel/Filter size, $P$ = Padding, $S$ = Stride.*

**2. Softmax Function (Output Layer)**
Converts raw logits ($z$) into a probability distribution summing to 1:
$$\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**3. Categorical Cross-Entropy Loss**
The standard loss function for multi-class classification:
$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
*Where $y_i$ is the true one-hot label (or 1 for sparse), and $\hat{y}_i$ is the predicted probability.*

**4. Intersection over Union (IoU) - Object Detection**
Measures the overlap between the Predicted Bounding Box ($A$) and Ground Truth ($B$):
$$IoU = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$

**5. Precision, Recall, and F1-Score**
* **Precision:** $\frac{TP}{TP + FP}$ *(Out of all predicted positives, how many were correct?)*
* **Recall:** $\frac{TP}{TP + FN}$ *(Out of all actual positives, how many did we find?)*
* **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

**6. LSTM Gate Equations**
* **Forget Gate:** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
* **Input Gate:** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
* **Candidate Cell:** $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
* **Cell State Update:** $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ *(The "Gradient Highway")*
* **Output Gate:** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
* **Hidden State:** $h_t = o_t \odot \tanh(C_t)$

---

## 🛠️ Troubleshooting Common Lab Errors

| Error / Symptom | Probable Cause | How to Fix It |
| :--- | :--- | :--- |
| **`CUDA Out of Memory (OOM)`** | Batch size too large for GPU VRAM. | Reduce `batch_size` (e.g., 64 $\to$ 16). Use `cache='ram'` in YOLO. |
| **Loss is `NaN` or `Inf`** | Learning rate too high, or exploding gradients. | Lower learning rate (e.g., $1e-3 \to 1e-4$). Add Gradient Clipping. Check for $\log(0)$ in custom loss. |
| **Accuracy stuck at random chance** *(e.g., ~10% for 10 classes)* | Forgot to normalize images, or wrong loss function. | Divide images by `255.0`. Ensure you are using `CrossEntropy` and not `MSE`. |
| **`ValueError: Expected 4D tensor`** | Shape mismatch in CNN. | CNN expects `(batch, height, width, channels)`. Add `np.expand_dims` or `tf.expand_dims` if missing batch dim. |
| **`ValueError: Expected 3D tensor`** | Shape mismatch in RNN/LSTM. | RNN expects `(batch, time_steps, features)`. Ensure MFCC/Text data is shaped correctly. |
| **Model memorizes training data (100% Train Acc, 40% Val Acc)** | Severe Overfitting. | Add `Dropout`, use Data Augmentation, apply `EarlyStopping`, or simplify the model. |
| **Ultralytics `Read-only file system`** | Running on Kaggle mount or OneDrive. | Copy dataset to a writable directory (`/kaggle/working/` or `C:\datasets\`). |

---

## 🎛️ Hyperparameter Tuning Cheat Sheet

| Hyperparameter | Effect of Increasing It | Rule of Thumb |
| :--- | :--- | :--- |
| **Learning Rate** | Faster convergence, but risks overshooting minima or causing `NaN` loss. | Start with `1e-3` for Adam. Use `ReduceLROnPlateau` to decay it. |
| **Batch Size** | Smoother gradients, faster epoch time, but requires more VRAM. | 32 or 64 for standard CNNs. 128+ for YOLO/Transfer Learning. |
| **Epochs** | More training time. Too many leads to overfitting. | Set high (e.g., 100) but **always** use `EarlyStopping(patience=5 to 10)`. |
| **Dropout Rate** | Stronger regularization. Too high causes underfitting. | `0.2` to `0.3` for Conv layers. `0.4` to `0.5` for Dense layers. |
| **Optimizer** | **Adam**: Fast, adaptive (best default). **SGD + Momentum**: Slower, but often generalizes slightly better on large datasets. | Use **Adam** for almost all lab practicals unless specified otherwise. |

---

## 📊 Overfitting vs. Underfitting Diagnostics

Examiners will often show you a Loss/Accuracy graph and ask you to diagnose it.

* **Overfitting (High Variance):**
  * **Symptoms:** Training loss keeps decreasing, but Validation loss starts *increasing* after a certain epoch. Train Accuracy $\gg$ Val Accuracy.
  * **Cures:** More data, Data Augmentation, Dropout, L2 Regularization (Weight Decay), Early Stopping, reduce model complexity.
* **Underfitting (High Bias):**
  * **Symptoms:** Both Training and Validation loss are high and plateau early. Neither accuracy reaches a good number.
  * **Cures:** Make the model deeper/wider, train for more epochs, increase learning rate, remove Dropout/Regularization, add better features.

---

## 🗂️ Dataset Quick-Recall Table
*(Memorize this to instantly answer "What dataset did you use and what are its characteristics?")*

| Dataset | Domain | Size / Shape | Classes | Key Characteristic / Catch |
| :--- | :--- | :--- | :--- | :--- |
| **MNIST** | Vision | 70k images, 28x28x1 | 10 (Digits) | The "Hello World" of DL. Very clean, high accuracy (~99%). |
| **Oxford Flowers** | Vision | ~8k images, RGB | 5 or 102 | Used for Transfer Learning. |
| **Shakespeare** | NLP | ~1M characters | ~38 chars | Used for RNN/LSTM character generation. |
| **FER-2013** | Vision | ~35k images, 48x48x1 | 7 Emotions | In-the-wild, noisy labels, highly imbalanced (Disgust is rare). Human acc is ~65%. |
| **CK+ (CK+48)** | Vision | 981 images, 48x48x1 | 7 Emotions | Lab-controlled, posed apex frames. Much easier than FER-2013 (~90%+ acc). |
| **GTZAN** | Audio | 1000 tracks, 30s each | 10 Genres | Requires MFCC feature extraction. Watch out for song-level data leakage! |
| **Fruits & Veg** | Vision | ~3.8k images, RGB | 36 | Folder-based labels. No bounding boxes (Classification only). |
| **Kaggle Fruits** | Vision | 400 images, RGB | 10 | **Catch:** Original `.txt` labels are all `0`. Must extract class from filename. |