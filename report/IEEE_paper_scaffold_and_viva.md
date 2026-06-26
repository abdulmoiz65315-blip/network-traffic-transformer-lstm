# IEEE Paper Scaffold + Viva Prep
### Network Traffic Classification Using Transformer-Enhanced LSTM Models

> **Read this first.** Your CEP brief states the report must be written *in your own
> words* and that AI-generated writing in the final paper means a zero. So this file
> is a **scaffold**: it tells you what each section must contain and where to paste
> *your own* numbers from `results/metrics.txt`. Do **not** paste these sentences
> verbatim into the IEEE template — rewrite every paragraph yourself. Fill every
> `[YOUR RESULT]` from your actual run.

---

## Part A — IEEE Paper Section Guide

### Abstract (~200 words)
Cover, in order: the problem (classifying Normal/VPN/Tor/encrypted/application
traffic), why it matters (security, QoS, encrypted-traffic blind spots), what you
built (Transformer-Enhanced LSTM = BiLSTM + multi-head self-attention), the
datasets (CIC-Darknet2020 + secondary), and your headline result
([YOUR ACCURACY], [YOUR F1]) versus the best baseline. End with one sentence on the
contribution.

### I. Introduction
Motivate automated traffic classification; note the shift from port-based and DPI
methods (broken by encryption and dynamic ports) toward ML/DL. State that LSTM
captures sequence behaviour while attention captures global feature relationships,
motivating the hybrid. List your contributions as 3–4 bullets.

### II. Problem Statement
One precise paragraph: given flow-level features, assign each flow to a traffic
category; the challenge is encrypted traffic, class imbalance, and generalisation.

### III. Literature Review / Related Work
Summarise the four base papers **in your own words** and cite them:
1. *Network Traffic Classification: Techniques, Datasets and Challenges* — taxonomy
   of port-based, DPI, ML and DL methods; dataset limitations.
2. *Towards the Deployment of ML Solutions in NTC: A Systematic Survey* —
   pipelines, feature engineering, evaluation, deployment.
3. *Efficient Dark Web Traffic Classification using a Hybrid CNN-LSTM Model* —
   your closest reference; note you do **not** copy it (you use attention, not CNN).
4. *Robust Network Traffic Classification* — robustness, zero-day, stability.
End with the **research gap**: prior work used CNN / LSTM / CNN-LSTM /
Transformer-only separately; you combine BiLSTM sequence learning with a
Transformer self-attention encoder.

### IV. Dataset Description
Describe CIC-Darknet2020 (flow features from CICFlowMeter; categories Non-Tor,
NonVPN, Tor, VPN; application sub-labels) and your secondary dataset. State row
counts **after cleaning** and the class distribution (from the EDA cell).

### V. Methodology
Walk through the pipeline exactly as in the notebook: cleaning → label encoding →
stratified 80/10/10 split → standard scaling → SMOTE on train only. Justify each
choice (e.g. SMOTE on train only to avoid test leakage).

### VI. Model Architecture
Reproduce the architecture and explain each block:
`Dense embedding → BiLSTM(128) → BiLSTM(64) → MultiHeadAttention(4 heads) →
residual + LayerNorm → FFN → residual + LayerNorm → GlobalAvgPool → Dropout →
Dense → Softmax`. Explain *why* attention helps (long-range feature dependencies)
and why bidirectional LSTM (context from both directions).

### VII. Experimental Setup
Colab + GPU, TensorFlow [version], Adam (lr=1e-3), sparse categorical cross-entropy,
batch 256, up to 40 epochs, EarlyStopping(patience=6), ReduceLROnPlateau,
ModelCheckpoint, seed=42. Mention baselines: RF, DT, LR, SVM, KNN, XGBoost, LSTM.

### VIII. Results
Paste the comparison table from `results/comparison_table.csv`. Report the proposed
model: Accuracy [YOUR RESULT], Precision [YOUR RESULT], Recall [YOUR RESULT],
F1 [YOUR RESULT], ROC-AUC [YOUR RESULT]. Include confusion-matrix, accuracy/loss
curve, and ROC figures from `figures/`. Add the cross-dataset table.

### IX. Discussion
Which baseline was strongest and why (tree ensembles usually shine on flow
features)? Did attention beat the plain LSTM? Which classes get confused (read the
confusion matrix — VPN vs Tor is a common one)? Be honest if a baseline beat the
proposed model — explain it (small per-class data, tabular nature).

### X. Conclusion & Future Work
Summarise contribution and result. Future: real packet-sequence windows (true
timesteps), CNN front-end, SHAP/LIME explainability, real-time deployment.

### References
Use IEEE numbered style for the four base papers, the dataset sources, and the
TensorFlow/scikit-learn/XGBoost tools. (Wikipedia and ChatGPT are not valid
sources — per your brief.)

---

## Part B — Viva Questions & Answers

**Q: Why combine LSTM with a Transformer instead of using either alone?**
LSTM models sequential/temporal dependencies through its gates but attends to
context only locally and sequentially; self-attention computes relationships
between *all* feature positions at once (global context) but has no inherent notion
of order. Combining them gives both sequential memory and global feature
interaction.

**Q: What is multi-head self-attention doing here?**
Each head learns a different projection of the input and computes scaled
dot-product attention (softmax(QKᵀ/√d)V). Multiple heads let the model attend to
different feature-relationship patterns simultaneously; outputs are concatenated and
projected back.

**Q: Why the residual connections and layer normalization?**
Residuals (`Add`) let gradients flow past the attention/FFN blocks, preventing
vanishing gradients and letting the block learn a refinement rather than a full
transform. LayerNorm stabilises activations per-sample, speeding convergence.

**Q: Why did you apply SMOTE only to the training set?**
Applying it before the split (or to test) leaks synthetic information into
evaluation and inflates metrics. SMOTE on train only keeps the test set a faithful
sample of real traffic.

**Q: Your data has timesteps=1. Is this really sequential modelling?**
Honest answer: each flow is fed as a single timestep, so the LSTM here mainly learns
feature interactions rather than true packet-level temporal order. Real temporal
modelling would window consecutive packets/flows — that's stated as future work.
(Examiners respect this honesty more than a hand-wave.)

**Q: Why standardize features?**
Flow features span very different scales (bytes vs durations vs ports). Standard
scaling (zero mean, unit variance) keeps gradient-based training stable and stops
large-magnitude features from dominating.

**Q: How do you prevent overfitting?**
Dropout (0.2–0.3), EarlyStopping on val loss with best-weight restore,
ReduceLROnPlateau, and a held-out validation split separate from test.

**Q: What's the research gap you fill?**
Prior dark-web/traffic work used CNN, LSTM, CNN-LSTM or Transformer-only in
isolation. We pair a *bidirectional* LSTM stack with a Transformer self-attention
encoder — sequential learning plus global attention — which (per your results) is
[competitive with / better than] the baselines.

**Q: Why two datasets if you don't merge them?**
Their label schemes differ, so merging would create meaningless labels. Running the
same model on each separately tests generalisation and satisfies the comparison
requirement honestly.

**Q: What are the limitations?**
Single-timestep framing, SMOTE synthetic artefacts, dependence on label
granularity, and results tied to CICFlowMeter feature extraction.

**Q: Walk me through one prediction.**
Raw flow → cleaned & scaled feature vector → reshaped (1, n_features) → Dense
embedding → BiLSTM layers → attention encoder → pooled vector → softmax over
classes → argmax is the predicted traffic type.
