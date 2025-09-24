# AIPI540_Final_Project

by Haochen Li

---

## Topic

**Stock market prediction using environment embedding and transformer.**

---

## Themes

- Finance
- Technology

---

## Approaches

- **Naive approach:** Mean model
- **Classical ML approach:** Non-deep learning techniques
- **Deep learning approach:** Neural network-based (Transformer)

---

## Deep Learning Approach Specification

- **Input Data:** Tensor of shape `[batch_size, seq_len, input_dim]`, where `input_dim` includes features like high price, low price, and volume.
- **Output Predictions:** Tensor of shape `[batch_size, seq_len-1, input_dim]`, predicting the next vector for each timestep except the last during training.
- **Loss Function:** Mean Squared Error (MSELoss), suitable for regression tasks.

**Model specs**
The model is a transformer with a Mixture-of-Experts (MoE) style feedforward block, where a router dynamically combines outputs from multiple experts for each token in the sequence. This design allows for more flexible and powerful modeling of complex time series data.

In each transformer block, by the order of input to output, there are LayerNorm [${LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$],  multiheadattention layer, add&norm(with skip connection), LayerNorm, router & MoE layer (shared+specialized), and add&norm layer(with skip connection).

**Inference:**  
During inference or testing on unseen data, the `predict` function generates predictions without requiring target labels.

---

## Deliverables (AIPI540)

1. **Interactive Proof-of-Concept Application:**  
   - Publicly accessible via the internet for at least one week post-submission.
   - Any front-end framework is allowed (e.g., Flask, Streamlit, Dash).
   - Model deployment can use any cloud platform (GCP, Azure, AWS).

2. **10-Minute Presentation Video:**  
   - Submit as a video link (YouTube, WarpWire, Panopto).
   - Must include:
     - Problem statement
     - Data sources
     - Review of relevant literature/previous efforts
     - Model evaluation process & metric selection
     - Modeling approach
     - Data processing pipeline
     - Models evaluated & selected (at least 1 non-deep learning and 1 deep learning model)
     - Comparison to naive approach
     - Brief project demo
     - Results and conclusions
     - Ethics statement

3. **Demo Day "Pitch":**  
   - Max 3 minutes, max 3 slides.
   - Focus on demonstration for a broad audience:
     - Problem/motivation (1 slide)
     - Approach overview (1 slide)
     - Project demo
     - Results/conclusions/interesting findings (1 slide)
   - *Note: For public demo, avoid sharing proprietary information if you plan to commercialize.*

4. **Deployed Application Link:**  
   - Submit a working, publicly accessible web/mobile interface deployed on a cloud platform.

---

## Process for Replicating the Project

1. Clone the repository.
2. Create a virtual environment.
3. Install required libraries.
4. Run `make_dataset.py`.
5. Run `data_processing.py`.
6. **For NN model:**
   - Run `model_training_nn.py` to train the neural network (set `previous_model` to `None` for first training).
   - Run `model_test_nn.py` to test the NN model.
7. Run `model_test_naive.py` to evaluate the naive model (real-time training).
8. Run `model_test_ml.py` to evaluate the ML model (real-time training).

---

## Running the App

To launch locally:
```bash
streamlit run app.py
```
---

## Continuous Development Log

- Jun 6, 2026: 
   1. added model specs; ✅
   2. start training original model until convergence; 
   3. update the prediction functionality to include most recent macro data. 

- Next plan: try smaller time interals and seek for short term opportunities; prediction target shift to mean and variation for result interpretability.