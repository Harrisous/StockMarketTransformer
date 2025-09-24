# Stock Market Transformer (Developed From AIPI540_Final_Project)
# by Haochen Li

## Topic
**Stock market prediction using environment embedding and transformer.**  
The idea is first to encode stock parameters (prices, volumes), market parameters (indices), economic parameters into vector using autoencoder. Then a transformer with MoE and multi-head attention can be trained based on time series of encoded vectors. The aim is to predict the future stock price(s), and the loss metrics in this project is the MSE. A streamlit site can be built using the trained model and provide visualization for predictions. For detail please see below     
Themes:
- Finance & Technology
- Trading
- Time Series
- Transformer

## Approaches

- **Naive approach:** Mean model
- **Classical ML approach:** Non-deep learning techniques
- **Deep learning approach:** Neural network-based (Transformer) (main focus)

---

## Deep Learning Approach Specification

- **Input Data:** Tensor of shape `[batch_size, seq_len, input_dim]`, where `input_dim` includes features like high price, low price, and volume.
- **Output Predictions:** Tensor of shape `[batch_size, seq_len-1, input_dim]`, predicting the next vector for each timestep except the last during training.
- **Loss Function:** Mean Squared Error (MSELoss), suitable for regression tasks.

**Model specs**
The model is a transformer with a Mixture-of-Experts (MoE) style feedforward block, where a router dynamically combines outputs from multiple experts for each token in the sequence. This design allows for more flexible and powerful modeling of complex time series data.

In each transformer block, by the order of input to output, there are LayerNorm ${LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$,  multiheadattention layer, add&norm(with skip connection), LayerNorm, router & MoE layer (shared+specialized), and add&norm layer(with skip connection).

**Inference:**  
During inference or testing on unseen data, the `predict` function generates predictions without requiring target labels.

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


## Have a try

Online streamlit demo: https://harry-demo.streamlit.app/ (based on not converged version)  
10-min introduction video: https://youtu.be/nIk1aV_UYc0 (based on not converged version)  

To launch locally:  
1. clone the repo
2. create and activate `.venv`
(optional): substitute `./app_models/transformer_model.pth` with latest model (see below) and rename to `transformer_model.pth` to run the converged model. 
3. run the app using the following:
```bash
streamlit run app.py
```

## Latest Content Update (Jun 09):  

Latest model (pth, converged): `./models/checkpoints/transformer_model_checkpoint_epoch_63.pth`  
Latest training loss log: `./models/nn_model_loss.log` or `./scripts/results/nn_train_loss.txt`  
Result: converged loss is still huge, and the backtesting result is only good for open percentage change. Should consider using simpler model and expand training data(without MoE).

---

## Development Log
- Apr 22, 2025
   1. Build initial model ✅ (not converged version)
   2. Submit demo app and video for AIPI 540 ✅ 

- Jun 6, 2025
   1. added model specs; ✅
   2. start training original model until convergence; ✅

- Jun 09, 2025
   1. Get converged result ✅
   3. update the prediction functionality to include most recent macro data. ✅ (applied to app.py only since it will pull most recent financial data)


- Future plan: evaluation; try smaller time interals and seek for short term opportunities; prediction target shift to mean and variation for result interpretability.
