# A-Model-Free-Toolface-Control-Strategy-for-Cross-Well-Intelligent-Directional-Drilling
This code belongs to research paper entitled as "A Model-Free Toolface Control Strategy for Cross-Well Intelligent Directional Drilling".

This paper proposes a model-free online learning-based adaptive decision approach to the directional drilling toolface, which enables self-study and applicability in complex downhole environments.

## Dependencies
Python 3.8

## Dataset
Due to the commercial confidentiality agreement with our collaborator, related real drilling data was removed from the repository. If the data is necessary for the review phase, please kindly contact us and we will then request the relevant authorizations from our collaborator.

## Programs Description
### EXP 1: Multi-head attention LSTM training
*Run* [Multi-head attention LSTM/compare.py](https://github.com/AshenOne22/A-Model-Free-Toolface-Control-Strategy-for-Cross-Well-Intelligent-Directional-Drilling/blob/main/Mutli-head%20attention%20LSTM/compare.py)
1. Dataset Preparation
2. LSTM Model Architecture
3. Data Preprocessing
4. Model Training and Validation
5. Performance Evaluation
6. Further Analysis (comparsion with original LSTM, RNN)

### EXP 2: Improved DDPG training
*Run* [proposed improved DDPG/rl_ddpg_t2.py](https://github.com/AshenOne22/A-Model-Free-Toolface-Control-Strategy-for-Cross-Well-Intelligent-Directional-Drilling/blob/main/proposed%20improved%20DDPG/rl_ddpg_t2.py)
1. Environment Setup
2. DDPG Algorithm
3. Training
4. Exploration and Exploitation
5. Evaluation and Visualization
6. Saving Results and Model

## Simulation Schemes
### EXP 3: Migrated Model Test
*Run* [Multi-head attention LSTM/lstm_time1_out3_t9.py](https://github.com/AshenOne22/A-Model-Free-Toolface-Control-Strategy-for-Cross-Well-Intelligent-Directional-Drilling/blob/main/Mutli-head%20attention%20LSTM/lstm_time1_out3_t9.py)
**Objective**: Evaluate the performance of control parameter predict of the initial model

### EXP 4: Simulation Analysis for Decision Algorithm Performance
*Run* [field exp toolface & torque/rl_compare.py](https://github.com/AshenOne22/A-Model-Free-Toolface-Control-Strategy-for-Cross-Well-Intelligent-Directional-Drilling/blob/main/field%20exp%20toolface%20%26%20torque/rl_compare.py)

**Objective**: Evaluate and compare the performance of different reinforcement learning algorithms including Policy Gradient (PG), Deep Deterministic Policy Gradient (DDPG), and our improved DDPG.
