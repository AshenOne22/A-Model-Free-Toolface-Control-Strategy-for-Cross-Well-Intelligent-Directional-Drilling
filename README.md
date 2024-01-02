# A-Model-Free-Toolface-Control-Strategy-for-Cross-Well-Intelligent-Directional-Drilling
This code belongs to research paper entitled as "A Model-Free Toolface Control Strategy for Cross-Well Intelligent Directional Drilling".

This paper proposes a model-free online learning-based adaptive decision approach to the directional drilling toolface, which enables self-study and applicability in complex downhole environments.

## Dependencies
Python 3.8

## Dataset
Due to the commercial confidentiality agreement with our collaborator, related real drilling data was removed from the repository. If necessary, please kindly contact us and we will then request the relevant authorizations from our collaborator

## Programs Description
### Exp 1: Multi-head attention LSTM
*Run Multi-head attention LSTM/compare.py*
1. Dataset Preparation
2. LSTM Model Architecture
3. Data Preprocessing
4. Model Training and Validation
5. Performance Evaluation
6. Further Analysis (comparsion)

### Exp 2: Improved DDPG training
*Run proposed improved DDPG/rl_ddpg_t2.py*
1. Environment Setup
2. DDPG Algorithm
3. Training
4. Exploration and Exploitation
5. Evaluation and Visualization
6. Saving Results and Model

## Folder Description
Directory ***Mutli-head attention LSTM*** contains code for the system model proposed in **Section 3.1**, and ***proposed improved DDPG*** contains code for the decision algorithm proposed in **Section 3.2**.

Directory ***Strategy Model Train*** corresponds to model encapsulation part in **Section 4.2.1**

Directory ***field exp toolface & torque*** corresponds to simulation analysis part (rl_compare) in **Section 4.1**, and field experimental verification part in **Section 4.2.2**

Directory ***field exp control param*** corresponds to field experimental verification part in **Section 4.2.2**.



