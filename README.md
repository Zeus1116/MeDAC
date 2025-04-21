## MeDAC

<a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache2.0-a5fed.svg"></a> 

This repository provides the tool (i.e., MeDAC) and all experimental data for our work: "Accelerating Deep Learning Compiler Testing via
Message Passing Neural Network with Attention", which has been submitted to IEEE Transactions on Reliability. 

### File Structure
The code repository's root directory contains four subfolders, each corresponding to one of the four test scenarios mentioned in the paper.
Each project directory includes the following main items:
* **NNSmith**: The NNSmith folder contains the original NNSmith code, along with our implementation for logging three types of features of the generated deep learning models, as well as the generation time of each model.
* **MPNN_edge_message_TCP.ipynb**: The file MPNN_edge_message_TCP.ipynb contains the core implementation of the MeDAC approach. This Jupyter notebook provides a detailed demonstration of:(1) How we construct and train a message-passing neural network (MPNN) with an attention mechanism, along with the iterative process of updating the intermediate representations of the deep learning model by refining node and edge features. (2) How to load our preprocessed deep learning model data to accelerate the testing process of DL compilers, along with a performance evaluation of the achieved speedup.
* **LET.ipynb**: The LET.ipynb file demonstrates how we construct a deep learning model based on the LET concept to accelerate the testing process of deep learning compilers, along with the resulting speedup performance.
* **GCN.ipynb**: Similarly, the GCN.ipynb file demonstrates how we construct a deep learning model based on the GCN to accelerate the testing process of deep learning compilers, along with the resulting speedup performance.
* **Other files**: Additionally, each subfolder contains numerous auxiliary files, which include both the training datasets and the implementation details of how we utilized the NNSmith tool to generate each individual model, record model generation times, and extract internal structural features of the models.

The complete experimental data and implementation code for each testing scenario enable straightforward reproduction of all speedup effects reported in our paper.

