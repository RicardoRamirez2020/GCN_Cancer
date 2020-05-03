# GCN_Cancer
# Classification of Cancer Types Using Graph Convolutional Neural Networks

This is the code that is associated with the paper title above. Credit most go to Michaël Defferrard en al for
using his code as a base for the adaptation of this code, the citation is shown below. the code was modified to follow 
the GCN model decribed in Kipf's "Semi-Supervised Classification with Graph Convolutional Networks". 


## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/RicardoRamirez2020/GCN_Cancer
   cd GCN_Cancer
   ```

2. Install the dependencies. The code should run with TensorFlow 1.0 and newer.
   ```sh
   pip install -r requirements.txt  # or make install
   ```

## Reproducing our results
   ```sh
   cd GCN_Cancer
   PPI.py
   ```

## Using the model

To use the model you need 

1. A data matrix where each row is a sample and each column is a feature
2. Target Labels
3. An adjacency matrix which encodes the structure as a graph.

To use the GCN model given by Kipf, keep the K hyper parameter to 1 and implement the renormalization to your graph. 
Data can be sent individually. 

## Reference 
Please perfer to https://github.com/mdeff/cnn_graph for installation and other value sources about the topic.
