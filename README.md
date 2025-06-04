# AI vs. the Algorithm: Measuring Success on Twitch

## Description
This git repository contains the resources described in the paper of the same name (link to the paper to be added post publication).
These resources aim to enable continued work in measuring success on Twitch. Two main resources are needed.

 - The Pickled K-Maxoid model
 - The Standard Scaler model

 Both models are described on the paper.

 ## Requirements
 Python implementation of the K-Maxoid algorithm is available [here](https://github.com/ChitaAPC/KMaxoids) and is needed for this project.

 Data used to train the model was collected through Kaggle, with the dataset available [here](https://www.kaggle.com/datasets/rankirsh/evolution-of-top-games-on-twitch).

 ## Usage
 Example usage can be found in the `ModelExample.py` file.

 This includes:
  - Downloading and processing the data from the original Kaggle repository
  - Loading the trained K-Maxoid model and scaler models
  - Clustering the entire dataset
  - Visualising clusters through time for a given game

## Cite this work
Information on publication to be added.