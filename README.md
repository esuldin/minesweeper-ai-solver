# Minesweeper AI Solver
This is a project to create a Minesweeper game solver based on a convolutional neural network (CNN) using Pytorch.
The project includes:
* the neural network model with 1 input, 4 hidden and 1 output layers
* the simulation of minesweeper game
* the trainer to train the model using the results of game simulation
* the adapter to play Microsoft Minesweeper game from Windows Store using trained model

## Prepare Environment
Before you start, please make sure that you have installed necessary packages:
          
    pip install -r requirements.txt

## Train Model
Use `train_model.py` script to train the model using the specified parameters, for example:

     python train_model.py --game-mode=classic --training-iterations=10 --epochs=4 --batches=5 --batch-size=200

## Verify Model
The project provides two options to see how the trained model can play:
1. If you have Windows 10 or newer with Windows Store, you can install Microsoft Minesweeper game and use
   `play_ms_minesweeper.py` script to play the game using th trained model:
   1. Run Microsoft Minesweeper game.
   2. Select the game mode and wait until the game field is fully shown.
   3. Run `play_ms_minesweeper.py` script:

          python play_ms_minesweeper.py
   
2. Use `play_minesweeper.py` script to verify the model using the Minesweeper game simulation:

       python play_ms_minesweeper.py --game-mode=expert

The included pretrained models for different modes provide following win rate:

|Game Mode| Win Rate |
|---------|----------|
| Classic | ~88%     |
| Easy    | ~93%     |
| Medium  | ~77%     | 
| Expert  | ~29%     |

## References
https://github.com/ryanbaldini/MineSweeperNeuralNet
