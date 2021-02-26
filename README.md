# Thinking Deeply with Recurrence: Generalizing from Easy to Hard Sequential Reasoning Problems

This repository is the official implementation of [Thinking Deeply with Recurrence: Generalizing from Easy to Hard Sequential Reasoning Problems](https://arxiv.org/abs/2102.11011). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Maze Data

The maze data can be downloaded from [here](https://drive.google.com/drive/folders/1ad_ZESAddlfx-b3CnK1ohoKz6Sp8U-5g?usp=sharing). Maze generation code is also available in the [maze_data directory](./maze_data).

## Training

Our training routine will run an evaluation after training is done. To train a recurrent model to solve mazes, run this command:

```train
python train_model.py --model recur_residual_network_segment --width 2 --depth 36 --dataset mazes_small --lr 0.001 --lr_factor 0.25 --lr_schedule 40 100 --epochs 160 --problem segment --save_json --save_period 50 --train_batch_size 50 
```

Launch scripts that run sets of experiments included in the paper are available in the [launch directory](./launch). For training models one at a time, or tinkering with the setup, you can run `train_model.py` on its own. See the arguments in `train_model.py` for all the options.

## Evaluation

To evaluate a model after training, run `train_model.py` with the `--model_path` argument to point to the saved model. Be sure to set the number of epochs to the number of epochs used to train the saved model, as this will skip the training loop.

## Results

Results form experiments exploring depth versus recurrence and maze solving networks are available in the paper.

