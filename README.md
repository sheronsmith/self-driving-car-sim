# self-driving-car-sim

so basically what this is — i taught a car to drive itself in a simulator. you drive around a track, the simulator records your camera footage and steering, then you train a neural network to copy you. picture goes in, steering angle comes out. thats it. its called behavioral cloning.

the model is based on nvidia's 2016 paper ["End to End Learning for Self-Driving Cars"](https://arxiv.org/abs/1604.07316) by Bojarski et al. one network, raw camera pixels straight to steering. no hand-built lane detector or anything like that.

## how it works

1. **a picture is just numbers.** every pixel is 3 numbers (red, green, blue). crop off the sky and the hood, resize to 66x200, and each frame is 39,600 numbers. thats all the computer sees — it has no idea what a "road" is

2. **every picture gets a label.** the sim saves each frame next to your steering angle in `driving_log.csv`. steering goes from -1 (hard left) to +1 (hard right)

3. **the network takes a guess.** 5 convolutional layers and 3 dense layers, about 250k weights. a convolutional layer slides a small filter across the image and learns to pick up edges, curves, lane lines — all on its own

4. **measure the miss.** mean squared error between what the model guessed and what you actually steered. squaring it makes big misses hurt way more than small ones

5. **nudge the weights and repeat.** gradient descent figures out which way to push each weight to shrink the loss, then moves it a little. one full pass through the data = one epoch. do that a bunch of times and the model gets good

## whats in here

```
├── model.py        # nvidia's architecture in keras
├── utils.py        # data loading, preprocessing, augmentation
├── train.py        # training script
├── drive.py        # autonomous mode — talks to the simulator
├── track.py        # generates a practice course with synthetic data
├── requirements.txt
└── README.md
```

## getting started

### 1. install stuff

```bash
pip install -r requirements.txt
```

### 2. get the simulator

grab the [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim/releases) — pick the one for your OS, unzip, run it.

heads up tho — this thing is from 2017 so it might act weird on newer systems. if it wont start check the github issues or just use the [Donkey Car simulator](https://docs.donkeycar.com/guide/deep_learning/simulator/) instead, its free

### 3. generate a practice course (no simulator needed)

if you just wanna test the pipeline or dont have the sim running yet, you can generate synthetic training data:

```bash
python track.py --samples 3000
```

this creates a `data/` folder with fake camera frames (simple road scenes with lane lines) and a `driving_log.csv`. the images are basic but they work for making sure everything runs end to end

### 4. record real driving data

- open the sim → **Training Mode**
- hit record, pick a folder, drive a few smooth laps
- also record yourself steering back from the edges — recovery maneuvers — or the car never learns what to do when it drifts
- youll get an `IMG/` folder and `driving_log.csv`
- drop them in a `data/` directory here

### 5. train it

```bash
python train.py --data_dir data/ --epochs 10
```

saves `model.keras` and a `loss.png` so you can see if its learning or just memorizing

### 6. let it drive

- open the sim → **Autonomous Mode**
- run:

```bash
python drive.py model.keras
```

connects to the sim on port 4567, gets camera frames, sends back steering. pretty cool to watch honestly

### 7. fix it

it WILL leave the road somewhere. record more data at that spot, retrain, repeat. thats most of the work and honestly the fun part

## data augmentation

without this the model just learns to say "go straight" because most of the training data is straight roads:

- **left/right cameras** with a ±0.2 steering offset — simulates recovery
- **random horizontal flips** with flipped steering — doubles the data
- **random brightness** — simulates different lighting

## how you know its working

- **loss plot** — validation loss should go down with training loss. if training goes down but validation goes up thats overfitting
- **the real test** — can the car do a full lap without leaving the road?
- **nvidia's metric** — `autonomy = (1 - interventions × 6s / lap_time) × 100%`

## common issues

| problem | whats probably wrong |
|---|---|
| sim wont start | its old. check github issues or use donkey car |
| car swerves immediately | `drive.py` isnt doing the same crop + resize as training |
| model just predicts ~0 | not enough augmentation. most data is straight driving |
| validation loss going up | overfitting — add dropout, more data, fewer epochs |

## references

- [3Blue1Brown: "But what is a neural network?"](https://www.youtube.com/watch?v=aircAruvnKk) — best intro out there
- [Nvidia's blog post](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) — plain english version
- [Nvidia's paper](https://arxiv.org/abs/1604.07316) — the full architecture
- [Udacity Behavioral Cloning project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) — the original task
- [naokishibuya/car-behavioral-cloning](https://github.com/naokishibuya/car-behavioral-cloning) — cleanest full example
- [Keras docs](https://keras.io/getting_started/intro_to_keras_for_engineers/) — for writing the model
- [Donkey Car sim](https://docs.donkeycar.com/guide/deep_learning/simulator/) — free alternative

## built by

[@sheronsmith](https://github.com/sheronsmith)

based on a writeup by Hari ([@harixganesh](https://github.com/harixganesh))
