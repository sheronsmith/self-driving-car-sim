# self-driving-car-sim

A car that drives itself in a simulator.

Here is the idea. You drive a car around a track in a simulator. While you drive, the simulator saves pictures from the car's camera and records which way you turned the wheel. You feed all of that into a program called a neural network. The program studies the pictures and the matching steering, and it learns the pattern. After training, it can look at a new picture and decide how to steer on its own.

This is called behavioral cloning. The computer copies how you drive by watching you do it.

The car's brain is based on a design from Nvidia. They published it in 2016 in a paper called ["End to End Learning for Self-Driving Cars"](https://arxiv.org/abs/1604.07316). One program looks at the camera and decides the steering. Nothing else. Pictures go in, steering comes out.

## How the car learns

1. **Pictures are just numbers.** Every dot in a picture (a pixel) has 3 numbers for how red, green, and blue it is. We cut off the sky and the hood, shrink what is left to a small size, and now the program has a set of numbers to work with.

2. **Every picture has a correct answer.** The simulator saves your steering for each picture. That is the answer the program tries to match.

3. **The program makes a guess.** It holds a set of numbers called weights. It multiplies the picture numbers by the weights and adds them up through several layers. One number comes out at the end, and that is its steering guess.

4. **We measure how wrong it was.** If you turned left but it said turn right, that is a big miss. We measure the miss with mean squared error, which squares the difference so big mistakes count much more than small ones.

5. **Adjust the weights and repeat.** The program works out which weights should go up and which should go down to make the next guess better. It does this thousands of times, improving a little each round. One full pass through all the pictures is called an epoch.

## What is in here

```
├── model.py        # the car's brain, Nvidia's design in Keras
├── utils.py        # loads pictures, crops them, adds variety
├── train.py        # trains the brain on the driving data
├── drive.py        # connects the brain to the simulator
├── track.py        # makes a practice course if you do not have the sim
├── requirements.txt
└── README.md
```

## How to use it

### 1. Install what you need

```bash
pip install -r requirements.txt
```

### 2. Get the simulator

Download the [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim/releases). Pick the version for your computer (Windows, Mac, or Linux), unzip it, and open it.

It is from 2017, so it may behave strangely on newer computers. If it will not open, try the [Donkey Car simulator](https://docs.donkeycar.com/guide/deep_learning/simulator/) instead. It is also free.

### 3. Make a practice course (optional)

If you do not have the simulator yet, you can make test driving data:

```bash
python track.py --samples 3000
```

This creates a `data/` folder with simple road pictures and a file listing the steering values. It is not as good as real data, but it works for testing the pipeline.

### 4. Drive and record

- Open the simulator and choose **Training Mode**
- Press record, pick a folder, and drive a few laps
- Drive smoothly, with no sudden swerving
- Also practice steering back from the edges so the car learns how to recover if it drifts
- You get a folder called `IMG/` with the pictures and a file called `driving_log.csv`
- Put them in a `data/` folder in this project

### 5. Train the car

```bash
python train.py --data_dir data/ --epochs 10
```

This trains the brain and saves it as `model.keras`. It also makes a picture called `loss.png` that shows how well the car is learning.

### 6. Watch it drive

- Open the simulator and choose **Autonomous Mode**
- Then run:

```bash
python drive.py model.keras
```

The program connects to the simulator and drives the car using what it learned.

### 7. Keep improving it

The car will make mistakes. That is expected. Record more driving at the spot where it failed, train it again, and try again. This back-and-forth is most of the work.

## Tricks to make it better

The car mostly sees straight road while you drive, so without some help it just learns to go straight. These tricks fix that:

- **Use all 3 cameras.** The simulator gives you a center, left, and right view. The side cameras teach the car how to steer back toward the middle.
- **Flip pictures sideways.** A flipped left turn becomes a right turn, which doubles your data.
- **Change the brightness.** This helps the car handle different lighting.

## How you know it is working

- Look at `loss.png`. Both lines should go down. If the training line goes down but the validation line goes up, the car is memorizing instead of learning.
- The real test is the track. Can the car complete a full lap without driving off the road?

## If something goes wrong

| Problem | What to do |
|---|---|
| Simulator will not open | It is old. Check the GitHub issues or use Donkey Car instead |
| Car swerves right away | Make sure `drive.py` crops the picture the same way as training |
| Car only goes straight | Needs more turning data. Flip images and use all 3 cameras |
| It was learning but got worse | That is overfitting. Try fewer epochs or add more data |

## References

- [3Blue1Brown: "But what is a neural network?"](https://www.youtube.com/watch?v=aircAruvnKk), the best video for understanding this
- [Nvidia's blog post](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), the plain-language version
- [Nvidia's paper](https://arxiv.org/abs/1604.07316), the full details
- [Udacity Behavioral Cloning project](https://github.com/udacity/CarND-Behavioral-Cloning-P3), where the idea came from
- [naokishibuya/car-behavioral-cloning](https://github.com/naokishibuya/car-behavioral-cloning), the cleanest example I found
- [Keras docs](https://keras.io/getting_started/intro_to_keras_for_engineers/), for writing the brain
- [Donkey Car sim](https://docs.donkeycar.com/guide/deep_learning/simulator/), the free backup simulator

## Built by

[@sheronsmith](https://github.com/sheronsmith)

Based on a writeup by Hari ([@harixganesh](https://github.com/harixganesh))
