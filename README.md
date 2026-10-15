# self-driving-car-sim

i made a car that drives itself in a video game. for real.

heres how it works — you drive a car around a track in a simulator (like a racing game). while you drive, the game saves pictures from the car's camera and remembers which way you turned the wheel. then you give all of that to a computer program called a neural network. the program looks at all the pictures and all the turns you made, and it learns the pattern. after that, it can look at a new picture and go "oh i should turn left here" all by itself.

this is called **behavioral cloning** — the computer copies how you drive by watching you do it.

the brain of the car is based on a design from nvidia (the gpu company). they wrote a paper about it in 2016 called ["End to End Learning for Self-Driving Cars"](https://arxiv.org/abs/1604.07316). basically one program looks at the camera and tells the car where to steer. no extra steps. no cheat codes. just pictures in, steering out.

## how the car learns

1. **pictures are just numbers.** every dot in a picture (called a pixel) has 3 numbers — how red, green, and blue it is. we chop off the sky and the hood of the car, shrink what's left to a small size, and now the computer has a bunch of numbers to work with

2. **every picture has a right answer.** the game saves what you did with the steering wheel for each picture. thats the answer the computer is trying to match

3. **the computer takes a guess.** it has a bunch of numbers inside it called weights. it multiplies the picture numbers by the weights, adds stuff up through a bunch of layers, and one number pops out at the end — thats its guess for the steering

4. **we check how wrong it was.** if you turned left but it said go right, thats a big miss. we measure the miss with something called mean squared error — basically we square the difference so big mistakes count way more than little ones

5. **fix the weights and try again.** the computer figures out which weights need to go up and which need to go down to make the guess better. it does this over and over, thousands of times, getting a little better each round. one full pass through all the pictures is called an epoch

## whats in here

```
├── model.py        # the car's brain — nvidia's design in keras
├── utils.py        # loads pictures, crops them, adds variety
├── train.py        # teaches the brain using the driving data
├── drive.py        # connects the brain to the simulator
├── track.py        # makes a practice course if you dont have the sim
├── requirements.txt
└── README.md
```

## how to use it

### 1. install the stuff you need

```bash
pip install -r requirements.txt
```

### 2. get the simulator

download the [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim/releases). pick the one for your computer (windows, mac, or linux), unzip it, and open it up.

its from 2017 so it might be a little weird on new computers. if it wont open, try the [Donkey Car simulator](https://docs.donkeycar.com/guide/deep_learning/simulator/) instead — its free too

### 3. make a practice course (if you dont have the simulator)

you can make fake driving data to test things out:

```bash
python track.py --samples 3000
```

this makes a `data/` folder with simple road pictures and a file that says what the steering should be. its not as good as real data but it works for testing

### 4. drive around and record it

- open the simulator and pick **Training Mode**
- hit record, pick a folder, and drive some laps
- try to drive smooth — no crazy swerving
- also practice steering back from the edges so the car learns what to do if it drifts
- youll get a folder called `IMG/` with pictures and a file called `driving_log.csv`
- put them in a `data/` folder in this project

### 5. teach the car

```bash
python train.py --data_dir data/ --epochs 10
```

this trains the brain and saves it as `model.keras`. it also makes a picture called `loss.png` that shows how well the car is learning

### 6. watch it drive

- open the simulator and pick **Autonomous Mode**
- then run:

```bash
python drive.py model.keras
```

the program connects to the game and starts driving the car using what it learned. pretty cool to watch

### 7. keep fixing it

the car is gonna mess up somewhere. thats normal. go record more driving at the spot where it messed up, train it again, and try again. this back-and-forth is most of the project honestly

## tricks to make it better

the car sees mostly straight road when you drive, so without some tricks it just learns to go straight. here's what helps:

- **use all 3 cameras** — the sim gives you center, left, and right camera views. the side cameras teach the car how to steer back to the middle
- **flip pictures sideways** — if you flip a left turn picture, it becomes a right turn. boom, double the data
- **change the brightness** — makes the car handle different lighting

## how you know its working

- look at `loss.png` — both lines should go down. if the orange one (training) goes down but the blue one (validation) goes up, thats bad — the car is memorizing instead of learning
- the real test — can the car do a whole lap without driving off the road?

## if something goes wrong

| whats happening | what to do |
|---|---|
| simulator wont open | its old. check github issues or use donkey car instead |
| car goes crazy right away | make sure `drive.py` crops the picture the same way as training |
| car only goes straight | need more turning data — flip images and use all 3 cameras |
| it was learning but now its getting worse | thats overfitting — try fewer epochs or add more data |

## stuff i learned from

- [3Blue1Brown: "But what is a neural network?"](https://www.youtube.com/watch?v=aircAruvnKk) — best video to understand this stuff
- [Nvidia's blog post](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) — explains it in normal words
- [Nvidia's paper](https://arxiv.org/abs/1604.07316) — the full thing if you want all the details
- [Udacity Behavioral Cloning project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) — where the idea came from
- [naokishibuya/car-behavioral-cloning](https://github.com/naokishibuya/car-behavioral-cloning) — the cleanest example i found
- [Keras docs](https://keras.io/getting_started/intro_to_keras_for_engineers/) — for writing the brain
- [Donkey Car sim](https://docs.donkeycar.com/guide/deep_learning/simulator/) — free backup simulator

## built by

[@sheronsmith](https://github.com/sheronsmith)

based on a writeup by Hari ([@harixganesh](https://github.com/harixganesh))
