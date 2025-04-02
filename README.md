# DDPG-based UAV Deployment
_Mohammadmahdi Ghadaksaz <sup>1, 2</sup>, Mobeen Mahmood <sup>1, 2</sup>, Tho Le-Ngoc <sup>1, 2</sup>_

<sup>1</sup>Department of Electrical and Computer Engineering, McGill University

<sup>2</sup> Broadband Communication Lab
### Description
This repository contains the TensorFlow implementation of DDPG-based UAV deployment in static environments, used for simulations in various journal and conference publications by these authors. [IEEE Internet of Things Journal, EuCNC 2024].

This repository contains code and configurations for training the UAV agents for this environment. This Repository will be soon updated with the dynamic environment.

![teaser](DDPG_Block.jpg) 

### Adding The PWD to the Python Path
First, add your current directory to the Python path:
```sh
export PYTHONPATH=${PWD}
```
Then, after installing all required libraries, you may run the main.py file as
```sh
python src/main.py
```
### Changing the Code

You may change the number of users, location of the users, location of the UAV, and to name but a few. Also, you can change the number of episodes, number of runs, and many other attribute of the DDPG Agent Model.
