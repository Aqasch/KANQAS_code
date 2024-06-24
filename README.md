# KANQAS
In this work we explore tthe possibility of utilizing Kolmogorov-Arnold Network for Quantum Architecture Search i.e., namely KANQAS

## We run the noiseless/noisy experiments with:
`python main.py --seed 1 --config 2q_bell_state_seed1 --experiment_name "DDQN/"`

for MLP and for KAN

`python main.py --seed 1 --config 2q_bell_state_seed1 --experiment_name "KAQN/"`

## Configuration of experiment
The configuration for exoeriments to Bell and GHZ state constructions are in `configuration_files/` folder where the `DDQN` folder contains Double Deep Q-Network with Multi Layer Parcepton and `KAQN` is Double Deep Q-Learning with Kolmogorov Arnold Network. 

## Results
The results are saved in the `results/` folder.