# KANQAS
In this work, we explore the possibility of utilizing Kolmogorov-Arnold Network for Quantum Architecture Search i.e., namely KANQAS

## We run the noiseless/noisy experiments with:
```
python main.py --seed 1 --config 2q_bell_state_seed1 --experiment_name "DDQN/"
```

for MLP and for KAN
```
python main.py --seed 1 --config 2q_bell_state_seed1 --experiment_name "KAQN/"
```

## Configuration of experiment
The configuration for experiments to Bell and GHZ state constructions are in `configuration_files/` folder where the `DDQN` folder contains Double Deep Q-Network with Multi-Layer Perceptron and `KAQN` is Double Deep Q-Learning with Kolmogorov Arnold Network. 

## Results
The results are saved in the `results/` folder.

## The MLP code
The MLP part of the code is built using the [RL-VQE code agent](https://github.com/mostaszewski314/RL_for_optimization_of_VQE_circuit_architectures/blob/main/agents/DeepQ.py) and [RL-VQSD code agent](https://github.com/iitis/RL_for_VQSD_ansatz_optimization/blob/main/agents/DeepQ.py)
