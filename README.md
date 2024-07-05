# KANQAS
In this code, which compliments our [paper](https://scirate.com/arxiv/2406.17630), we provide the code that opens up the possibility of harnessing Kolmogorov-Arnold Network for Quantum Architecture Search i.e., namely KANQAS

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


# To study the interpretability of KANs
As a motivation for future work towards the interpretability of KAN, we illustrate trained KAN in constructing Bell state
![The learned nerwotk](pics/the_network_after_training_bell_state.png)
where we use the `[84,2,12]` configuration. The `Tensor encoded quantum circuit as input to KAN` contains 84 entries because the quantum circuit is encoded into $D\times (N\times(N+5))$ dimension tensor where $D=6$ corresponds to maximum depth. For more details please check our [paper](https://scirate.com/arxiv/2406.17630). We can see that not all the neurons are actively contributing to the choice of action, which is defined as `Quantum gates as output of KAN`.


Due to the huge dimension of the KAN in the previous picture the `activation function` is in between the input and the output layers the activation functions are not visible. Hence in the following illustration, we explicitly show the trend of the `activation function` of the trained KAN
![The learned nerwotk](pics/2q_activation_function.png)
