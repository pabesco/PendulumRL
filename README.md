# PendulumRL
Open AI gym package Pendulum control using Q-Learning
https://www.gymlibrary.dev/environments/classic_control/pendulum/

Pendulum has torque control at its pivot. The objective is to stabilize the pendulum in upright position.

The action space of pendulum is a continuous torque between (-2.0, 2.0)

The reward function for the Pendulum motion is: 
r = -(theta^2 + 0.1 * theta_dt^2 + 0.001 * torque^2)

The observation space of the Pendulum has following states: 
0: x = cos(theta) 
1: y = sin(theta) 
2: Angular velocity 

Q-Learning Algorithm: 

For Q-Learning using Q-Table, the action space is discretized into 1000 torque steps.

The Q_table is initialized with random q-values.

The robot applies torque corresponding to its current state with maximum q-value.

The q-value is updated as based on discounted q-value of the next state. 

The process continues for multiple episodes and the q-table gets updated to find minimum number of steps to reach the termination state.
