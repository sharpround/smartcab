import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
from itertools import product
import numpy as np




class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        vi = self.env.valid_inputs
        va = self.env.valid_actions
        
        self.gamma   = 0.8
        self.alpha   = 0.3
        self.epsilon = 0.2
        Q_init       = 40.0
        self.success = []
        self.trial   = 0

        self.Q_table = pd.DataFrame(index=product(['red', 'green'], vi['oncoming'], vi['right'], vi['left'], va), columns=va)
        self.Q_table.fillna(value=Q_init, inplace=True)


    def reset(self, destination=None):
        self.planner.route_to(destination)

        # TODO: Prepare for a new trip; reset any variables here, if required
        self.last_state = None
        self.trial += 1
        if self.trial == 90:
            self.epsilon = 0.0

        inputs = self.env.sense(self)
        self.state = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint)


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        deadline = self.env.get_deadline(self)
        
        # TODO: Select action according to your policy
        # action = self.next_waypoint # always go to the next waypoint
        Q_row = self.Q_table.loc[[self.state]]
        Q_perturbed = Q_row + 0.001*np.random.rand(*Q_row.shape)
        action = Q_perturbed.idxmax(axis=1)[0]

        # Execute action and get reward
        self.last_state = self.state
        reward = self.env.act(self, action)

        if reward > 2.0:
            self.success.append(self.trial)

        # Update state
        inputs = self.env.sense(self)
        self.state = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint)

        # TODO: Learn policy based on state, action, reward
        Q_old = Q_row[action]
        Q_cur = self.Q_table.loc[[self.state]].max(axis=1)[0]

        Q_new = (1.0 - self.alpha)*Q_old + self.alpha*(reward + self.gamma*Q_cur)

        self.Q_table[action].loc[[self.last_state]] = Q_new

        print Q_row
        print "t = {:6},\ts = {:55},\ta = {:10},\tr = {:>4.1f},\tQ_i = {:>6.2f},\tQ_i+1 = {:>6.2f}\n".format(deadline, self.last_state, action, reward, Q_old[0], Q_new[0])  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    
    pd.Series(a.success).to_pickle('success3.pickle')
    a.Q_table.to_pickle('qtable3.pickle')


if __name__ == '__main__':
    run()
