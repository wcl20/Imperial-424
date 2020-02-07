import numpy as np
from Grid import Grid
from Agent import Agent

def main():

	grid = Grid(p=0.45)
	agent = Agent(grid.all_states(), ['N', 'E', 'S', 'W'], epsilon=0.2)


	# Train 500 epochs
	for _ in range(5000):

		# Set initial state
		state = (1,3)
		grid.set_state(state)

		# Generate an episode using current policy
		episode = []
		while not grid.terminal_state():
			action = agent.act(state)
			reward = grid.move(action)
			episode.append((state, action, reward))

			state = grid.current_state()

		# Train agent using episode
		agent.train(episode, gamma=0.65)

	grid.show_values(agent.optimal_value_function())
	grid.show_policy(agent.optimal_policy())



if __name__ == '__main__':
	main()