import numpy as np
import seaborn as sns
from hmmlearn import hmm
import matplotlib.pyplot as plt

random_seed = 42  # Fixed random seed
n_rewards = 3   # The reward value may be 0, 1, 2
n_states = 9    # 9 hidden states, corresponding to the 9 positions of the grid
grid_size = 3   # Define the grid size
np.set_printoptions(suppress=True, precision=3)     # Suppress scientific notation and set precision

def load_data():
    with open('rewards.txt', 'r') as f:
        rewards = [int(line.strip()) for line in f]

    # Convert rewards to one-hot
    X = np.zeros((len(rewards), n_rewards), dtype=int)
    for idx, reward in enumerate(rewards):
        X[idx, reward] = 1

    return X

def create_true_trasition_matrix():
    # Create a mapping from position to state index
    position_to_state = {}
    state_to_position = {}
    state = 0
    for x in range(grid_size):
        for y in range(grid_size):
            position_to_state[(x, y)] = state
            state_to_position[state] = (x, y)
            state += 1

    n_states = grid_size * grid_size
    transmat_true = np.zeros((n_states, n_states))

    # Construct the transition probability matrix
    for state in range(n_states):
        x, y = state_to_position[state]
        neighbors = []
        # Up
        if y + 1 < grid_size:
            neighbors.append(position_to_state[(x, y + 1)])
        # Below
        if y - 1 >= 0:
            neighbors.append(position_to_state[(x, y - 1)])
        # Left
        if x - 1 >= 0:
            neighbors.append(position_to_state[(x - 1, y)])
        # Right
        if x + 1 < grid_size:
            neighbors.append(position_to_state[(x + 1, y)])
        n_neighbors = len(neighbors)
        trans_prob = 1 / n_neighbors
        for neighbor in neighbors:
            transmat_true[state, neighbor] = trans_prob

    return transmat_true


def hmm_method(trasition_matrix = None):
    # Initialize HMM model
    if trasition_matrix is not None:
        model = hmm.MultinomialHMM(n_components=n_states, n_iter=1000, tol=0.01, params='se', init_params='se', random_state=random_seed)
        model.transmat_ = trasition_matrix
    else:
        model = hmm.MultinomialHMM(n_components=n_states, n_iter=1000, tol=0.01, random_state=random_seed)

    model.n_features = n_rewards
    # Train the model
    model.fit(X)

    # Output results
    print("Initial probabilities:")
    print(model.startprob_)
    print("\nTransition probability matrix:")
    print(model.transmat_)
    print("\nEmission probability matrix:")
    print(model.emissionprob_)

    return model

def analyze_results(model, true_transition_matrix):
    # Get learned transition matrix
    learned_transition_matrix = model.transmat_

    # Calculate and output differences, average error, and maximum error
    difference = learned_transition_matrix - true_transition_matrix
    average_error = np.mean(np.abs(difference))
    max_error = np.max(np.abs(difference))

    print("\nDifference between learned and true transition probabilities:")
    print(difference)
    print(f"\nAverage error: {average_error}")
    print(f"Maximum error: {max_error}")

    # Visualize transition matrices
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(learned_transition_matrix, annot=True, fmt=".2f")
    plt.title("Learned Transition Matrix")
    plt.subplot(1, 2, 2)
    sns.heatmap(true_transition_matrix, annot=True, fmt=".2f")
    plt.title("True Transition Matrix")
    plt.show()

if __name__ == "__main__":
    X = load_data()
    # Code Task 14
    print("\nHidden Markov Model without transition probabilities:")
    model_without_true = hmm_method()
    # Code Task 15
    transmat_true = create_true_trasition_matrix()
    print("\nHidden Markov Model with true transition probabilities:")
    hmm_method(transmat_true)

    # Analyze results with true transition matrix provided
    analyze_results(model_without_true, transmat_true)