import numpy as np

def initialize_population(population_size, dimension):
    return np.random.rand(population_size, dimension)

def calculate_fitness(positions):
    return np.sum(positions, axis=1)

def update_producers(positions, fitness_values, iteration, max_iterations, alert_value, safe_threshold):
    alpha = np.random.rand()
    Q = np.random.normal(0, 1, size=positions.shape)
    L = np.ones(positions.shape)
    R2 = np.random.rand()

    if R2 >= safe_threshold:
        positions += alpha * Q * (fitness_values.reshape(-1, 1) - positions)
    else:
        positions += alpha * L * (fitness_values.reshape(-1, 1) - positions)

    return positions

def update_explorers(positions, fitness_values, worst_position, exploration_rate):
    A = np.random.choice([-1, 1], size=positions.shape)
    R1 = np.random.rand()

    positions += A * R1 * (worst_position - positions) * exploration_rate

    return positions

def update_scouts(positions, global_best_position, epsilon, beta, best_fitness, worst_fitness, hazard_awareness):
    K = np.random.uniform(-1, 1)
    direction = positions - global_best_position

    positions += beta * K * direction / (epsilon + (best_fitness - worst_fitness))

    if hazard_awareness > 0:
        positions += hazard_awareness * (positions.mean(axis=0) - positions)

    return positions

def sparrow_search_algorithm(population_size, dimension, max_iterations, producer_ratio, safe_threshold, hazard_awareness):
    positions = initialize_population(population_size, dimension)
    best_position = positions.copy()
    best_fitness = calculate_fitness(best_position)

    for iteration in range(max_iterations):
        fitness_values = calculate_fitness(positions)

        producers_count = int(producer_ratio * population_size)
        producers_indices = np.argpartition(fitness_values, -producers_count)[-producers_count:]
        producers_positions = positions[producers_indices]

        worst_position = positions[np.argmin(fitness_values)]
        exploration_rate = np.random.rand()

        positions[producers_indices] = update_producers(
            producers_positions, fitness_values[producers_indices], iteration, max_iterations, 0, 0
        )

        positions = update_explorers(positions, fitness_values, worst_position, exploration_rate)
        positions = update_scouts(
            positions, best_position.mean(axis=0), 1e-8, 1, best_fitness, fitness_values.min(), hazard_awareness
        )

        # Update best position
        current_best_fitness = fitness_values.max()
        if current_best_fitness > best_fitness:
            best_position = positions.copy()
            best_fitness = current_best_fitness

    return best_position, best_fitness

# Example usage
population_size = 50
dimension = 10
max_iterations = 100
producer_ratio = 0.2
safe_threshold = 0.7
hazard_awareness = 0.1

best_position, best_fitness = sparrow_search_algorithm(
    population_size, dimension, max_iterations, producer_ratio, safe_threshold, hazard_awareness
)

print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
