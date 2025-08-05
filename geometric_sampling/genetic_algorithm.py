import numpy as np
from typing import List, Tuple

from geometric_sampling.design import DesignGenetic
from geometric_sampling.GeneticOptimizer import GeneticOptimizer
from geometric_sampling.criteria import VarNHT


class GeometricSamplingGA:
    # Algorithm constants
    DEFAULT_MAX_GENERATIONS = 1000
    VALIDATION_TOLERANCE = 1e-6
    EARLY_SEARCH_PHASE = 0.3
    LATE_SEARCH_PHASE = 0.7

    def __init__(
            self,
            inclusions: np.ndarray,
            auxiliary_var: np.ndarray,
            population_size: int = 40,
            elitism_rate: float = 0.15,
            mutation_intensity: int = 3,
            use_partitions: bool = True,
            random_pull: bool = False,
            adaptive_parameters: bool = False,
            selection_pressure: float = 1.5,

    ):
        # Core algorithm parameters
        self.random_pull = random_pull
        self.inclusions = inclusions
        self.auxiliary_var = auxiliary_var
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.mutation_intensity = mutation_intensity
        self.selection_pressure = selection_pressure
        self.use_partitions = use_partitions
        self.adaptive_parameters = adaptive_parameters

        # diversity threshold
        self.diversity_threshold = 0.1

        # Simple adaptive parameters (only if enabled)
        if adaptive_parameters:
            self.initial_mutation_rate = 0.2
            self.mutation_rate = self.initial_mutation_rate
            self.stagnation_counter = 0
            self.last_best_fitness = float('inf')
        else:
            self.mutation_rate = 2.0  # Always mutate when not adaptive
            self.stagnation_counter = 0  # Initialize for statistics even when not adaptive
            self.last_best_fitness = float('inf')

        # Algorithm components initialization
        self.optimizer = GeneticOptimizer()
        self.criterion = VarNHT(auxiliary_var, inclusions)
        self.rng = np.random.default_rng()

        # Setup partitions and border units if requested
        if self.use_partitions:
            self.partitions, self.border_units = self.optimizer.partition_design(
                inclusions.tolist(), 2
            )
        else:
            self.partitions = None
            self.border_units = None

        # Algorithm state tracking
        self.fitness_history = []
        self.best_design = None
        self.best_fitness = float('inf')

    def create_initial_population(self) -> List[DesignGenetic]:
        population = []
        base_design = DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(42))
        population.append(base_design)

        for i in range(self.population_size - 1):
            design = DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(i))
            num_interchanges = self.rng.integers(1, 50)
            for _ in range(num_interchanges):
                if len(design.heap) >= 2:
                    design.iterate(
                        random_pull=self.random_pull,
                        switch_coefficient=0.5,
                        partitions=self.partitions,
                        border_units=self.border_units
                    )
                    # if design.changes % 3 == 0 :
                    #     design.merge_identical()
            design.merge_identical()
            population.append(design)

        return population

    def chaotic_initial_population(self, M=100, alpha=0.5):
        population = []
        for _ in range(self.population_size):
            design = DesignGenetic(self.inclusions, rng=self.rng)
            for _ in range(10):
                design.iterate(
                    random_pull=self.random_pull,
                    switch_coefficient=self.rng.uniform(alpha * 0.5, alpha * 1.5),
                    partitions=self.partitions,
                    border_units=self.border_units
                )
                if design.changes % 3 == 0:
                    design.merge_identical()
            design.merge_identical()
            population.append(design)
        return population

    def evaluate_fitness(self, design: DesignGenetic) -> float:
        try:
            variance = self.criterion(design)
            return variance
        except Exception as e:
            print(f"Fitness calculation failed: {e}")
            return 1e10

    def rank_based_selection(self, population: List[DesignGenetic],
                             fitness_scores: List[float],
                             parent_counts: dict,
                             max_children: int = 6,
                             selection_pressure: float = 1.5) -> int:

        available_indices = [i for i in range(len(population)) if parent_counts.get(i, 0) < max_children]

        if not available_indices:
            parent_counts.clear()
            available_indices = list(range(len(population)))

        # Get fitness scores and rank them (lower fitness = better rank)
        available_fitness = [fitness_scores[i] for i in available_indices]
        sorted_indices = sorted(range(len(available_fitness)), key=lambda i: available_fitness[i])

        # Assign ranks (0 = best, n-1 = worst)
        ranks = [0] * len(available_fitness)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank

        # Calculate selection probabilities using linear ranking
        n = len(available_fitness)
        weights = []
        for rank in ranks:
            # Linear ranking formula: P(i) = (2 - SP + 2*(SP-1)*(N-i-1)/(N-1)) / N
            # where SP = selection_pressure, N = population_size, i = rank
            prob = (2 - selection_pressure + 2 * (selection_pressure - 1) * (n - rank - 1) / (n - 1)) / n
            weights.append(max(0.01, prob))  # Ensure minimum probability

        # Create cumulative distribution
        cumulative_weights = np.cumsum(weights)
        total_weight = cumulative_weights[-1]

        # Select
        random_value = self.rng.random() * total_weight
        selected_position = np.searchsorted(cumulative_weights, random_value)
        selected_position = min(int(selected_position), len(available_indices) - 1)

        return available_indices[selected_position]

    def select_parent(self,
                      population: List[DesignGenetic],
                      fitness_scores: List[float],
                      parent_counts: dict,
                      max_children: int = 6) -> int:

        return self.rank_based_selection(population, fitness_scores,
                                         parent_counts, max_children,
                                         self.selection_pressure)

    def mutate_design(self, design: DesignGenetic) -> DesignGenetic:
        if self.rng.random() > self.mutation_rate:
            return design.copy()

        mutated = design.copy()
        # if self.rng.random() < 0.2:  # 20% chance: apply chaotic mutation
        #     for _ in range(self.mutation_intensity * 5):
        #         mutated.iterate(
        #             random_pull=self.random_pull,
        #             switch_coefficient=self.rng.uniform(0.2, 0.8),
        #             partitions=None,  # intentionally ignore partitions
        #             border_units=self.border_units
        #         )
        #         if design.changes % 3 == 0 :
        #             mutated.merge_identical()
        mutated.merge_identical()
        # Apply several segment interchanges to increase entropy
        for _ in range(self.mutation_intensity * 15):
            if len(mutated.heap) >= 2:
                mutated.iterate(
                    random_pull=self.random_pull,
                    switch_coefficient=self.rng.uniform(0.3, 0.7),
                    partitions=self.partitions,
                    border_units=self.border_units
                )
            # if design.changes % 5 == 0 :
            #     mutated.merge_identical()
        mutated.merge_identical()
        return mutated

    def crossover(self, parent1: DesignGenetic,
                  parent2: DesignGenetic) -> Tuple[DesignGenetic, DesignGenetic]:
        try:
            child1, child2 = self.optimizer.combine_n_parents([parent1, parent2],
                                                              random_pull=self.random_pull)

            # Clean up children
            child1.merge_identical()
            child2.merge_identical()

            return child1, child2
        except Exception as e:
            print(f"Crossover failed: {e}")
            return self.mutate_design(parent1), self.mutate_design(parent2)

    def select_elites(self,
                      population: List[DesignGenetic],
                      fitness_scores: List[float]) -> List[DesignGenetic]:

        elite_count = max(1, int(self.elitism_rate * self.population_size))
        elite_indices = np.argsort(fitness_scores)[:elite_count]
        return [population[i].copy() for i in elite_indices]

    def validate_design(self, design: DesignGenetic) -> bool:
        # Calculate actual inclusion probabilities
        id_probs = {}
        for sample in design.heap:
            for unit in sample.ids:
                id_probs[unit] = id_probs.get(unit, 0) + sample.probability

        # Check if all units are present and probabilities match
        for i, expected_prob in enumerate(self.inclusions):
            actual_prob = id_probs.get(i, 0)
            if abs(actual_prob - expected_prob) > self.VALIDATION_TOLERANCE:
                print("Validation failed for unit", i,
                      f"expected {expected_prob:.6f}, got {actual_prob:.6f}"),
                print("Design heap:", design.heap,
                      "Inclusions:", self.inclusions,

                      )
                return False

        return True

    def calculate_population_diversity(self,
                                       population: List[DesignGenetic],
                                       fitness_scores: List[float]) -> float:
        if len(population) < 2:
            return 0.0

        fitness_std = np.std(fitness_scores)
        fitness_mean = np.mean(fitness_scores)
        fitness_diversity = fitness_std / (fitness_mean + 1e-10) if fitness_mean > 0 else 0

        sample_counts = [len(design.heap) for design in population]
        structural_diversity = np.std(sample_counts) / (np.mean(sample_counts) + 1e-10)

        combined_diversity = (fitness_diversity + structural_diversity) / 2
        return min(1.0, combined_diversity)

    def adapt_parameters(self, current_fitness: float, generation: int):
        if not self.adaptive_parameters:
            return

        # Check for improvement
        improvement = self.last_best_fitness - current_fitness
        has_improvement = improvement > 1e-6

        if has_improvement:
            # Good progress: slightly reduce mutation rate for fine-tuning
            self.mutation_rate *= 0.95
            self.stagnation_counter = 0
            self.last_best_fitness = current_fitness
        else:
            # No improvement: increase stagnation counter
            self.stagnation_counter += 1

            # If stagnated for several generations, increase mutation
            if self.stagnation_counter > 5:
                self.mutation_rate *= 1.05

            # Reset if stagnated too long
            if self.stagnation_counter > 20:
                self.mutation_rate = self.initial_mutation_rate
                self.stagnation_counter = 0

        # Keep mutation rate in reasonable bounds
        self.mutation_rate = max(0.05, min(0.8, self.mutation_rate))

    def run(self, max_generations: int = 100,
            verbose: bool = True) -> DesignGenetic:

        if verbose:
            print("Initializing Geometric Sampling Genetic Algorithm...")
            print(f"Population size: {self.population_size}")
            print(f"Inclusion probabilities: {self.inclusions}")
            print(f"Auxiliary variable: {self.auxiliary_var}")
            print(f"Using partitions: {self.use_partitions}")
            print("=" * 50)

        # Initialize population
        population = self.create_initial_population()[:self.population_size // 2] + \
                     self.chaotic_initial_population(M=20, alpha=0.7)[:self.population_size // 2]

        for generation in range(max_generations):

            # Evaluate fitness for all designs
            fitness_scores = [self.evaluate_fitness(design) for design in population]

            # Track best design
            best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[best_idx]

            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_design = population[best_idx].copy()

            self.fitness_history.append(current_best_fitness)

            # Calculate population diversity and adapt parameters
            population_diversity = self.calculate_population_diversity(population, fitness_scores)

            # adapt parameters based on diversity and fitness trends
            self.adapt_parameters(current_best_fitness, generation)

            # Create next generation
            population = self._create_next_generation(population, fitness_scores)

            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {current_best_fitness:.8f}, "
                      f"Diversity = {population_diversity:.3f}, "
                      f"MutRate = {self.mutation_rate:.3f}, "
                      f"Elite = {self.elitism_rate:.3f}")

        if verbose:
            print("=" * 50)
            print(f"Algorithm completed after {max_generations} generations")
            print(f"Best fitness achieved: {self.best_fitness:.8f}")
            # Validate final best design
            if self.validate_design(self.best_design):
                print("✅ Best design passes validation!")
            else:
                print("❌ Best design failed validation!")

        return self.best_design

    def _create_next_generation(self,
                                population: List[DesignGenetic],
                                fitness_scores: List[float]) -> List[DesignGenetic]:

        # 1. Elitism: preserve best designs
        new_population = self.select_elites(population, fitness_scores)

        # 2. Initialize selection tracking
        parent_counts = {}  # Dictionary tracking how many children each parent has produced
        max_children_per_parent = 3  # Each individual can be parent at most 3 times

        # 3. Fill rest with offspring
        while len(new_population) < self.population_size:
            # Smart selection using configured method (optimized for VarNHT minimization)
            parent1_idx = self.select_parent(population, fitness_scores,
                                             parent_counts, max_children_per_parent)
            parent2_idx = self.select_parent(population, fitness_scores,
                                             parent_counts, max_children_per_parent)

            # Make sure parents are different
            attempts = 0
            while parent1_idx == parent2_idx and attempts < 10:
                parent2_idx = self.select_parent(population, fitness_scores,
                                                 parent_counts, max_children_per_parent)
                attempts += 1

            # Get parent designs
            parent1 = population[parent1_idx].copy()
            parent2 = population[parent2_idx].copy()

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            child1 = self.mutate_design(child1)
            child2 = self.mutate_design(child2)

            # Validate and add children, then update parent counts
            children_added = 0
            if self.validate_design(child1):
                new_population.append(child1)
                children_added += 1
            if len(new_population) < self.population_size and self.validate_design(child2):
                new_population.append(child2)
                children_added += 1
            # Update parent counts only if children were actually added
            if children_added > 0:
                parent_counts[parent1_idx] = parent_counts.get(parent1_idx, 0) + children_added
                parent_counts[parent2_idx] = parent_counts.get(parent2_idx, 0) + children_added

            # Safety check to prevent infinite loop
            if len(new_population) >= self.population_size:
                break

        # Trim to exact population size
        return new_population[:self.population_size]

