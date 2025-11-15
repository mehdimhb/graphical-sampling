import numpy as np
from typing import List, Tuple, Optional

from geometric_sampling.design import DesignGenetic
from geometric_sampling.GeneticOptimizer import GeneticOptimizer
from geometric_sampling.criteria import VarNHT
from geometric_sampling.monitoring import GAMonitor


class GeometricSamplingGA:
    VALIDATION_TOLERANCE = 1e-6

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
            enable_monitoring: bool = True,
            enable_live_plots: bool = False,
            save_metrics: bool = True,
            mutation_rate: float = 2.0
    ):
        # Core algorithm parameters
        self.inclusions = inclusions
        self.auxiliary_var = auxiliary_var
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.mutation_intensity = mutation_intensity
        self.selection_pressure = selection_pressure
        self.use_partitions = use_partitions
        self.random_pull = random_pull
        self.adaptive_parameters = adaptive_parameters

        # Algorithm components
        self.rng = np.random.default_rng()
        self.optimizer = GeneticOptimizer()
        self.criterion = VarNHT(auxiliary_var, inclusions)
        self.monitor = GAMonitor(enable_live_plots=enable_live_plots,
                                 save_data=save_metrics) if enable_monitoring else None

        # Adaptive parameter state
        self.mutation_rate = mutation_rate if adaptive_parameters else 2.0
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')

        # Algorithm state
        self.best_design: Optional[DesignGenetic] = None
        self.best_fitness = float('inf')

        # Setup partitions if needed
        self.partitions, self.border_units = (
            self.optimizer.partition_design(inclusions.tolist(), 2)
            if self.use_partitions else (None, None)
        )

    # --- Main Execution Method ---
    def run(self, max_generations: int = 100,
            verbose: bool = True,
            save_report_path: Optional[str] = None) -> Optional[
        DesignGenetic]:
        """Executes the genetic algorithm for a specified number of generations."""
        self._initialize_run(verbose)
        population = self.create_initial_population()
        for design in population:
            self.validate_design(design)
        for generation in range(max_generations):
            # for design in population:
            #     if design.changes >5:
            #         design.merge_identical()

            fitness_scores = [self.evaluate_fitness(design) for design in population]

            self._update_best_design(population, fitness_scores)

            if self.monitor:
                self.monitor.record_generation(generation, population, fitness_scores, self)

            self.adapt_parameters(self.best_fitness)

            population = self._create_next_generation(population, fitness_scores)

            if verbose and generation % 10 == 0:
                self._log_generation_progress(generation, fitness_scores)
            if generation == 99:
                for design in population:
                    # if len(design.heap) < 5:
                    design.show()
        self._finalize_run(max_generations, verbose, save_report_path)
        return self.best_design

    # --- Helper methods for the `run` process ---
    def _update_best_design(self, population: List[DesignGenetic], fitness_scores: List[float]):
        """Checks for and updates the best design found so far."""
        current_best_fitness = min(fitness_scores)
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            best_idx = np.argmin(fitness_scores)
            self.best_design = population[best_idx].copy()

    def _initialize_run(self, verbose: bool):
        """Prints initial parameters and sets up the run."""
        if verbose:
            print("Initializing Geometric Sampling Genetic Algorithm...")
            print(
                f"Population size: {self.population_size}, Elitism: {self.elitism_rate}, Mutation Intensity: {self.mutation_intensity}")
            print(f"Adaptive Parameters: {self.adaptive_parameters}, Selection Pressure: {self.selection_pressure}")
            print("=" * 50)

    def _log_generation_progress(self, generation: int, fitness_scores: List[float]):
        """Logs the progress of the current generation."""
        diversity = self.calculate_population_diversity([], fitness_scores)
        print(f"Generation {generation: >4}: Best Fitness = {self.best_fitness:.8f}, "
              f"Diversity = {diversity:.3f}, MutRate = {self.mutation_rate:.3f}")

    def _finalize_run(self, max_generations: int, verbose: bool, save_report_path: Optional[str]):
        """Wraps up the algorithm run, printing summaries and generating reports."""
        if verbose:
            print("=" * 50)
            print(f"Algorithm completed after {max_generations} generations.")
            print(f"Best fitness achieved: {self.best_fitness:.8f}")
            if self.best_design and self.validate_design(self.best_design):
                print("✅ Best design passes validation!")
            else:
                print("❌ Best design failed validation!")

        if self.monitor:
            if verbose:
                print("\n📊 Generating final monitoring report...")
            self.monitor.generate_final_report(save_path=save_report_path)

    # --- Population Creation and Evolution ---
    def _create_next_generation(self, population: List[DesignGenetic], fitness_scores: List[float]) -> List[
        DesignGenetic]:
        """Creates the next generation through elitism, crossover, and mutation."""
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        elite_indices = np.argsort(fitness_scores)[:elite_count]
        new_population = [population[i].copy() for i in elite_indices]

        parent_counts = {}
        max_children_per_parent = 2

        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parent_pair(population, fitness_scores, parent_counts,
                                                        max_children_per_parent)

            child1, child2 = self._create_offspring(parent1, parent2)
            self.validate_design(child1)
            self.validate_design(child2)

            # Add valid children to the new population
            for child in [child1, child2]:
                if len(new_population) < self.population_size and self.validate_design(child) :
                    new_population.append(child)

                    parent_counts[id(parent1)] = parent_counts.get(id(parent1), 0) + 1
                    parent_counts[id(parent2)] = parent_counts.get(id(parent2), 0) + 1


        return new_population

    def _select_parent_pair(self, population: List[DesignGenetic], fitness_scores: List[float], parent_counts: dict,
                            max_children: int) -> Tuple[DesignGenetic, DesignGenetic]:
        parent1_idx = self.rank_based_selection(population, fitness_scores, parent_counts, max_children)

        attempts = 0
        while attempts < 10:
            parent2_idx = self.rank_based_selection(population, fitness_scores, parent_counts, max_children)
            if parent1_idx != parent2_idx:
                return population[parent1_idx], population[parent2_idx]
            attempts += 1

        raise Exception("Failed to select two distinct parents after multiple attempts.")

    def _create_offspring(self, parent1: DesignGenetic, parent2: DesignGenetic) -> Tuple[DesignGenetic, DesignGenetic]:
        try:

            child1, child2 = self.optimizer.combine_n_parents([parent1, parent2],
                                                              border_units= self.border_units)


        except Exception as e:
            print(f"Crossover failed: {e}. Returning mutated parents instead.")
            return self.mutate_design(parent1), self.mutate_design(parent2)

        child1.merge_identical()
        child2.merge_identical()

        mutated_child1 = self.mutate_design(child1)
        mutated_child2 = self.mutate_design(child2)
        return mutated_child1, mutated_child2

    def mutate_design(self, design: DesignGenetic) -> DesignGenetic:
        """Applies mutation to a design by performing interchanges."""
        if self.rng.random() > self.mutation_rate:
            return design  # No mutation occurs

        mutated = design.copy()
        for _ in range(self.mutation_intensity):
            if len(mutated.heap) >= 2:
                mutated.iterate(
                    random_pull=self.random_pull,
                    switch_coefficient=(self.rng.integers(1, 100) / 100),
                    partitions=self.partitions,
                    border_units=self.border_units
                )
        mutated.merge_identical()
        return mutated

    # --- Core GA Components (Selection, Fitness, etc.) ---
    def evaluate_fitness(self, design: DesignGenetic) -> float:
        """Calculates the fitness of a single design."""
        try:
            return self.criterion(design)
        except Exception as e:
            print(f"Fitness calculation failed for a design: {e}")
            return float('inf')  # Return a very high (bad) fitness value

    def create_initial_population(self) -> List[DesignGenetic]:
        """Creates the starting population with varied individuals."""
        population = []
        # Add a base design
        population.append(DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(42)))

        # Add mutated designs
        for i in range(self.population_size - 1):
            design = DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(i))
            num_interchanges = self.rng.integers(1, 50)
            for _ in range(num_interchanges):
                if len(design.heap) >= 2:
                    design.iterate(random_pull=self.random_pull,
                                   switch_coefficient=(self.rng.integers(1, 100) / 100),
                                   partitions=self.partitions,
                                   border_units=self.border_units)
            design.merge_identical()
            population.append(design)
        return population

    def rank_based_selection(self, population: List[DesignGenetic], fitness_scores: List[float], parent_counts: dict,
                             max_children: int) -> int:
        """Selects a parent index using rank-based weighting."""
        available_indices = [i for i in range(len(population)) if
                             parent_counts.get(id(population[i]), 0) < max_children]
        if not available_indices:
            available_indices = list(range(len(population)))

        available_fitness = [fitness_scores[i] for i in available_indices]
        sorted_indices_in_available = np.argsort(available_fitness)

        n = len(available_indices)
        ranks = np.arange(n)

        # Linear ranking probabilities
        sp = self.selection_pressure
        probs = (2 - sp + 2 * (sp - 1) * (n - 1 - ranks) / (n - 1)) / n
        probs /= probs.sum()

        # Select one index based on rank probabilities
        chosen_rank_idx = self.rng.choice(len(available_indices), p=probs)
        original_idx_in_available = sorted_indices_in_available[chosen_rank_idx]

        return available_indices[original_idx_in_available]

    def adapt_parameters(self, current_best_fitness: float):
        """Dynamically adjusts mutation rate based on performance stagnation."""
        if not self.adaptive_parameters:
            return

        if self.last_best_fitness - current_best_fitness > 1e-6:
            self.stagnation_counter = 0
            self.mutation_rate *= 0.98  # Decrease mutation on improvement
        else:
            self.stagnation_counter += 1

        if self.stagnation_counter > 5:
            self.mutation_rate *= 1.05  # Increase mutation on stagnation

        if self.stagnation_counter > 20:  # Major reset if stuck too long
            self.mutation_rate = 0.2
            self.stagnation_counter = 0

        self.mutation_rate = np.clip(self.mutation_rate, 0.05, 0.8)
        self.last_best_fitness = current_best_fitness

    # --- Validation and Utility Methods ---
    def validate_design(self, design: DesignGenetic) -> bool:
        """Checks if a design's inclusion probabilities match the target."""
        id_probs = {}
        for sample in design.heap:
            for unit in sample.ids:
                id_probs[unit] = id_probs.get(unit, 0) + sample.probability

        for i, expected_prob in enumerate(self.inclusions):
            if abs(id_probs.get(i, 0) - expected_prob) > self.VALIDATION_TOLERANCE:
                print(f"Validation failed for unit {i}: expected {expected_prob:.6f}, got {id_probs.get(i, 0):.6f}")
                return False
        return True

    def calculate_population_diversity(self, population: List[DesignGenetic], fitness_scores: List[float]) -> float:
        """Calculates a diversity score based on fitness and structural variance."""
        if len(fitness_scores) < 2:
            return 0.0

        fitness_mean = np.mean(fitness_scores)
        fitness_diversity = np.std(fitness_scores) / (abs(fitness_mean) + 1e-10)

        # This part requires the population, if not passed, we can only use fitness
        # sample_counts = [len(design.heap) for design in population]
        # structural_diversity = np.std(sample_counts) / (np.mean(sample_counts) + 1e-10)
        # combined_diversity = (fitness_diversity + structural_diversity) / 2

        return min(1.0, fitness_diversity)
