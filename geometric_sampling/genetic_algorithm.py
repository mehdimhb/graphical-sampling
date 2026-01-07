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
            main_var: np.ndarray,
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
            mutation_rate: float = 0.3,
            max_children_per_parent: int = 2,
            # NEW PARAMETERS FOR IMPROVED CONVERGENCE
            crossover_rate: float = 0.9,
            local_search_intensity: int = 5,
            tournament_size: int = 3,
            restart_threshold: int = 50,
            diversity_threshold: float = 0.01,
    ):
        # Core algorithm parameters
        self.inclusions = inclusions
        self.auxiliary_var = auxiliary_var
        self.main_var = main_var
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.mutation_intensity = mutation_intensity
        self.selection_pressure = selection_pressure
        self.use_partitions = use_partitions
        self.random_pull = random_pull
        self.adaptive_parameters = adaptive_parameters
        self.max_children_per_parent = max_children_per_parent

        # NEW: crossover and local search parameters
        self.crossover_rate = crossover_rate
        self.local_search_intensity = local_search_intensity
        self.tournament_size = tournament_size
        self.restart_threshold = restart_threshold
        self.diversity_threshold = diversity_threshold

        # Algorithm components
        self.rng = np.random.default_rng()
        self.optimizer = GeneticOptimizer()
        self.criterion = VarNHT(auxiliary_var, inclusions)  # optimized
        self.criterion_y = VarNHT(main_var, inclusions)  # reported only

        self.monitor = GAMonitor(enable_live_plots=enable_live_plots,
                                 save_data=save_metrics) if enable_monitoring else None

        # Adaptive parameter state
        self.mutation_rate = mutation_rate if adaptive_parameters else 0.3
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')
        self.global_stagnation_counter = 0  # NEW: track long-term stagnation

        # Algorithm state
        self.best_design: Optional[DesignGenetic] = None
        self.best_fitness = float('inf')
        self.fitness_history = []  # NEW: track fitness progress

        # Setup partitions if needed
        self.partitions, self.border_units = (
            self.optimizer.partition_design(inclusions.tolist(), 2)
            if self.use_partitions else (None, None)
        )

    # --- Main Execution Method ---
    def run(self, max_generations: int = 100,
            verbose: bool = True,
            save_report_path: Optional[str] = None) -> Optional[DesignGenetic]:

        population = self.create_initial_population()

        for generation in range(max_generations):

            fitness_scores = [self.evaluate_fitness(design) for design in population]

            self._update_best_design(population, fitness_scores)
            self.fitness_history.append(self.best_fitness)

            if self.monitor:
                self.monitor.record_generation(generation, population, fitness_scores, self)

            self.adapt_parameters(self.best_fitness)

            # NEW: Check for diversity collapse and restart if needed
            diversity = self.calculate_population_diversity(population, fitness_scores)
            if diversity < self.diversity_threshold and self.global_stagnation_counter > self.restart_threshold:
                if verbose:
                    print(f"Gen {generation}: Diversity collapse detected. Injecting new individuals...")
                population = self._inject_diversity(population, fitness_scores)
                self.global_stagnation_counter = 0

            population = self._create_next_generation(population, fitness_scores)

            # NEW: Apply local search to elite individuals
            if generation % 10 == 0:
                population = self._apply_local_search(population, fitness_scores)

            self._report(verbose, generation, fitness_scores, population)

        self._finalize_run(max_generations, verbose, save_report_path, population)

        return self.best_design

    # NEW: Inject diversity when population converges too much
    def _inject_diversity(self, population: List[DesignGenetic], fitness_scores: List[float]) -> List[DesignGenetic]:
        """Replace worst individuals with fresh random designs."""
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        sorted_indices = np.argsort(fitness_scores)

        # Keep best individuals
        new_population = [population[i].copy() for i in sorted_indices[:elite_count]]

        # Add some fresh designs
        num_fresh = self.population_size // 4
        for i in range(num_fresh):
            design = DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(self.rng.integers(10000)))
            num_interchanges = self.rng.integers(10, 100)
            for _ in range(num_interchanges):
                if len(design.heap) >= 2:
                    design.iterate(random_pull=True,
                                   switch_coefficient=self.rng.random(),
                                   partitions=self.partitions,
                                   border_units=self.border_units)
            new_population.append(design)

        # Fill rest with mutated elites
        while len(new_population) < self.population_size:
            elite = population[sorted_indices[self.rng.integers(elite_count)]].copy()
            for _ in range(self.rng.integers(5, 30)):
                if len(elite.heap) >= 2:
                    elite.iterate(random_pull=True,
                                  switch_coefficient=self.rng.random(),
                                  partitions=self.partitions,
                                  border_units=self.border_units)
            new_population.append(elite)

        return new_population

    # NEW: Local search to refine elite individuals
    def _apply_local_search(self, population: List[DesignGenetic], fitness_scores: List[float]) -> List[DesignGenetic]:
        """Apply greedy local search to top individuals."""
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        sorted_indices = np.argsort(fitness_scores)

        for idx in sorted_indices[:elite_count]:
            design = population[idx]
            current_fitness = fitness_scores[idx]

            for _ in range(self.local_search_intensity):
                # Try a random change
                candidate = design.copy()
                if len(candidate.heap) >= 2:
                    candidate.iterate(random_pull=True,
                                      switch_coefficient=self.rng.random(),
                                      partitions=self.partitions,
                                      border_units=self.border_units)
                    candidate_fitness = self.evaluate_fitness(candidate)

                    # Accept if better (greedy)
                    if candidate_fitness < current_fitness:
                        population[idx] = candidate
                        current_fitness = candidate_fitness

        return population

    # --- Helper methods for the `run` process ---
    def _update_best_design(self, population: List[DesignGenetic], fitness_scores: List[float]):
        """Checks for and updates the best design found so far."""
        current_best_fitness = min(fitness_scores)
        best_idx = np.argmin(fitness_scores)
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_fitness_y = self.criterion_y(population[best_idx])
            self.best_design = population[best_idx].copy()
            self.global_stagnation_counter = 0
        else:
            self.global_stagnation_counter += 1

    def _report(self,verbose: bool, generation: int, fitness_scores: List[float], population: List[DesignGenetic]):
        if verbose and generation % 5 == 0:
            heap_sizes = [len(d.heap) for d in population]
            print(
                f"Gen {generation} heap sizes - min: {min(heap_sizes)}, max: {max(heap_sizes)}, avg: {sum(heap_sizes) / len(heap_sizes):.1f}")

            diversity = self.calculate_population_diversity([], fitness_scores)
            n = np.round(np.sum(self.inclusions))
            N = len(self.inclusions)
            var_srs_z = N ** 2 * (1 - n / N) * np.var(self.auxiliary_var) / n
            var_srs_y = N ** 2 * (1 - n / N) * np.var(self.main_var) / n
            print(
                f"Gen {generation:>4} | "
                f"Best Fitness = {self.best_fitness:.8f} | "
                f"Best Var(aux) = {var_srs_z / self.best_fitness:.3f} | "
                f"Best Var(main) = {var_srs_y / self.best_fitness_y:.3f} | "
                f"Diversity = {diversity:.3f} | "
                f"MutRate = {self.mutation_rate:.3f}"
            )

            # diversity = self.calculate_population_diversity([], fitness_scores)
            # print(f"Generation {generation: >4}: Best Fitness = {self.best_fitness:.8f}, "
            #       f"Diversity = {diversity:.3f}, MutRate = {self.mutation_rate:.3f}")

            # self._log_generation_progress(generation, fitness_scores)

    def _finalize_run(self, max_generations: int, verbose: bool, save_report_path: Optional[str], population: List[DesignGenetic]):
        """Wraps up the algorithm run, printing summaries and generating reports."""
        if verbose:
            print("=" * 50)
            print(f"Algorithm completed after {max_generations} generations.")
            print(f"Best fitness achieved: {self.best_fitness:.8f}")
            for design in population:
                if not self.validate_design(design):
                    print("some design are failed validation")

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

        while len(new_population) < self.population_size:
            # NEW: Use tournament selection instead of rank-based for more diversity
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            # Avoid same parent
            attempts = 0
            while parent1 is parent2 and attempts < 5:
                parent2 = self._tournament_selection(population, fitness_scores)
                attempts += 1

            child1, child2 = self._create_offspring(parent1, parent2)

            # Add valid children to the new population
            for child in [child1, child2]:
                if len(new_population) < self.population_size:
                    new_population.append(child)

        return new_population

    # NEW: Tournament selection for better exploration
    def _tournament_selection(self, population: List[DesignGenetic], fitness_scores: List[float]) -> DesignGenetic:
        """Select best individual from a random tournament."""
        tournament_indices = self.rng.choice(len(population), size=min(self.tournament_size, len(population)), replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]

    def _select_parent_pair(self, population: List[DesignGenetic], fitness_scores: List[float], parent_counts: dict,
                            max_children: int) -> Tuple[DesignGenetic, DesignGenetic]:
        parent1_idx = self.rank_based_selection(population, fitness_scores, parent_counts, max_children)

        attempts = 0
        while attempts < 10:
            parent2_idx = self.rank_based_selection(population, fitness_scores, parent_counts, max_children)
            if parent1_idx != parent2_idx:
                return population[parent1_idx], population[parent2_idx]
            attempts += 1

        # If selection keeps returning the same index, manually pick a different one
        available_indices = [i for i in range(len(population)) if i != parent1_idx]
        if available_indices:
            parent2_idx = self.rng.choice(available_indices)
            return population[parent1_idx], population[parent2_idx]

        # Last resort: return the same parent twice (will produce similar children via mutation)
        return population[parent1_idx], population[parent1_idx]

    def _create_offspring(self, parent1: DesignGenetic, parent2: DesignGenetic) -> Tuple[DesignGenetic, DesignGenetic]:
        # NEW: Check crossover rate first
        if self.rng.random() > self.crossover_rate:
            # No crossover, just mutate parents
            return self.mutate_design(parent1.copy()), self.mutate_design(parent2.copy())

        try:
            # Use the SAFE crossover method that preserves inclusion probabilities
            child1, child2 = self.optimizer.combine_n_parents([parent1, parent2],
                                                              border_units=self.border_units)
            if not (self.validate_design(child1) and self.validate_design(child2)):
                child1, child2 = self.optimizer.combine_parents_safe(
                    [parent1, parent2],
                    crossover_rate=(self.rng.integers(30, 70) / 100),
                    num_iterations=30,
                )

        except Exception as e:
            print(f"Crossover failed: {e}. Returning mutated parents instead.")
            return self.mutate_design(parent1.copy()), self.mutate_design(parent2.copy())

        mutated_child1 = self.mutate_design(child1)
        mutated_child2 = self.mutate_design(child2)
        return mutated_child1, mutated_child2

    def mutate_design(self, design: DesignGenetic) -> DesignGenetic:
        """Applies mutation to a design by performing interchanges."""
        if self.rng.random() > self.mutation_rate:
            return design  # No mutation occurs

        mutated = design.copy()
        # NEW: Variable mutation intensity based on design state
        actual_intensity = self.rng.integers(1, self.mutation_intensity + 1)

        for _ in range(actual_intensity):
            if len(mutated.heap) >= 2:
                mutated.iterate(
                    random_pull=self.random_pull,
                    switch_coefficient=(self.rng.integers(20, 80) / 100),  # More varied coefficient
                    partitions=self.partitions,
                    border_units=self.border_units
                )
        return mutated

    # --- Core GA Components (Selection, Fitness, etc.) ---
    def evaluate_fitness(self, design: DesignGenetic) -> float:
        """Calculates the fitness of a single design."""


        try:
            self.validate_design(design)
            design.merge_identical()
            return self.criterion(design)
        except Exception as e:
            print(f"Fitness calculation failed for a design: {e}")
            return float('inf')  # Return a very high (bad) fitness value

    def create_initial_population(self) -> List[DesignGenetic]:
        """Creates the starting population with varied individuals."""
        population = []
        # Add a base design
        base_design = DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(42))
        population.append(base_design)
        print(f"DEBUG: Base design heap size: {len(base_design.heap)}")

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
            population.append(design)

        heap_sizes = [len(d.heap) for d in population]
        print(f"DEBUG: Initial population heap sizes - min: {min(heap_sizes)}, max: {max(heap_sizes)}, avg: {sum(heap_sizes)/len(heap_sizes):.1f}")
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

        improvement = self.last_best_fitness - current_best_fitness

        if improvement > 1e-6:
            self.stagnation_counter = 0
            # Decrease mutation on improvement (exploit mode)
            self.mutation_rate *= 0.95
        else:
            self.stagnation_counter += 1

        # Gradual increase in mutation rate during stagnation
        if self.stagnation_counter > 5:
            self.mutation_rate *= 1.1  # Increase mutation on stagnation

        if self.stagnation_counter > 15:
            # More aggressive mutation
            self.mutation_rate = min(0.6, self.mutation_rate * 1.2)
            self.mutation_intensity = min(20, self.mutation_intensity + 1)

        if self.stagnation_counter > 30:  # Major reset if stuck too long
            self.mutation_rate = 0.5
            self.mutation_intensity = 15
            self.stagnation_counter = 0

        self.mutation_rate = np.clip(self.mutation_rate, 0.1, 0.7)
        self.last_best_fitness = current_best_fitness

    # --- Validation and Utility Methods ---
    def validate_design(self, design: DesignGenetic) -> bool:
        """Checks if a design's inclusion probabilities match the target."""
        total_prob = sum(sample.probability for sample in design.heap)
        if abs(total_prob - 1.0) > self.VALIDATION_TOLERANCE:
            print('total probability', total_prob)

        id_probs = {}
        for sample in design.heap:
            for unit in sample.ids:
                id_probs[unit] = id_probs.get(unit, 0) + sample.probability

        for i, expected_prob in enumerate(self.inclusions):
            if abs(id_probs.get(i, 0) - expected_prob) > self.VALIDATION_TOLERANCE:
                # print(f"Validation failed for unit {i}: expected {expected_prob:.6f}, got {id_probs.get(i, 0):.6f}")
                return False
        return True

    def calculate_population_diversity(self, population: List[DesignGenetic], fitness_scores: List[float]) -> float:
        """Calculates a diversity score based on fitness and structural variance."""
        if len(fitness_scores) < 2:
            return 0.0

        fitness_mean = np.mean(fitness_scores)
        fitness_diversity = np.std(fitness_scores) / (abs(fitness_mean) + 1e-10)

        return min(1.0, fitness_diversity)
