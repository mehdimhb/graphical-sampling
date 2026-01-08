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
        crossover_rate: float = 0.9,
        local_search_intensity: int = 5,
        tournament_size: int = 3,
        restart_threshold: int = 50,
        diversity_threshold: float = 0.01,
    ):
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
        self.crossover_rate = crossover_rate
        self.local_search_intensity = local_search_intensity
        self.tournament_size = tournament_size
        self.restart_threshold = restart_threshold
        self.diversity_threshold = diversity_threshold

        self.rng = np.random.default_rng()
        self.optimizer = GeneticOptimizer()
        self.criterion = VarNHT(auxiliary_var, inclusions)
        self.criterion_y = VarNHT(main_var, inclusions)
        self.monitor = GAMonitor(enable_live_plots=enable_live_plots, save_data=save_metrics) if enable_monitoring else None

        self.mutation_rate = mutation_rate if adaptive_parameters else 0.3
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')
        self.global_stagnation_counter = 0

        self.best_design: Optional[DesignGenetic] = None
        self.best_fitness = float('inf')
        self.fitness_history = []

        self.partitions, self.border_units = (
            self.optimizer.partition_design(inclusions.tolist(), 2)
            if self.use_partitions else (None, None)
        )

    def run(self, max_generations: int = 100, verbose: bool = True, save_report_path: Optional[str] = None) -> Optional[DesignGenetic]:
        population = self.create_initial_population()

        for generation in range(max_generations):
            fitness_scores = [self.criterion(design) for design in population]
            self._update_best_design(population, fitness_scores)
            self.fitness_history.append(self.best_fitness)
            self.adapt_parameters(self.best_fitness)

            population = self._create_next_generation(population, fitness_scores)

            if generation % 10 == 0:
                population = self._apply_local_search(population, fitness_scores)

            self._report(verbose, generation, fitness_scores, population)

        self._finalize_run(max_generations, verbose, save_report_path, population)

        # Validate only final best design
        if self.best_design:
            valid = self.validate_design(self.best_design)
            print("✅ Final Best Design Validation:" if valid else "❌ Final Best Design Failed Validation")
        return self.best_design

    # --- Core GA Helpers ---
    def create_initial_population(self) -> List[DesignGenetic]:
        population = [DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(42))]
        for i in range(self.population_size - 1):
            design = DesignGenetic(inclusions=self.inclusions, rng=np.random.default_rng(i))
            for _ in range(self.rng.integers(1, 50)):
                if len(design.heap) >= 2:
                    design.iterate(random_pull=self.random_pull, switch_coefficient=self.rng.random(), partitions=self.partitions, border_units=self.border_units)
            population.append(design)
        return population

    def _update_best_design(self, population: List[DesignGenetic], fitness_scores: List[float]):
        current_best_fitness = min(fitness_scores)
        best_idx = np.argmin(fitness_scores)
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_fitness_y = self.criterion_y(population[best_idx])
            self.best_design = population[best_idx].copy()
            self.global_stagnation_counter = 0
        else:
            self.global_stagnation_counter += 1

    def _create_next_generation(self, population: List[DesignGenetic], fitness_scores: List[float]) -> List[DesignGenetic]:
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        elite_indices = np.argsort(fitness_scores)[:elite_count]
        new_population = [population[i].copy() for i in elite_indices]

        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            attempts = 0
            while parent1 is parent2 and attempts < 5:
                parent2 = self._tournament_selection(population, fitness_scores)
                attempts += 1
            child1, child2 = self._create_offspring(parent1, parent2)
            for child in [child1, child2]:
                if len(new_population) < self.population_size:
                    new_population.append(child)
        return new_population

    def _tournament_selection(self, population: List[DesignGenetic], fitness_scores: List[float]) -> DesignGenetic:
        indices = self.rng.choice(len(population), size=min(self.tournament_size, len(population)), replace=False)
        winner_idx = indices[np.argmin([fitness_scores[i] for i in indices])]
        return population[winner_idx]

    def _create_offspring(self, parent1: DesignGenetic, parent2: DesignGenetic) -> Tuple[DesignGenetic, DesignGenetic]:
        if self.rng.random() > self.crossover_rate:
            return self.mutate_design(parent1.copy()), self.mutate_design(parent2.copy())
        try:
            child1, child2 = self.optimizer.combine_n_parents([parent1, parent2], border_units=self.border_units)
            if not (self.validate_design(child1) and self.validate_design(child2)):
                child1, child2 = self.optimizer.combine_parents_safe([parent1, parent2], crossover_rate=self.rng.random(), num_iterations=30)
        except Exception:
            return self.mutate_design(parent1.copy()), self.mutate_design(parent2.copy())
        return self.mutate_design(child1), self.mutate_design(child2)

    def mutate_design(self, design: DesignGenetic) -> DesignGenetic:
        if self.rng.random() > self.mutation_rate:
            return design
        for _ in range(self.rng.integers(1, self.mutation_intensity + 1)):
            if len(design.heap) >= 2:
                design.iterate(random_pull=self.random_pull, switch_coefficient=self.rng.random(), partitions=self.partitions, border_units=self.border_units)
        return design

    def adapt_parameters(self, current_best_fitness: float):
        if not self.adaptive_parameters:
            return
        improvement = self.last_best_fitness - current_best_fitness
        if improvement > 1e-6:
            self.stagnation_counter = 0
            self.mutation_rate *= 0.95
        else:
            self.stagnation_counter += 1
        if self.stagnation_counter > 5:
            self.mutation_rate *= 1.1
        if self.stagnation_counter > 15:
            self.mutation_rate = min(0.6, self.mutation_rate * 1.2)
            self.mutation_intensity = min(20, self.mutation_intensity + 1)
        if self.stagnation_counter > 30:
            self.mutation_rate = 0.5
            self.mutation_intensity = 15
            self.stagnation_counter = 0
        self.mutation_rate = np.clip(self.mutation_rate, 0.1, 0.7)
        self.last_best_fitness = current_best_fitness

    # --- Validation & Metrics ---
    def validate_design(self, design: DesignGenetic) -> bool:
        total_prob = sum(sample.probability for sample in design.heap)
        id_probs = {}
        for sample in design.heap:
            for unit in sample.ids:
                id_probs[unit] = id_probs.get(unit, 0) + sample.probability
        for i, expected_prob in enumerate(self.inclusions):
            if abs(id_probs.get(i, 0) - expected_prob) > self.VALIDATION_TOLERANCE:
                return False
        return True

    def calculate_population_diversity(self, population: List[DesignGenetic], fitness_scores: List[float]) -> float:
        if len(fitness_scores) < 2:
            return 0.0
        return min(1.0, np.std(fitness_scores) / (abs(np.mean(fitness_scores)) + 1e-10))

    def _apply_local_search(self, population: List[DesignGenetic], fitness_scores: List[float]) -> List[DesignGenetic]:
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        sorted_indices = np.argsort(fitness_scores)
        for idx in sorted_indices[:elite_count]:
            design = population[idx]
            current_fitness = fitness_scores[idx]
            for _ in range(self.local_search_intensity):
                candidate = design.copy()
                if len(candidate.heap) >= 2:
                    candidate.iterate(random_pull=True, switch_coefficient=self.rng.random(), partitions=self.partitions, border_units=self.border_units)
                    candidate_fitness = self.evaluate_fitness(candidate)
                    if candidate_fitness < current_fitness:
                        population[idx] = candidate
                        current_fitness = candidate_fitness
        return population

    def evaluate_fitness(self, design: DesignGenetic) -> float:
        try:
            design.merge_identical()
            return self.criterion(design)
        except Exception:
            return float('inf')
    def _report(self, verbose: bool, generation: int, fitness_scores: List[DesignGenetic], population: List[DesignGenetic]):
        if verbose and generation % 10 == 0:
            heap_sizes = [len(d.heap) for d in population]
            diversity = self.calculate_population_diversity([], fitness_scores)
            n = np.round(np.sum(self.inclusions))
            N = len(self.inclusions)
            var_srs_z = N ** 2 * (1 - n / N) * np.var(self.auxiliary_var) / n
            var_srs_y = N ** 2 * (1 - n / N) * np.var(self.main_var) / n
            print(
            f"Gen {generation:>4} | "
            f"Eff_z GA / A* = {(var_srs_z / self.best_fitness / 2.1):.3f} | "
            f"Eff_y GA / A* = {(var_srs_y / self.best_fitness_y / 2.6):.3f} | "
            f"Diversity = {diversity:.3f} | "
            f"MutRate = {self.mutation_rate:.3f}"
              )
    def _finalize_run(self, max_generations: int, verbose: bool, save_report_path: Optional[str], population: List[DesignGenetic]):
        if verbose:
            print("="*50)
            print(f"Algorithm completed after {max_generations} generations.")
            print(f"Best fitness achieved: {self.best_fitness:.8f}")
            if self.monitor:
                self.monitor.generate_final_report(save_path=save_report_path)