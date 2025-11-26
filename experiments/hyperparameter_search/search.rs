//! # Algoritmos de Busca de Hiperparâmetros
//!
//! Implementações de diferentes estratégias de otimização:
//! - Grid Search: Busca exaustiva em grade
//! - Random Search: Amostragem aleatória
//! - Bayesian Optimization: Otimização baseada em modelo surrogate
//! - Evolutionary Search: Algoritmos genéticos

use std::collections::HashMap;
use super::param_space::{ParameterSpace, ParameterValue, ParameterRange};

/// Resultado de uma busca
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Configuração de parâmetros
    pub config: HashMap<String, ParameterValue>,
    /// Score obtido
    pub score: f64,
    /// Métricas detalhadas
    pub metrics: HashMap<String, f64>,
    /// Número do trial
    pub trial_number: usize,
}

impl SearchResult {
    pub fn new(config: HashMap<String, ParameterValue>, score: f64) -> Self {
        Self {
            config,
            score,
            metrics: HashMap::new(),
            trial_number: 0,
        }
    }
}

/// Trait para estratégias de busca
pub trait SearchStrategy {
    /// Sugere próxima configuração a testar
    fn suggest(&mut self, space: &ParameterSpace) -> HashMap<String, ParameterValue>;

    /// Registra resultado de um trial
    fn register_result(&mut self, result: SearchResult);

    /// Retorna melhor resultado até agora
    fn best_result(&self) -> Option<&SearchResult>;

    /// Retorna histórico de resultados
    fn history(&self) -> &[SearchResult];

    /// Nome da estratégia
    fn name(&self) -> &str;
}

// =============================================================================
// GRID SEARCH
// =============================================================================

/// Busca em grade exaustiva
pub struct GridSearch {
    /// Pontos por parâmetro contínuo
    points_per_param: usize,
    /// Índice atual no grid
    current_index: usize,
    /// Grid pré-computado
    grid: Vec<HashMap<String, ParameterValue>>,
    /// Resultados obtidos
    results: Vec<SearchResult>,
}

impl GridSearch {
    pub fn new(points_per_param: usize) -> Self {
        Self {
            points_per_param,
            current_index: 0,
            grid: Vec::new(),
            results: Vec::new(),
        }
    }

    fn ensure_grid(&mut self, space: &ParameterSpace) {
        if self.grid.is_empty() {
            self.grid = space.generate_grid(self.points_per_param);
        }
    }
}

impl SearchStrategy for GridSearch {
    fn suggest(&mut self, space: &ParameterSpace) -> HashMap<String, ParameterValue> {
        self.ensure_grid(space);

        let config = if self.current_index < self.grid.len() {
            self.grid[self.current_index].clone()
        } else {
            // Retorna config aleatória se grid acabou
            let mut seed = self.current_index as u64;
            let mut rng = || {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed >> 16) & 0x7fff) as f64 / 32768.0
            };
            space.sample(&mut rng)
        };

        self.current_index += 1;
        config
    }

    fn register_result(&mut self, result: SearchResult) {
        self.results.push(result);
    }

    fn best_result(&self) -> Option<&SearchResult> {
        self.results.iter().max_by(|a, b| {
            a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn history(&self) -> &[SearchResult] {
        &self.results
    }

    fn name(&self) -> &str {
        "GridSearch"
    }
}

// =============================================================================
// RANDOM SEARCH
// =============================================================================

/// Busca aleatória com amostragem uniforme
pub struct RandomSearch {
    /// Semente para RNG
    seed: u64,
    /// Contador de trials
    trial_count: usize,
    /// Resultados obtidos
    results: Vec<SearchResult>,
}

impl RandomSearch {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            trial_count: 0,
            results: Vec::new(),
        }
    }

    fn next_random(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (u32::MAX as f64)
    }
}

impl SearchStrategy for RandomSearch {
    fn suggest(&mut self, space: &ParameterSpace) -> HashMap<String, ParameterValue> {
        self.trial_count += 1;
        let mut rng = || self.next_random();
        space.sample(&mut rng)
    }

    fn register_result(&mut self, result: SearchResult) {
        self.results.push(result);
    }

    fn best_result(&self) -> Option<&SearchResult> {
        self.results.iter().max_by(|a, b| {
            a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn history(&self) -> &[SearchResult] {
        &self.results
    }

    fn name(&self) -> &str {
        "RandomSearch"
    }
}

// =============================================================================
// BAYESIAN OPTIMIZATION
// =============================================================================

/// Ponto no espaço de busca (vetor normalizado)
#[derive(Debug, Clone)]
struct BayesianPoint {
    values: Vec<f64>,
    score: f64,
}

/// Otimização Bayesiana com Gaussian Process simplificado
pub struct BayesianSearch {
    /// Semente para RNG
    seed: u64,
    /// Pontos observados
    observations: Vec<BayesianPoint>,
    /// Resultados completos
    results: Vec<SearchResult>,
    /// Fator de exploração (kappa no UCB)
    exploration_factor: f64,
    /// Número de amostras para otimização da aquisição
    acquisition_samples: usize,
    /// Nomes dos parâmetros em ordem
    param_names: Vec<String>,
}

impl BayesianSearch {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            observations: Vec::new(),
            results: Vec::new(),
            exploration_factor: 2.0,
            acquisition_samples: 1000,
            param_names: Vec::new(),
        }
    }

    pub fn with_exploration(mut self, kappa: f64) -> Self {
        self.exploration_factor = kappa;
        self
    }

    fn next_random(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (u32::MAX as f64)
    }

    /// Converte config para vetor normalizado [0, 1]
    fn config_to_vector(&self, config: &HashMap<String, ParameterValue>, space: &ParameterSpace) -> Vec<f64> {
        self.param_names.iter().map(|name| {
            let value = config.get(name);
            let def = space.parameters.get(name);

            match (value, def) {
                (Some(ParameterValue::Float(v)), Some(def)) => {
                    if let ParameterRange::Continuous { min, max, .. } = &def.range {
                        (v - min) / (max - min)
                    } else {
                        0.5
                    }
                }
                (Some(ParameterValue::Int(v)), Some(def)) => {
                    if let ParameterRange::Integer { min, max } = &def.range {
                        (*v - min) as f64 / (max - min) as f64
                    } else {
                        0.5
                    }
                }
                (Some(ParameterValue::Bool(v)), _) => if *v { 1.0 } else { 0.0 },
                _ => 0.5,
            }
        }).collect()
    }

    /// Converte vetor normalizado para config
    fn vector_to_config(&self, vector: &[f64], space: &ParameterSpace) -> HashMap<String, ParameterValue> {
        self.param_names.iter().enumerate().map(|(i, name)| {
            let v = vector.get(i).copied().unwrap_or(0.5);
            let def = space.parameters.get(name).unwrap();

            let value = match &def.range {
                ParameterRange::Continuous { min, max, .. } => {
                    ParameterValue::Float(min + v * (max - min))
                }
                ParameterRange::Integer { min, max } => {
                    ParameterValue::Int(min + (v * (*max - *min) as f64) as i64)
                }
                ParameterRange::Boolean => {
                    ParameterValue::Bool(v > 0.5)
                }
                ParameterRange::Categorical { options } => {
                    let idx = (v * options.len() as f64) as usize;
                    ParameterValue::String(options[idx.min(options.len() - 1)].clone())
                }
            };

            (name.clone(), value)
        }).collect()
    }

    /// Kernel RBF (Radial Basis Function)
    fn rbf_kernel(&self, x1: &[f64], x2: &[f64], length_scale: f64) -> f64 {
        let sq_dist: f64 = x1.iter().zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        (-sq_dist / (2.0 * length_scale * length_scale)).exp()
    }

    /// Predição do GP (média e variância) usando método simplificado
    fn gp_predict(&self, x: &[f64], length_scale: f64) -> (f64, f64) {
        if self.observations.is_empty() {
            return (0.0, 1.0);
        }

        // Kernel weights
        let weights: Vec<f64> = self.observations.iter()
            .map(|obs| self.rbf_kernel(x, &obs.values, length_scale))
            .collect();

        let total_weight: f64 = weights.iter().sum();

        if total_weight < 1e-10 {
            return (0.0, 1.0);
        }

        // Média ponderada
        let mean: f64 = weights.iter()
            .zip(self.observations.iter())
            .map(|(w, obs)| w * obs.score)
            .sum::<f64>() / total_weight;

        // Variância (distância do ponto mais próximo como proxy)
        let max_kernel = weights.iter().fold(0.0f64, |a, &b| a.max(b));
        let variance = (1.0 - max_kernel).max(0.01);

        (mean, variance)
    }

    /// Upper Confidence Bound acquisition function
    fn ucb(&self, x: &[f64]) -> f64 {
        let (mean, variance) = self.gp_predict(x, 0.2);
        mean + self.exploration_factor * variance.sqrt()
    }
}

impl SearchStrategy for BayesianSearch {
    fn suggest(&mut self, space: &ParameterSpace) -> HashMap<String, ParameterValue> {
        // Inicializa nomes na primeira chamada
        if self.param_names.is_empty() {
            self.param_names = space.parameters.keys().cloned().collect();
            self.param_names.sort();
        }

        let n_params = self.param_names.len();

        // Fase de exploração inicial
        if self.observations.len() < n_params.min(10) {
            let mut rng = || self.next_random();
            return space.sample(&mut rng);
        }

        // Otimiza função de aquisição via amostragem
        let mut best_x: Vec<f64> = (0..n_params).map(|_| self.next_random()).collect();
        let mut best_acq = self.ucb(&best_x);

        for _ in 0..self.acquisition_samples {
            let x: Vec<f64> = (0..n_params).map(|_| self.next_random()).collect();
            let acq = self.ucb(&x);

            if acq > best_acq {
                best_acq = acq;
                best_x = x;
            }
        }

        self.vector_to_config(&best_x, space)
    }

    fn register_result(&mut self, result: SearchResult) {
        if !self.param_names.is_empty() {
            // Precisamos reconstruir o space para converter
            let values: Vec<f64> = self.param_names.iter().map(|name| {
                match result.config.get(name) {
                    Some(ParameterValue::Float(v)) => *v,
                    Some(ParameterValue::Int(v)) => *v as f64,
                    Some(ParameterValue::Bool(v)) => if *v { 1.0 } else { 0.0 },
                    _ => 0.5,
                }
            }).collect();

            // Normaliza usando min/max observados
            let normalized: Vec<f64> = values.iter().enumerate().map(|(i, &v)| {
                let col_values: Vec<f64> = self.observations.iter()
                    .map(|obs| obs.values.get(i).copied().unwrap_or(0.5))
                    .chain(std::iter::once(v))
                    .collect();
                let min = col_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = col_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                if (max - min).abs() < 1e-10 { 0.5 } else { (v - min) / (max - min) }
            }).collect();

            self.observations.push(BayesianPoint {
                values: normalized,
                score: result.score,
            });
        }

        self.results.push(result);
    }

    fn best_result(&self) -> Option<&SearchResult> {
        self.results.iter().max_by(|a, b| {
            a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn history(&self) -> &[SearchResult] {
        &self.results
    }

    fn name(&self) -> &str {
        "BayesianOptimization"
    }
}

// =============================================================================
// EVOLUTIONARY SEARCH
// =============================================================================

/// Indivíduo na população
#[derive(Debug, Clone)]
struct Individual {
    config: HashMap<String, ParameterValue>,
    fitness: Option<f64>,
}

/// Busca evolutiva com algoritmo genético
pub struct EvolutionarySearch {
    /// Semente para RNG
    seed: u64,
    /// Tamanho da população
    population_size: usize,
    /// Taxa de mutação
    mutation_rate: f64,
    /// Taxa de crossover
    crossover_rate: f64,
    /// Pressão de seleção (tournament size)
    tournament_size: usize,
    /// Elitismo (número de melhores preservados)
    elitism: usize,
    /// População atual
    population: Vec<Individual>,
    /// Índice do próximo indivíduo a avaliar
    eval_index: usize,
    /// Geração atual
    generation: usize,
    /// Resultados obtidos
    results: Vec<SearchResult>,
}

impl EvolutionarySearch {
    pub fn new(seed: u64, population_size: usize) -> Self {
        Self {
            seed,
            population_size,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            tournament_size: 3,
            elitism: 2,
            population: Vec::new(),
            eval_index: 0,
            generation: 0,
            results: Vec::new(),
        }
    }

    pub fn with_mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    pub fn with_crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    fn next_random(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (u32::MAX as f64)
    }

    fn initialize_population(&mut self, space: &ParameterSpace) {
        self.population.clear();
        for _ in 0..self.population_size {
            let mut rng = || self.next_random();
            let config = space.sample(&mut rng);
            self.population.push(Individual {
                config,
                fitness: None,
            });
        }
        self.eval_index = 0;
    }

    /// Seleção por torneio
    fn tournament_select(&mut self) -> usize {
        let evaluated: Vec<(usize, f64)> = self.population.iter().enumerate()
            .filter_map(|(i, ind)| ind.fitness.map(|f| (i, f)))
            .collect();

        if evaluated.is_empty() {
            return 0;
        }

        let mut best_idx = evaluated[0].0;
        let mut best_fitness = evaluated[0].1;

        for _ in 0..self.tournament_size {
            let idx = (self.next_random() * evaluated.len() as f64) as usize;
            let (i, f) = evaluated[idx.min(evaluated.len() - 1)];
            if f > best_fitness {
                best_fitness = f;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Crossover de dois configs
    fn crossover(&mut self, parent1: &HashMap<String, ParameterValue>,
                 parent2: &HashMap<String, ParameterValue>) -> HashMap<String, ParameterValue> {
        parent1.iter().map(|(key, v1)| {
            let value = if self.next_random() < 0.5 {
                v1.clone()
            } else {
                parent2.get(key).cloned().unwrap_or_else(|| v1.clone())
            };
            (key.clone(), value)
        }).collect()
    }

    /// Mutação de um config
    fn mutate(&mut self, config: &mut HashMap<String, ParameterValue>, space: &ParameterSpace) {
        for (name, value) in config.iter_mut() {
            if self.next_random() < self.mutation_rate {
                if let Some(def) = space.parameters.get(name) {
                    let mut rng = || self.next_random();
                    *value = def.range.sample(&mut rng);
                }
            }
        }
    }

    /// Evolui para próxima geração
    fn evolve(&mut self, space: &ParameterSpace) {
        // Ordena por fitness
        let mut sorted: Vec<(usize, f64)> = self.population.iter().enumerate()
            .filter_map(|(i, ind)| ind.fitness.map(|f| (i, f)))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut new_population = Vec::with_capacity(self.population_size);

        // Elitismo
        for &(idx, _) in sorted.iter().take(self.elitism) {
            new_population.push(Individual {
                config: self.population[idx].config.clone(),
                fitness: None, // Reset para reavaliação
            });
        }

        // Gera resto da população
        while new_population.len() < self.population_size {
            let p1_idx = self.tournament_select();
            let p2_idx = self.tournament_select();

            // Clone configs first to avoid borrow issues
            let parent1_config = self.population[p1_idx].config.clone();
            let parent2_config = self.population[p2_idx].config.clone();

            let mut child_config = if self.next_random() < self.crossover_rate {
                self.crossover(&parent1_config, &parent2_config)
            } else {
                parent1_config
            };

            self.mutate(&mut child_config, space);

            new_population.push(Individual {
                config: child_config,
                fitness: None,
            });
        }

        self.population = new_population;
        self.eval_index = 0;
        self.generation += 1;
    }
}

impl SearchStrategy for EvolutionarySearch {
    fn suggest(&mut self, space: &ParameterSpace) -> HashMap<String, ParameterValue> {
        // Inicializa população se necessário
        if self.population.is_empty() {
            self.initialize_population(space);
        }

        // Se todos foram avaliados, evolui
        if self.eval_index >= self.population.len() {
            self.evolve(space);
        }

        let config = self.population[self.eval_index].config.clone();
        config
    }

    fn register_result(&mut self, result: SearchResult) {
        if self.eval_index < self.population.len() {
            self.population[self.eval_index].fitness = Some(result.score);
            self.eval_index += 1;
        }
        self.results.push(result);
    }

    fn best_result(&self) -> Option<&SearchResult> {
        self.results.iter().max_by(|a, b| {
            a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn history(&self) -> &[SearchResult] {
        &self.results
    }

    fn name(&self) -> &str {
        "EvolutionarySearch"
    }
}

// =============================================================================
// TESTES
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::param_space::create_full_parameter_space;

    #[test]
    fn test_random_search() {
        let space = create_full_parameter_space().filter_by_importance(0.8);
        let mut search = RandomSearch::new(42);

        for i in 0..10 {
            let config = search.suggest(&space);
            assert!(!config.is_empty());

            let result = SearchResult {
                config,
                score: i as f64 * 0.1,
                metrics: HashMap::new(),
                trial_number: i,
            };
            search.register_result(result);
        }

        assert_eq!(search.history().len(), 10);
        assert!(search.best_result().is_some());
    }

    #[test]
    fn test_bayesian_search() {
        let space = create_full_parameter_space().filter_by_importance(0.9);
        let mut search = BayesianSearch::new(42);

        for i in 0..15 {
            let config = search.suggest(&space);
            assert!(!config.is_empty());

            let result = SearchResult {
                config,
                score: (i as f64 * 0.05).sin().abs(),
                metrics: HashMap::new(),
                trial_number: i,
            };
            search.register_result(result);
        }

        assert!(search.best_result().is_some());
    }

    #[test]
    fn test_evolutionary_search() {
        let space = create_full_parameter_space().filter_by_importance(0.8);
        let mut search = EvolutionarySearch::new(42, 10);

        // Avalia duas gerações
        for i in 0..20 {
            let config = search.suggest(&space);
            assert!(!config.is_empty());

            let result = SearchResult {
                config,
                score: i as f64 * 0.05,
                metrics: HashMap::new(),
                trial_number: i,
            };
            search.register_result(result);
        }

        assert!(search.generation >= 1);
        assert!(search.best_result().is_some());
    }

    #[test]
    fn test_grid_search() {
        let mut space = ParameterSpace::new();
        space.add_parameter(
            super::super::param_space::ParameterDef::new(
                "test_param",
                "test",
                "Test parameter",
                ParameterRange::continuous(0.0, 1.0),
                ParameterValue::Float(0.5),
            )
        );

        let mut search = GridSearch::new(3);

        for i in 0..5 {
            let config = search.suggest(&space);
            let result = SearchResult {
                config,
                score: i as f64,
                metrics: HashMap::new(),
                trial_number: i,
            };
            search.register_result(result);
        }

        assert_eq!(search.history().len(), 5);
    }
}
