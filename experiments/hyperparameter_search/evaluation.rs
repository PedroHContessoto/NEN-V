//! # Sistema de Avaliação e Benchmarks
//!
//! Define benchmarks e métricas para avaliar configurações de rede neural.
//!
//! ## Tipos de Benchmark
//!
//! - **TaskBenchmark**: Avalia performance em tarefas específicas
//! - **ConvergenceBenchmark**: Mede velocidade de convergência
//! - **StabilityBenchmark**: Avalia estabilidade do aprendizado
//! - **EfficiencyBenchmark**: Mede eficiência energética e computacional

use std::collections::HashMap;
use super::param_space::ParameterValue;

/// Métricas de avaliação
#[derive(Debug, Clone, Default)]
pub struct EvaluationMetrics {
    /// Score principal (objetivo de otimização)
    pub primary_score: f64,
    /// Reward total obtido
    pub total_reward: f64,
    /// Taxa de sucesso em tarefas
    pub success_rate: f64,
    /// Velocidade de convergência (steps até critério)
    pub convergence_speed: f64,
    /// Estabilidade (variância da performance)
    pub stability: f64,
    /// Eficiência energética
    pub energy_efficiency: f64,
    /// Taxa de disparo média
    pub firing_rate: f64,
    /// Free energy média
    pub free_energy: f64,
    /// Métricas extras
    pub extras: HashMap<String, f64>,
}

impl EvaluationMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_primary_score(mut self, score: f64) -> Self {
        self.primary_score = score;
        self
    }

    /// Combina múltiplas métricas em score único
    pub fn compute_combined_score(&mut self, weights: &MetricWeights) {
        let normalized_reward = self.total_reward.tanh();
        let normalized_convergence = (1.0 / (1.0 + self.convergence_speed / 1000.0));
        let normalized_stability = self.stability.clamp(0.0, 1.0);
        let normalized_efficiency = self.energy_efficiency.clamp(0.0, 1.0);

        self.primary_score = weights.reward * normalized_reward
            + weights.success * self.success_rate
            + weights.convergence * normalized_convergence
            + weights.stability * normalized_stability
            + weights.efficiency * normalized_efficiency;
    }
}

/// Pesos para combinar métricas
#[derive(Debug, Clone)]
pub struct MetricWeights {
    pub reward: f64,
    pub success: f64,
    pub convergence: f64,
    pub stability: f64,
    pub efficiency: f64,
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self {
            reward: 0.4,
            success: 0.3,
            convergence: 0.1,
            stability: 0.1,
            efficiency: 0.1,
        }
    }
}

/// Resultado de um benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Nome do benchmark
    pub name: String,
    /// Métricas obtidas
    pub metrics: EvaluationMetrics,
    /// Tempo de execução (ms)
    pub duration_ms: u64,
    /// Configuração testada
    pub config: HashMap<String, ParameterValue>,
    /// Número de steps executados
    pub steps: usize,
    /// Episódios executados
    pub episodes: usize,
}

impl BenchmarkResult {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            metrics: EvaluationMetrics::new(),
            duration_ms: 0,
            config: HashMap::new(),
            steps: 0,
            episodes: 0,
        }
    }
}

/// Trait para benchmarks
pub trait Benchmark {
    /// Nome do benchmark
    fn name(&self) -> &str;

    /// Executa o benchmark com uma configuração
    fn run(&self, config: &HashMap<String, ParameterValue>, seed: u64) -> BenchmarkResult;

    /// Descrição do benchmark
    fn description(&self) -> &str;
}

// =============================================================================
// TASK BENCHMARK - Avalia performance em tarefas
// =============================================================================

/// Configuração de tarefa
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// Nome da tarefa
    pub name: String,
    /// Número de episódios
    pub episodes: usize,
    /// Steps máximos por episódio
    pub max_steps: usize,
    /// Threshold de sucesso
    pub success_threshold: f64,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            name: "navigation".to_string(),
            episodes: 50,
            max_steps: 500,
            success_threshold: 10.0,
        }
    }
}

/// Benchmark de performance em tarefa
pub struct TaskBenchmark {
    config: TaskConfig,
}

impl TaskBenchmark {
    pub fn new(config: TaskConfig) -> Self {
        Self { config }
    }

    pub fn navigation(episodes: usize, max_steps: usize) -> Self {
        Self::new(TaskConfig {
            name: "navigation".to_string(),
            episodes,
            max_steps,
            success_threshold: 10.0,
        })
    }

    /// Simula execução da rede (placeholder - seria integrado com simulação real)
    fn simulate_network(&self, _config: &HashMap<String, ParameterValue>, seed: u64) -> (f64, f64, f64) {
        // Esta função seria substituída pela integração real com a rede
        // Por agora, retorna valores baseados no seed para teste
        let mut rng_state = seed;
        let mut next_random = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };

        let total_reward = next_random() * 20.0 - 5.0;
        let success_rate = next_random();
        let avg_steps = next_random() * self.config.max_steps as f64;

        (total_reward, success_rate, avg_steps)
    }
}

impl Benchmark for TaskBenchmark {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn run(&self, config: &HashMap<String, ParameterValue>, seed: u64) -> BenchmarkResult {
        let start = std::time::Instant::now();

        let (total_reward, success_rate, avg_steps) = self.simulate_network(config, seed);

        let mut metrics = EvaluationMetrics::new();
        metrics.total_reward = total_reward;
        metrics.success_rate = success_rate;
        metrics.convergence_speed = avg_steps;

        let weights = MetricWeights::default();
        metrics.compute_combined_score(&weights);

        BenchmarkResult {
            name: self.config.name.clone(),
            metrics,
            duration_ms: start.elapsed().as_millis() as u64,
            config: config.clone(),
            steps: (avg_steps as usize) * self.config.episodes,
            episodes: self.config.episodes,
        }
    }

    fn description(&self) -> &str {
        "Avalia performance em tarefas de navegação/RL"
    }
}

// =============================================================================
// CONVERGENCE BENCHMARK - Mede velocidade de convergência
// =============================================================================

/// Benchmark de convergência
pub struct ConvergenceBenchmark {
    /// Número máximo de steps
    max_steps: usize,
    /// Threshold de convergência
    convergence_threshold: f64,
    /// Tamanho da janela para detectar convergência
    window_size: usize,
}

impl ConvergenceBenchmark {
    pub fn new(max_steps: usize, convergence_threshold: f64) -> Self {
        Self {
            max_steps,
            convergence_threshold,
            window_size: 100,
        }
    }

    pub fn default_config() -> Self {
        Self::new(10000, 0.01)
    }
}

impl Benchmark for ConvergenceBenchmark {
    fn name(&self) -> &str {
        "convergence"
    }

    fn run(&self, config: &HashMap<String, ParameterValue>, seed: u64) -> BenchmarkResult {
        let start = std::time::Instant::now();

        // Simulação de convergência
        let mut rng_state = seed;
        let mut next_random = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };

        // Extrai learning rate do config se disponível
        let learning_rate = config.get("learning.base_learning_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.01);

        // Simula convergência (learning rate maior = convergência mais rápida, mas menos estável)
        let convergence_factor = learning_rate.sqrt();
        let steps_to_converge = ((1.0 - convergence_factor) * self.max_steps as f64) as usize;
        let stability = (1.0 - learning_rate * 5.0).max(0.0);

        let mut metrics = EvaluationMetrics::new();
        metrics.convergence_speed = steps_to_converge as f64;
        metrics.stability = stability + next_random() * 0.1;

        let weights = MetricWeights {
            reward: 0.0,
            success: 0.0,
            convergence: 0.6,
            stability: 0.4,
            efficiency: 0.0,
        };
        metrics.compute_combined_score(&weights);

        BenchmarkResult {
            name: "convergence".to_string(),
            metrics,
            duration_ms: start.elapsed().as_millis() as u64,
            config: config.clone(),
            steps: steps_to_converge,
            episodes: 1,
        }
    }

    fn description(&self) -> &str {
        "Mede velocidade de convergência do aprendizado"
    }
}

// =============================================================================
// STABILITY BENCHMARK - Avalia estabilidade
// =============================================================================

/// Benchmark de estabilidade
pub struct StabilityBenchmark {
    /// Número de runs para medir variância
    num_runs: usize,
    /// Steps por run
    steps_per_run: usize,
}

impl StabilityBenchmark {
    pub fn new(num_runs: usize, steps_per_run: usize) -> Self {
        Self {
            num_runs,
            steps_per_run,
        }
    }

    pub fn default_config() -> Self {
        Self::new(5, 1000)
    }
}

impl Benchmark for StabilityBenchmark {
    fn name(&self) -> &str {
        "stability"
    }

    fn run(&self, config: &HashMap<String, ParameterValue>, seed: u64) -> BenchmarkResult {
        let start = std::time::Instant::now();

        let mut rng_state = seed;
        let mut next_random = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };

        // Simula múltiplas runs
        let mut rewards: Vec<f64> = Vec::new();

        for _ in 0..self.num_runs {
            let reward = next_random() * 10.0 + 5.0;
            rewards.push(reward);
        }

        // Calcula estatísticas
        let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
        let variance = rewards.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / rewards.len() as f64;
        let std_dev = variance.sqrt();

        // Coeficiente de variação (menor = mais estável)
        let cv = std_dev / mean.abs().max(0.001);
        let stability = (1.0 - cv).clamp(0.0, 1.0);

        let mut metrics = EvaluationMetrics::new();
        metrics.total_reward = mean;
        metrics.stability = stability;

        let weights = MetricWeights {
            reward: 0.3,
            success: 0.0,
            convergence: 0.0,
            stability: 0.7,
            efficiency: 0.0,
        };
        metrics.compute_combined_score(&weights);

        BenchmarkResult {
            name: "stability".to_string(),
            metrics,
            duration_ms: start.elapsed().as_millis() as u64,
            config: config.clone(),
            steps: self.steps_per_run * self.num_runs,
            episodes: self.num_runs,
        }
    }

    fn description(&self) -> &str {
        "Avalia estabilidade do aprendizado através de múltiplas execuções"
    }
}

// =============================================================================
// EFFICIENCY BENCHMARK - Mede eficiência
// =============================================================================

/// Benchmark de eficiência energética/computacional
pub struct EfficiencyBenchmark {
    /// Steps para medição
    num_steps: usize,
}

impl EfficiencyBenchmark {
    pub fn new(num_steps: usize) -> Self {
        Self { num_steps }
    }

    pub fn default_config() -> Self {
        Self::new(1000)
    }
}

impl Benchmark for EfficiencyBenchmark {
    fn name(&self) -> &str {
        "efficiency"
    }

    fn run(&self, config: &HashMap<String, ParameterValue>, seed: u64) -> BenchmarkResult {
        let start = std::time::Instant::now();

        let mut rng_state = seed;
        let mut next_random = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_state >> 33) as f64) / (u32::MAX as f64)
        };

        // Extrai parâmetros relevantes
        let target_firing = config.get("homeostasis.target_firing_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.15);

        let energy_cost = config.get("energy.cost_fire_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.05);

        // Simula eficiência
        let firing_rate = target_firing + (next_random() - 0.5) * 0.05;
        let energy_used = firing_rate * energy_cost * self.num_steps as f64;
        let reward_obtained = (1.0 - (firing_rate - 0.1).abs()) * 10.0 + next_random() * 2.0;

        let efficiency = reward_obtained / energy_used.max(0.01);

        let mut metrics = EvaluationMetrics::new();
        metrics.firing_rate = firing_rate;
        metrics.energy_efficiency = efficiency.tanh();
        metrics.total_reward = reward_obtained;

        let weights = MetricWeights {
            reward: 0.3,
            success: 0.0,
            convergence: 0.0,
            stability: 0.0,
            efficiency: 0.7,
        };
        metrics.compute_combined_score(&weights);

        BenchmarkResult {
            name: "efficiency".to_string(),
            metrics,
            duration_ms: start.elapsed().as_millis() as u64,
            config: config.clone(),
            steps: self.num_steps,
            episodes: 1,
        }
    }

    fn description(&self) -> &str {
        "Mede eficiência energética e computacional"
    }
}

// =============================================================================
// BENCHMARK SUITE - Conjunto de benchmarks
// =============================================================================

/// Suite de benchmarks
pub struct BenchmarkSuite {
    benchmarks: Vec<Box<dyn Benchmark>>,
    weights: HashMap<String, f64>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            weights: HashMap::new(),
        }
    }

    pub fn add_benchmark(&mut self, benchmark: Box<dyn Benchmark>, weight: f64) {
        let name = benchmark.name().to_string();
        self.benchmarks.push(benchmark);
        self.weights.insert(name, weight);
    }

    /// Cria suite padrão
    pub fn default_suite() -> Self {
        let mut suite = Self::new();

        suite.add_benchmark(
            Box::new(TaskBenchmark::navigation(50, 500)),
            0.4
        );
        suite.add_benchmark(
            Box::new(ConvergenceBenchmark::default_config()),
            0.3
        );
        suite.add_benchmark(
            Box::new(StabilityBenchmark::default_config()),
            0.2
        );
        suite.add_benchmark(
            Box::new(EfficiencyBenchmark::default_config()),
            0.1
        );

        suite
    }

    /// Executa todos os benchmarks
    pub fn run_all(&self, config: &HashMap<String, ParameterValue>, seed: u64) -> EvaluationMetrics {
        let mut combined = EvaluationMetrics::new();
        let mut total_weight = 0.0;

        for benchmark in &self.benchmarks {
            let result = benchmark.run(config, seed);
            let weight = self.weights.get(benchmark.name()).copied().unwrap_or(1.0);

            combined.primary_score += result.metrics.primary_score * weight;
            combined.total_reward += result.metrics.total_reward * weight;
            combined.success_rate += result.metrics.success_rate * weight;
            combined.stability += result.metrics.stability * weight;
            combined.convergence_speed += result.metrics.convergence_speed * weight;
            combined.energy_efficiency += result.metrics.energy_efficiency * weight;

            total_weight += weight;
        }

        // Normaliza
        if total_weight > 0.0 {
            combined.primary_score /= total_weight;
            combined.total_reward /= total_weight;
            combined.success_rate /= total_weight;
            combined.stability /= total_weight;
            combined.convergence_speed /= total_weight;
            combined.energy_efficiency /= total_weight;
        }

        combined
    }

    /// Número de benchmarks
    pub fn len(&self) -> usize {
        self.benchmarks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.benchmarks.is_empty()
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::default_suite()
    }
}

/// Avaliador principal
pub struct Evaluator {
    suite: BenchmarkSuite,
    base_seed: u64,
}

impl Evaluator {
    pub fn new(suite: BenchmarkSuite, seed: u64) -> Self {
        Self {
            suite,
            base_seed: seed,
        }
    }

    pub fn default_evaluator(seed: u64) -> Self {
        Self::new(BenchmarkSuite::default_suite(), seed)
    }

    /// Avalia uma configuração
    pub fn evaluate(&self, config: &HashMap<String, ParameterValue>, trial: usize) -> EvaluationMetrics {
        let seed = self.base_seed.wrapping_add(trial as u64);
        self.suite.run_all(config, seed)
    }
}

// =============================================================================
// TESTES
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_benchmark() {
        let benchmark = TaskBenchmark::navigation(10, 100);
        let config = HashMap::new();

        let result = benchmark.run(&config, 42);
        assert_eq!(result.name, "navigation");
        assert!(result.metrics.primary_score >= 0.0);
    }

    #[test]
    fn test_convergence_benchmark() {
        let benchmark = ConvergenceBenchmark::default_config();
        let config = HashMap::new();

        let result = benchmark.run(&config, 42);
        assert_eq!(result.name, "convergence");
    }

    #[test]
    fn test_stability_benchmark() {
        let benchmark = StabilityBenchmark::default_config();
        let config = HashMap::new();

        let result = benchmark.run(&config, 42);
        assert_eq!(result.name, "stability");
        assert!(result.metrics.stability >= 0.0 && result.metrics.stability <= 1.0);
    }

    #[test]
    fn test_efficiency_benchmark() {
        let benchmark = EfficiencyBenchmark::default_config();
        let config = HashMap::new();

        let result = benchmark.run(&config, 42);
        assert_eq!(result.name, "efficiency");
    }

    #[test]
    fn test_benchmark_suite() {
        let suite = BenchmarkSuite::default_suite();
        assert!(!suite.is_empty());

        let config = HashMap::new();
        let metrics = suite.run_all(&config, 42);

        assert!(metrics.primary_score >= -1.0 && metrics.primary_score <= 1.0);
    }

    #[test]
    fn test_evaluator() {
        let evaluator = Evaluator::default_evaluator(42);
        let config = HashMap::new();

        let metrics = evaluator.evaluate(&config, 0);
        assert!(metrics.primary_score.is_finite());
    }
}
