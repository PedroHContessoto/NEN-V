//! # Sistema de Avaliação Real com Rede Neural NEN-V Integrada
//!
//! Avalia configurações de hiperparâmetros executando a rede neural NEN-V real
//! em ambientes de benchmark, coletando métricas de performance genuínas.
//!
//! ## Arquitetura
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      REAL EVALUATION SYSTEM                             │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
//! │  │   Environment   │◄──▶│   NENV Agent    │◄──▶│    Metrics      │    │
//! │  │   (Real Task)   │    │   (Real NEN-V)  │    │   Collector     │    │
//! │  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
//! │           │                      │                      │             │
//! │           ▼                      ▼                      ▼             │
//! │  ┌─────────────────────────────────────────────────────────────┐      │
//! │  │                    Benchmark Runner                          │      │
//! │  │  • Runs N episodes per environment                          │      │
//! │  │  • Collects: reward, success, stability, efficiency         │      │
//! │  │  • Computes weighted combined score                         │      │
//! │  └─────────────────────────────────────────────────────────────┘      │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Adicionando Novos Benchmarks
//!
//! 1. Implemente `trait Benchmark` para seu benchmark
//! 2. Registre no `BenchmarkSuite::add_benchmark()`
//! 3. Configure peso relativo para o score combinado
//!
//! ```rust
//! struct MeuBenchmark { ... }
//!
//! impl Benchmark for MeuBenchmark {
//!     fn name(&self) -> &str { "MeuBenchmark" }
//!     fn run(&self, agent: &mut NENVAgent, seed: u64) -> BenchmarkResult { ... }
//!     fn weight(&self) -> f64 { 0.2 }
//! }
//!
//! suite.add_benchmark(Box::new(MeuBenchmark::new()));
//! ```
//!
//! ## Integração com NEN-V Real
//!
//! O `NENVAgent` usa:
//! - `AutoConfig` para criar a rede a partir da tarefa
//! - `Network` real com STDP, eligibility traces, neuromodulação
//! - `WorkingMemoryPool` para contexto temporal
//! - `CuriosityModule` para exploração intrínseca
//!
//! ## Configuração Externa
//!
//! Os parâmetros podem ser fornecidos via:
//! - HashMap de hiperparâmetros do hyperopt
//! - `EvaluationConfig` para controle fino
//! - `MetricWeights` para ajustar objetivos de otimização

use std::collections::HashMap;
use std::time::Instant;

use nenv_v2::{
    Network, ConnectivityType, LearningMode,
    autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity},
    WorkingMemoryPool,
    intrinsic_motivation::CuriosityModule,
};

use crate::environments::{Environment, NavigationEnv, PatternMemoryEnv, PredictionEnv, AssociationEnv};
use super::param_space::ParameterValue;

// =============================================================================
// CONFIGURAÇÃO DE AVALIAÇÃO
// =============================================================================

/// Configuração externa para avaliação
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Número de episódios por benchmark
    pub episodes_per_benchmark: usize,
    /// Steps máximos por episódio (override do ambiente)
    pub max_steps_per_episode: Option<usize>,
    /// Número de runs para calcular estabilidade
    pub runs_for_stability: usize,
    /// Seed base para reprodutibilidade
    pub base_seed: u64,
    /// Habilita logging detalhado
    pub verbose_logging: bool,
    /// Pesos das métricas
    pub metric_weights: MetricWeights,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            episodes_per_benchmark: 20,
            max_steps_per_episode: None,
            runs_for_stability: 3,
            base_seed: 42,
            verbose_logging: false,
            metric_weights: MetricWeights::default(),
        }
    }
}

impl EvaluationConfig {
    pub fn quick() -> Self {
        Self {
            episodes_per_benchmark: 5,
            runs_for_stability: 1,
            ..Default::default()
        }
    }

    pub fn thorough() -> Self {
        Self {
            episodes_per_benchmark: 50,
            runs_for_stability: 5,
            ..Default::default()
        }
    }
}

// =============================================================================
// NENV AGENT - Agente com Rede Neural NEN-V Real
// =============================================================================

/// Configuração do agente baseado em hiperparâmetros
#[derive(Debug, Clone)]
pub struct AgentConfig {
    // Arquitetura
    pub hidden_neurons: usize,
    pub inhibitory_ratio: f64,
    pub initial_threshold: f64,

    // Aprendizado
    pub learning_rate: f64,
    pub stdp_tau_plus: f64,
    pub stdp_tau_minus: f64,
    pub stdp_a_plus: f64,
    pub stdp_a_minus: f64,

    // Homeostase
    pub target_firing_rate: f64,
    pub homeo_eta: f64,
    pub homeo_interval: i64,

    // Eligibility Traces
    pub use_eligibility: bool,
    pub eligibility_tau: f64,
    pub eligibility_increment: f64,

    // Working Memory
    pub use_working_memory: bool,
    pub wm_capacity: usize,
    pub wm_recurrent_strength: f64,

    // Curiosidade
    pub use_curiosity: bool,
    pub curiosity_scale: f64,

    // Energia
    pub energy_cost_fire: f64,
    pub energy_recovery_rate: f64,

    // Competição
    pub competition_enabled: bool,
    pub competition_strength: f64,

    // Exploração
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub epsilon_min: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            hidden_neurons: 32,
            inhibitory_ratio: 0.2,
            initial_threshold: 0.15,

            learning_rate: 0.01,
            stdp_tau_plus: 20.0,
            stdp_tau_minus: 20.0,
            stdp_a_plus: 0.01,
            stdp_a_minus: 0.0105,

            target_firing_rate: 0.15,
            homeo_eta: 0.001,
            homeo_interval: 100,

            use_eligibility: true,
            eligibility_tau: 100.0,
            eligibility_increment: 1.0,

            use_working_memory: true,
            wm_capacity: 7,
            wm_recurrent_strength: 0.3,

            use_curiosity: true,
            curiosity_scale: 0.1,

            energy_cost_fire: 0.05,
            energy_recovery_rate: 0.02,

            competition_enabled: true,
            competition_strength: 0.3,

            epsilon: 0.3,
            epsilon_decay: 0.995,
            epsilon_min: 0.05,
        }
    }
}

impl AgentConfig {
    /// Cria config a partir de parâmetros do hyperopt
    pub fn from_params(params: &HashMap<String, ParameterValue>) -> Self {
        let mut config = Self::default();

        // Arquitetura
        if let Some(v) = params.get("architecture.hidden_neurons").and_then(|v| v.as_f64()) {
            config.hidden_neurons = (v as usize).max(8).min(256);
        }
        if let Some(v) = params.get("architecture.inhibitory_ratio").and_then(|v| v.as_f64()) {
            config.inhibitory_ratio = v.clamp(0.1, 0.4);
        }
        if let Some(v) = params.get("architecture.initial_threshold").and_then(|v| v.as_f64()) {
            config.initial_threshold = v.clamp(0.05, 0.5);
        }

        // Aprendizado
        if let Some(v) = params.get("learning.base_learning_rate").and_then(|v| v.as_f64()) {
            config.learning_rate = v;
        }
        if let Some(v) = params.get("stdp.tau_plus").and_then(|v| v.as_f64()) {
            config.stdp_tau_plus = v;
        }
        if let Some(v) = params.get("stdp.tau_minus").and_then(|v| v.as_f64()) {
            config.stdp_tau_minus = v;
        }
        if let Some(v) = params.get("stdp.a_plus").and_then(|v| v.as_f64()) {
            config.stdp_a_plus = v;
        }
        if let Some(v) = params.get("stdp.a_minus").and_then(|v| v.as_f64()) {
            config.stdp_a_minus = v;
        }

        // Homeostase
        if let Some(v) = params.get("homeostasis.target_firing_rate").and_then(|v| v.as_f64()) {
            config.target_firing_rate = v;
        }
        if let Some(v) = params.get("homeostasis.homeo_eta").and_then(|v| v.as_f64()) {
            config.homeo_eta = v;
        }

        // Eligibility
        if let Some(v) = params.get("timing.eligibility_trace_tau").and_then(|v| v.as_f64()) {
            config.eligibility_tau = v;
        }

        // Working Memory
        if let Some(v) = params.get("working_memory.capacity").and_then(|v| v.as_f64()) {
            config.wm_capacity = (v as usize).max(3).min(15);
        }
        if let Some(v) = params.get("working_memory.recurrent_strength").and_then(|v| v.as_f64()) {
            config.wm_recurrent_strength = v;
        }

        // Curiosidade
        if let Some(v) = params.get("curiosity.scale").and_then(|v| v.as_f64()) {
            config.curiosity_scale = v;
        }

        // Energia
        if let Some(v) = params.get("energy.cost_fire").and_then(|v| v.as_f64()) {
            config.energy_cost_fire = v;
        }
        if let Some(v) = params.get("energy.recovery_rate").and_then(|v| v.as_f64()) {
            config.energy_recovery_rate = v;
        }

        // Competição
        if let Some(v) = params.get("competition.strength").and_then(|v| v.as_f64()) {
            config.competition_strength = v;
        }

        config
    }
}

/// Agente neural usando a rede NEN-V real
pub struct NENVAgent {
    /// Rede neural NEN-V real
    network: Network,
    /// Working memory
    working_memory: Option<WorkingMemoryPool>,
    /// Módulo de curiosidade
    curiosity: Option<CuriosityModule>,
    /// Configuração
    config: AgentConfig,
    /// Epsilon atual para exploração
    current_epsilon: f64,
    /// Índices dos sensores
    sensor_indices: Vec<usize>,
    /// Índices dos atuadores
    actuator_indices: Vec<usize>,
    /// Seed para RNG
    seed: u64,
    /// Estatísticas
    stats: AgentStats,
    /// Última observação
    last_observation: Vec<f64>,
    /// Última ação
    last_action: usize,
}

impl NENVAgent {
    /// Cria agente para um ambiente específico
    pub fn new(input_size: usize, output_size: usize, config: AgentConfig, seed: u64) -> Self {
        // Calcula total de neurônios
        let total_neurons = input_size + config.hidden_neurons + output_size;

        // Cria rede
        let mut network = Network::new(
            total_neurons,
            ConnectivityType::FullyConnected,
            config.inhibitory_ratio,
            config.initial_threshold,
        );

        // Define índices de camadas
        let sensor_indices: Vec<usize> = (0..input_size).collect();
        let hidden_indices: Vec<usize> = (input_size..input_size + config.hidden_neurons).collect();
        let actuator_indices: Vec<usize> = (input_size + config.hidden_neurons..total_neurons).collect();

        network.set_layer_indices(
            sensor_indices.clone(),
            hidden_indices,
            actuator_indices.clone(),
        );

        // Configura modo de aprendizado
        network.set_learning_mode(LearningMode::STDP);
        network.stdp_window = 50;

        // Configura competição
        network.lateral_competition_enabled = config.competition_enabled;
        network.competition_strength = config.competition_strength;

        // Aplica parâmetros aos neurônios
        for neuron in &mut network.neurons {
            neuron.target_firing_rate = config.target_firing_rate;
            neuron.homeo_eta = config.homeo_eta;
            neuron.homeo_interval = config.homeo_interval;

            neuron.dendritoma.set_learning_rate(config.learning_rate);
            neuron.dendritoma.set_stdp_params(
                config.stdp_a_plus,
                config.stdp_a_minus,
                config.stdp_tau_plus,
                config.stdp_tau_minus,
            );

            neuron.dendritoma.trace_tau = config.eligibility_tau;
            neuron.dendritoma.trace_increment = config.eligibility_increment;

            neuron.glia.energy_cost_fire = config.energy_cost_fire;
            neuron.glia.energy_recovery_rate = config.energy_recovery_rate;
        }

        // Cria working memory
        let working_memory = if config.use_working_memory {
            let mut wm = WorkingMemoryPool::new(config.wm_capacity, input_size);
            Some(wm)
        } else {
            None
        };

        // Cria módulo de curiosidade
        let curiosity = if config.use_curiosity {
            Some(CuriosityModule::new(input_size, output_size))
        } else {
            None
        };

        Self {
            network,
            working_memory,
            curiosity,
            config: config.clone(),
            current_epsilon: config.epsilon,
            sensor_indices,
            actuator_indices,
            seed,
            stats: AgentStats::default(),
            last_observation: vec![0.0; input_size],
            last_action: 0,
        }
    }

    fn next_random(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (u32::MAX as f64)
    }

    /// Prepara input para a rede (com contexto da WM se disponível)
    fn prepare_network_input(&mut self, observation: &[f64]) -> Vec<f64> {
        let mut input = vec![0.0; self.network.num_neurons()];

        // Copia observação para neurônios sensores
        for (i, &val) in observation.iter().enumerate() {
            if i < self.sensor_indices.len() {
                input[self.sensor_indices[i]] = val;
            }
        }

        // Modula com contexto da working memory
        if let Some(ref wm) = self.working_memory {
            if wm.active_count() > 0 {
                // Obtém média do contexto
                let context_modulation = 0.2;
                // A WM modula levemente os inputs sensoriais
                for i in &self.sensor_indices {
                    if *i < input.len() {
                        // Adiciona pequeno boost baseado em padrões armazenados
                        input[*i] *= 1.0 + context_modulation * wm.active_count() as f64 / self.config.wm_capacity as f64;
                    }
                }
            }
        }

        input
    }

    /// Seleciona ação baseada na atividade dos atuadores
    pub fn select_action(&mut self, observation: &[f64]) -> usize {
        self.stats.total_steps += 1;
        self.last_observation = observation.to_vec();

        // Epsilon-greedy exploration
        if self.next_random() < self.current_epsilon {
            let action = (self.next_random() * self.actuator_indices.len() as f64) as usize;
            self.last_action = action;
            return action;
        }

        // Prepara input e atualiza rede
        let input = self.prepare_network_input(observation);
        self.network.update(&input);

        // Coleta atividade dos atuadores
        let mut actuator_activities: Vec<f64> = self.actuator_indices.iter()
            .map(|&i| {
                let neuron = &self.network.neurons[i];
                // Combina firing state com output_signal para decisão
                let activity = if neuron.is_firing { 1.0 } else { 0.0 };
                activity + neuron.output_signal * 0.5
            })
            .collect();

        // Softmax para seleção probabilística
        let max_val = actuator_activities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let temperature = 1.0;
        let exp_sum: f64 = actuator_activities.iter()
            .map(|&x| ((x - max_val) / temperature).exp())
            .sum();

        let probs: Vec<f64> = actuator_activities.iter()
            .map(|&x| ((x - max_val) / temperature).exp() / exp_sum)
            .collect();

        // Amostra ação
        let mut cumsum = 0.0;
        let r = self.next_random();
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                self.last_action = i;
                return i;
            }
        }

        // Atualiza estatísticas
        self.stats.total_spikes += self.network.num_firing();

        self.last_action = actuator_activities.len() - 1;
        self.last_action
    }

    /// Aplica aprendizado baseado em reward
    pub fn learn(&mut self, observation: &[f64], action: usize, reward: f64, next_observation: &[f64]) {
        // Calcula curiosidade intrínseca
        let intrinsic_reward = if let Some(ref mut curiosity) = self.curiosity {
            let action_vec: Vec<f64> = (0..self.actuator_indices.len())
                .map(|i| if i == action { 1.0 } else { 0.0 })
                .collect();
            curiosity.compute_intrinsic_reward(observation, &action_vec, next_observation)
        } else {
            0.0
        };

        let total_reward = reward + intrinsic_reward * self.config.curiosity_scale;

        // Propaga reward pela rede (usa eligibility traces internamente)
        self.network.propagate_reward(total_reward);

        // Atualiza working memory
        if let Some(ref mut wm) = self.working_memory {
            // Armazena padrões significativos (com reward alto)
            if total_reward.abs() > 0.3 {
                wm.encode(observation.to_vec(), self.network.current_time_step);
            }
            wm.sustain();
        }

        // Decai epsilon
        self.current_epsilon = (self.current_epsilon * self.config.epsilon_decay)
            .max(self.config.epsilon_min);

        // Atualiza estatísticas
        self.stats.avg_reward = self.stats.avg_reward * 0.99 + total_reward * 0.01;
        self.stats.firing_rate = self.network.num_firing() as f64 / self.network.num_neurons() as f64;
        self.stats.avg_energy = self.network.average_energy();
    }

    /// Reseta o agente para novo episódio
    pub fn reset_episode(&mut self) {
        // Limpa working memory
        if let Some(ref mut wm) = self.working_memory {
            wm.clear();
        }

        // Não reseta a rede completamente - mantém aprendizado
        // Apenas reseta estados transitórios
        self.last_observation.fill(0.0);
        self.last_action = 0;
    }

    /// Retorna estatísticas do agente
    pub fn get_stats(&self) -> AgentStats {
        AgentStats {
            firing_rate: self.stats.firing_rate,
            avg_energy: self.stats.avg_energy,
            epsilon: self.current_epsilon,
            threshold: self.network.neurons.get(0).map(|n| n.threshold).unwrap_or(0.15),
            total_spikes: self.stats.total_spikes,
            total_steps: self.stats.total_steps,
            avg_reward: self.stats.avg_reward,
            wm_active_slots: self.working_memory.as_ref().map(|wm| wm.active_count()).unwrap_or(0),
        }
    }

    /// Retorna estatísticas da rede
    pub fn get_network_stats(&self) -> NetworkAgentStats {
        let net_stats = self.network.get_stats();
        NetworkAgentStats {
            time_step: net_stats.time_step,
            firing_rate: net_stats.firing_rate,
            avg_energy: net_stats.avg_energy,
            avg_threshold: net_stats.avg_threshold,
            avg_novelty: net_stats.avg_novelty,
            alert_level: net_stats.alert_level,
            avg_eligibility: net_stats.avg_eligibility,
            dopamine: net_stats.dopamine,
            norepinephrine: net_stats.norepinephrine,
        }
    }
}

/// Estatísticas do agente
#[derive(Debug, Clone, Default)]
pub struct AgentStats {
    pub firing_rate: f64,
    pub avg_energy: f64,
    pub epsilon: f64,
    pub threshold: f64,
    pub total_spikes: usize,
    pub total_steps: usize,
    pub avg_reward: f64,
    pub wm_active_slots: usize,
}

/// Estatísticas detalhadas da rede
#[derive(Debug, Clone, Default)]
pub struct NetworkAgentStats {
    pub time_step: i64,
    pub firing_rate: f64,
    pub avg_energy: f64,
    pub avg_threshold: f64,
    pub avg_novelty: f64,
    pub alert_level: f64,
    pub avg_eligibility: f64,
    pub dopamine: f64,
    pub norepinephrine: f64,
}

// =============================================================================
// MÉTRICAS DE AVALIAÇÃO
// =============================================================================

/// Métricas detalhadas de avaliação
#[derive(Debug, Clone, Default)]
pub struct EvaluationMetrics {
    /// Score principal combinado
    pub primary_score: f64,
    /// Reward total médio por episódio
    pub avg_reward: f64,
    /// Desvio padrão do reward (estabilidade)
    pub reward_std: f64,
    /// Taxa de sucesso (episódios acima do threshold)
    pub success_rate: f64,
    /// Velocidade de melhoria (slope do reward ao longo dos episódios)
    pub learning_speed: f64,
    /// Taxa de disparo média
    pub avg_firing_rate: f64,
    /// Eficiência energética (reward / energy_used)
    pub energy_efficiency: f64,
    /// Tempo médio de execução por episódio (ms)
    pub avg_episode_time_ms: f64,
    /// Número total de episódios
    pub total_episodes: usize,
    /// Número de episódios bem-sucedidos
    pub successful_episodes: usize,
    /// Métricas por ambiente
    pub per_environment: HashMap<String, EnvironmentMetrics>,
    /// Métricas da rede neural
    pub network_metrics: NetworkAgentStats,
}

/// Métricas específicas de um ambiente
#[derive(Debug, Clone, Default)]
pub struct EnvironmentMetrics {
    pub avg_reward: f64,
    pub reward_std: f64,
    pub success_rate: f64,
    pub episodes: usize,
    pub best_reward: f64,
    pub worst_reward: f64,
    pub avg_firing_rate: f64,
    pub avg_energy: f64,
}

impl EvaluationMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calcula score combinado baseado nos pesos
    pub fn compute_combined_score(&mut self, weights: &MetricWeights) {
        // Normaliza métricas para [0, 1]
        let norm_reward = (self.avg_reward + 10.0).max(0.0) / 20.0; // Assume range [-10, 10]
        let norm_success = self.success_rate;
        let norm_stability = 1.0 - (self.reward_std / (self.avg_reward.abs() + 1.0)).min(1.0);
        let norm_learning = (self.learning_speed + 0.1).max(0.0) / 0.2;
        let norm_efficiency = self.energy_efficiency.min(1.0);

        self.primary_score = weights.reward * norm_reward
            + weights.success * norm_success
            + weights.stability * norm_stability
            + weights.learning * norm_learning
            + weights.efficiency * norm_efficiency;

        self.primary_score = self.primary_score.clamp(0.0, 1.0);
    }
}

/// Pesos para combinar métricas no score final
#[derive(Debug, Clone)]
pub struct MetricWeights {
    /// Peso do reward médio
    pub reward: f64,
    /// Peso da taxa de sucesso
    pub success: f64,
    /// Peso da estabilidade
    pub stability: f64,
    /// Peso da velocidade de aprendizado
    pub learning: f64,
    /// Peso da eficiência energética
    pub efficiency: f64,
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self {
            reward: 0.35,
            success: 0.30,
            stability: 0.15,
            learning: 0.10,
            efficiency: 0.10,
        }
    }
}

impl MetricWeights {
    /// Pesos focados em performance
    pub fn performance_focused() -> Self {
        Self {
            reward: 0.45,
            success: 0.35,
            stability: 0.10,
            learning: 0.05,
            efficiency: 0.05,
        }
    }

    /// Pesos focados em estabilidade
    pub fn stability_focused() -> Self {
        Self {
            reward: 0.25,
            success: 0.25,
            stability: 0.30,
            learning: 0.10,
            efficiency: 0.10,
        }
    }

    /// Pesos focados em eficiência
    pub fn efficiency_focused() -> Self {
        Self {
            reward: 0.25,
            success: 0.20,
            stability: 0.15,
            learning: 0.10,
            efficiency: 0.30,
        }
    }

    /// Valida e normaliza pesos
    pub fn normalize(&mut self) {
        let total = self.reward + self.success + self.stability + self.learning + self.efficiency;
        if total > 0.0 {
            self.reward /= total;
            self.success /= total;
            self.stability /= total;
            self.learning /= total;
            self.efficiency /= total;
        }
    }
}

// =============================================================================
// BENCHMARK RESULT
// =============================================================================

/// Resultado de um benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Nome do benchmark/ambiente
    pub name: String,
    /// Métricas coletadas
    pub metrics: EnvironmentMetrics,
    /// Tempo total de execução
    pub duration_ms: u64,
    /// Número de episódios executados
    pub episodes: usize,
    /// Histórico de rewards por episódio
    pub reward_history: Vec<f64>,
    /// Score deste benchmark (ponderado)
    pub score: f64,
    /// Logs detalhados por episódio (se verbose)
    pub episode_logs: Vec<EpisodeLog>,
}

/// Log detalhado de um episódio
#[derive(Debug, Clone)]
pub struct EpisodeLog {
    pub episode: usize,
    pub total_reward: f64,
    pub steps: usize,
    pub success: bool,
    pub final_firing_rate: f64,
    pub final_energy: f64,
}

// =============================================================================
// BENCHMARK TRAIT
// =============================================================================

/// Trait para benchmarks de avaliação
pub trait Benchmark: Send {
    /// Nome do benchmark
    fn name(&self) -> &str;

    /// Executa o benchmark com um agente
    fn run(&self, agent: &mut NENVAgent, seed: u64, verbose: bool) -> BenchmarkResult;

    /// Peso deste benchmark no score final
    fn weight(&self) -> f64;

    /// Número de episódios a executar
    fn episodes(&self) -> usize;

    /// Descrição do benchmark
    fn description(&self) -> &str {
        "No description"
    }
}

// =============================================================================
// ENVIRONMENT BENCHMARK - Benchmark genérico para ambientes
// =============================================================================

/// Benchmark genérico que executa um ambiente
pub struct EnvironmentBenchmark {
    /// Nome do ambiente
    env_name: String,
    /// Peso do benchmark
    benchmark_weight: f64,
    /// Número de episódios
    num_episodes: usize,
    /// Descrição
    bench_description: String,
}

impl EnvironmentBenchmark {
    pub fn new(env_name: &str, weight: f64, episodes: usize) -> Self {
        Self {
            env_name: env_name.to_string(),
            benchmark_weight: weight,
            num_episodes: episodes,
            bench_description: format!("Benchmark for {}", env_name),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.bench_description = desc.to_string();
        self
    }

    fn create_environment(&self, seed: u64) -> Box<dyn Environment> {
        use crate::environments::{NavigationConfig, PatternMemoryConfig, PredictionConfig, AssociationConfig};

        match self.env_name.as_str() {
            "NavigationEnv" => Box::new(NavigationEnv::with_config(NavigationConfig::default(), seed)),
            "PatternMemoryEnv" => Box::new(PatternMemoryEnv::with_config(PatternMemoryConfig::default(), seed)),
            "PredictionEnv" => Box::new(PredictionEnv::with_config(PredictionConfig::default(), seed)),
            "AssociationEnv" => Box::new(AssociationEnv::with_config(AssociationConfig::default(), seed)),
            _ => Box::new(NavigationEnv::new(10, 10, seed)),
        }
    }
}

impl Benchmark for EnvironmentBenchmark {
    fn name(&self) -> &str {
        &self.env_name
    }

    fn run(&self, agent: &mut NENVAgent, seed: u64, verbose: bool) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut env = self.create_environment(seed);

        let mut reward_history = Vec::with_capacity(self.num_episodes);
        let mut episode_logs = Vec::new();
        let mut successful_episodes = 0;
        let success_threshold = env.success_threshold();

        let mut total_firing_rate = 0.0;
        let mut total_energy = 0.0;

        for episode in 0..self.num_episodes {
            let episode_seed = seed.wrapping_add(episode as u64);
            agent.reset_episode();

            let mut obs = env.reset();
            let mut episode_reward = 0.0;
            let mut prev_obs = obs.clone();
            let mut steps = 0;

            loop {
                let action = agent.select_action(&obs);
                let result = env.step(action);

                agent.learn(&prev_obs, action, result.reward, &result.observation);

                episode_reward += result.reward;
                prev_obs = obs;
                obs = result.observation;
                steps += 1;

                if result.done {
                    break;
                }
            }

            reward_history.push(episode_reward);

            let stats = agent.get_stats();
            total_firing_rate += stats.firing_rate;
            total_energy += stats.avg_energy;

            let is_success = episode_reward >= success_threshold;
            if is_success {
                successful_episodes += 1;
            }

            if verbose {
                episode_logs.push(EpisodeLog {
                    episode,
                    total_reward: episode_reward,
                    steps,
                    success: is_success,
                    final_firing_rate: stats.firing_rate,
                    final_energy: stats.avg_energy,
                });
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        // Calcula métricas
        let avg_reward = reward_history.iter().sum::<f64>() / reward_history.len() as f64;

        let reward_std = if reward_history.len() > 1 {
            let variance = reward_history.iter()
                .map(|r| (r - avg_reward).powi(2))
                .sum::<f64>() / (reward_history.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let success_rate = successful_episodes as f64 / self.num_episodes as f64;

        let best_reward = reward_history.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst_reward = reward_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let avg_firing_rate = total_firing_rate / self.num_episodes as f64;
        let avg_energy = total_energy / self.num_episodes as f64;

        // Score do benchmark
        let norm_reward = (avg_reward - worst_reward) / (best_reward - worst_reward + 0.01);
        let norm_stability = 1.0 - (reward_std / (avg_reward.abs() + 1.0)).min(1.0);
        let score = 0.6 * success_rate + 0.25 * norm_reward.max(0.0) + 0.15 * norm_stability;

        BenchmarkResult {
            name: self.env_name.clone(),
            metrics: EnvironmentMetrics {
                avg_reward,
                reward_std,
                success_rate,
                episodes: self.num_episodes,
                best_reward,
                worst_reward,
                avg_firing_rate,
                avg_energy,
            },
            duration_ms,
            episodes: self.num_episodes,
            reward_history,
            score,
            episode_logs,
        }
    }

    fn weight(&self) -> f64 {
        self.benchmark_weight
    }

    fn episodes(&self) -> usize {
        self.num_episodes
    }

    fn description(&self) -> &str {
        &self.bench_description
    }
}

// =============================================================================
// BENCHMARK SUITE
// =============================================================================

/// Suite de benchmarks para avaliação completa
pub struct BenchmarkSuite {
    benchmarks: Vec<Box<dyn Benchmark>>,
    weights: MetricWeights,
    config: EvaluationConfig,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            weights: MetricWeights::default(),
            config: EvaluationConfig::default(),
        }
    }

    /// Adiciona um benchmark à suite
    pub fn add_benchmark(&mut self, benchmark: Box<dyn Benchmark>) {
        self.benchmarks.push(benchmark);
    }

    /// Define pesos das métricas
    pub fn with_weights(mut self, weights: MetricWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Define configuração
    pub fn with_config(mut self, config: EvaluationConfig) -> Self {
        self.config = config;
        self
    }

    /// Suite padrão com todos os ambientes
    pub fn default_suite() -> Self {
        let mut suite = Self::new();

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("NavigationEnv", 0.35, 15)
                .with_description("Spatial navigation with reward seeking and danger avoidance")
        ));

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("PatternMemoryEnv", 0.25, 20)
                .with_description("Sequential pattern memorization testing working memory")
        ));

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("PredictionEnv", 0.25, 12)
                .with_description("Time series prediction testing predictive coding")
        ));

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("AssociationEnv", 0.15, 18)
                .with_description("Stimulus-response association learning")
        ));

        suite
    }

    /// Suite rápida para testes
    pub fn quick_suite() -> Self {
        let mut suite = Self::new().with_config(EvaluationConfig::quick());

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("NavigationEnv", 0.5, 5)
        ));

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("AssociationEnv", 0.5, 5)
        ));

        suite
    }

    /// Suite focada em navegação
    pub fn navigation_focused() -> Self {
        let mut suite = Self::new();

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("NavigationEnv", 0.70, 25)
        ));

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("AssociationEnv", 0.30, 15)
        ));

        suite
    }

    /// Suite focada em memória
    pub fn memory_focused() -> Self {
        let mut suite = Self::new();

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("PatternMemoryEnv", 0.50, 30)
        ));

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("PredictionEnv", 0.30, 20)
        ));

        suite.add_benchmark(Box::new(
            EnvironmentBenchmark::new("AssociationEnv", 0.20, 15)
        ));

        suite
    }

    /// Executa todos os benchmarks
    pub fn run_all(&self, params: &HashMap<String, ParameterValue>, seed: u64) -> EvaluationMetrics {
        let start_time = Instant::now();

        // Determina tamanhos de input/output baseado no maior ambiente
        let input_size = 28; // NavigationEnv tem o maior
        let output_size = 10; // PredictionEnv tem o maior

        let agent_config = AgentConfig::from_params(params);
        let mut agent = NENVAgent::new(input_size, output_size, agent_config, seed);

        let mut all_rewards: Vec<f64> = Vec::new();
        let mut total_successful = 0;
        let mut total_episodes = 0;
        let mut per_environment = HashMap::new();
        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;
        let mut total_firing_rate = 0.0;
        let mut total_energy = 0.0;

        for benchmark in &self.benchmarks {
            let bench_seed = seed.wrapping_mul(benchmark.name().len() as u64 + 1);
            let result = benchmark.run(&mut agent, bench_seed, self.config.verbose_logging);

            all_rewards.extend(result.reward_history.iter());
            total_successful += (result.metrics.success_rate * result.episodes as f64) as usize;
            total_episodes += result.episodes;

            total_firing_rate += result.metrics.avg_firing_rate * result.episodes as f64;
            total_energy += result.metrics.avg_energy * result.episodes as f64;

            per_environment.insert(result.name.clone(), result.metrics.clone());

            weighted_score += result.score * benchmark.weight();
            total_weight += benchmark.weight();
        }

        // Calcula métricas agregadas
        let avg_reward = if !all_rewards.is_empty() {
            all_rewards.iter().sum::<f64>() / all_rewards.len() as f64
        } else {
            0.0
        };

        let reward_std = if all_rewards.len() > 1 {
            let variance = all_rewards.iter()
                .map(|r| (r - avg_reward).powi(2))
                .sum::<f64>() / (all_rewards.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Calcula learning speed (slope dos rewards)
        let learning_speed = if all_rewards.len() >= 10 {
            let n = all_rewards.len();
            let first_half: f64 = all_rewards[..n/2].iter().sum::<f64>() / (n/2) as f64;
            let second_half: f64 = all_rewards[n/2..].iter().sum::<f64>() / (n - n/2) as f64;
            (second_half - first_half) / first_half.abs().max(0.01)
        } else {
            0.0
        };

        let avg_firing_rate = if total_episodes > 0 {
            total_firing_rate / total_episodes as f64
        } else {
            0.0
        };

        let avg_energy_used = if total_episodes > 0 {
            100.0 - (total_energy / total_episodes as f64)
        } else {
            0.0
        };

        let energy_efficiency = if avg_energy_used > 0.0 {
            avg_reward.abs() / (avg_energy_used * 10.0 + 1.0)
        } else {
            0.0
        };

        let total_time = start_time.elapsed().as_millis() as f64;
        let avg_episode_time_ms = if total_episodes > 0 {
            total_time / total_episodes as f64
        } else {
            0.0
        };

        let network_metrics = agent.get_network_stats();

        let mut metrics = EvaluationMetrics {
            primary_score: 0.0,
            avg_reward,
            reward_std,
            success_rate: if total_episodes > 0 {
                total_successful as f64 / total_episodes as f64
            } else {
                0.0
            },
            learning_speed,
            avg_firing_rate,
            energy_efficiency,
            avg_episode_time_ms,
            total_episodes,
            successful_episodes: total_successful,
            per_environment,
            network_metrics,
        };

        // Score combinado
        if total_weight > 0.0 {
            metrics.primary_score = weighted_score / total_weight;
        }

        // Ajusta com pesos de métricas
        let mut adjusted = metrics.clone();
        adjusted.compute_combined_score(&self.weights);
        metrics.primary_score = (metrics.primary_score + adjusted.primary_score) / 2.0;

        metrics
    }

    /// Número de benchmarks
    pub fn len(&self) -> usize {
        self.benchmarks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.benchmarks.is_empty()
    }

    /// Lista benchmarks
    pub fn list_benchmarks(&self) -> Vec<(&str, f64, &str)> {
        self.benchmarks.iter()
            .map(|b| (b.name(), b.weight(), b.description()))
            .collect()
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::default_suite()
    }
}

// =============================================================================
// EVALUATOR - Interface principal
// =============================================================================

/// Avaliador principal para uso no hyperopt
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

    pub fn quick_evaluator(seed: u64) -> Self {
        Self::new(BenchmarkSuite::quick_suite(), seed)
    }

    /// Avalia uma configuração
    pub fn evaluate(&self, config: &HashMap<String, ParameterValue>, trial: usize) -> EvaluationMetrics {
        let seed = self.base_seed.wrapping_add(trial as u64 * 12345);
        self.suite.run_all(config, seed)
    }

    /// Retorna lista de benchmarks
    pub fn list_benchmarks(&self) -> Vec<(&str, f64, &str)> {
        self.suite.list_benchmarks()
    }
}

// =============================================================================
// TESTES
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nenv_agent_creation() {
        let config = AgentConfig::default();
        let agent = NENVAgent::new(10, 4, config, 42);

        assert_eq!(agent.sensor_indices.len(), 10);
        assert_eq!(agent.actuator_indices.len(), 4);
    }

    #[test]
    fn test_nenv_agent_action() {
        let config = AgentConfig::default();
        let mut agent = NENVAgent::new(10, 4, config, 42);

        let obs = vec![0.5; 10];
        let action = agent.select_action(&obs);
        assert!(action < 4);
    }

    #[test]
    fn test_nenv_agent_learning() {
        let config = AgentConfig::default();
        let mut agent = NENVAgent::new(10, 4, config, 42);

        let obs = vec![0.5; 10];
        let action = agent.select_action(&obs);
        agent.learn(&obs, action, 1.0, &obs);

        let stats = agent.get_stats();
        assert!(stats.total_steps >= 1);
    }

    #[test]
    fn test_environment_benchmark() {
        let benchmark = EnvironmentBenchmark::new("NavigationEnv", 1.0, 3);
        let config = AgentConfig::default();
        let mut agent = NENVAgent::new(28, 4, config, 42);

        let result = benchmark.run(&mut agent, 42, false);
        assert_eq!(result.episodes, 3);
        assert_eq!(result.reward_history.len(), 3);
    }

    #[test]
    fn test_benchmark_suite() {
        let suite = BenchmarkSuite::quick_suite();
        assert!(!suite.is_empty());

        let config = HashMap::new();
        let metrics = suite.run_all(&config, 42);

        assert!(metrics.total_episodes > 0);
        assert!(metrics.primary_score >= 0.0 && metrics.primary_score <= 1.0);
    }

    #[test]
    fn test_evaluator() {
        let evaluator = Evaluator::quick_evaluator(42);
        let config = HashMap::new();

        let metrics = evaluator.evaluate(&config, 0);
        assert!(metrics.primary_score.is_finite());
        assert!(metrics.total_episodes > 0);
    }

    #[test]
    fn test_metric_weights() {
        let mut weights = MetricWeights::default();
        weights.normalize();

        let total = weights.reward + weights.success + weights.stability + weights.learning + weights.efficiency;
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_agent_config_from_params() {
        let mut params = HashMap::new();
        params.insert("learning.base_learning_rate".to_string(), ParameterValue::Float(0.05));
        params.insert("working_memory.capacity".to_string(), ParameterValue::Float(5.0));

        let config = AgentConfig::from_params(&params);
        assert!((config.learning_rate - 0.05).abs() < 0.001);
        assert_eq!(config.wm_capacity, 5);
    }

    #[test]
    fn test_per_environment_metrics() {
        let suite = BenchmarkSuite::quick_suite();
        let config = HashMap::new();

        let metrics = suite.run_all(&config, 42);

        assert!(metrics.per_environment.contains_key("NavigationEnv"));
    }

    #[test]
    fn test_network_metrics() {
        let config = AgentConfig::default();
        let mut agent = NENVAgent::new(10, 4, config, 42);

        let obs = vec![0.5; 10];
        for _ in 0..10 {
            let action = agent.select_action(&obs);
            agent.learn(&obs, action, 0.1, &obs);
        }

        let net_stats = agent.get_network_stats();
        assert!(net_stats.time_step > 0);
    }
}
