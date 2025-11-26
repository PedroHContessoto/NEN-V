//! # Parâmetros da Rede Neural
//!
//! Define todas as estruturas de parâmetros derivados automaticamente.

use super::architecture::DerivedArchitecture;
use super::task::{TaskSpec, TaskType, RewardDensity};
use super::derivation;

/// Parâmetros completos da rede
#[derive(Debug, Clone)]
pub struct NetworkParams {
    // Estruturais
    pub target_firing_rate: f64,
    pub learning_rate: f64,
    pub avg_connections: usize,
    pub initial_excitatory_weight: f64,
    pub initial_inhibitory_weight: f64,

    // Metabólicos
    pub energy: EnergyParams,

    // Plasticidade
    pub stdp: STDPParams,
    pub istdp: ISTDPParams,
    pub plasticity: PlasticityParams,

    // Homeostase
    pub homeostatic: HomeostaticParams,

    // Input
    pub input: DerivedInputParams,

    // Memória
    pub memory: MemoryParams,

    // Novidade/Alerta
    pub novelty: NoveltyParams,

    // Sono
    pub sleep: SleepParams,

    // RL
    pub rl: Option<RLParams>,

    // Eligibility Traces (v2.0)
    pub eligibility: EligibilityParams,

    // STP (v2.0)
    pub stp: STPParams,

    // Competição Lateral (v2.0)
    pub competition: CompetitionParams,

    // Working Memory (v2.0)
    pub working_memory: WorkingMemoryParams,

    // Curiosidade Intrínseca (v2.0)
    pub curiosity: CuriosityParams,
}

impl NetworkParams {
    /// Deriva todos os parâmetros a partir da arquitetura
    pub fn from_architecture(arch: &DerivedArchitecture, task: &TaskSpec) -> Self {
        derivation::derive_all_params(arch, task)
    }
}

/// Parâmetros de energia
#[derive(Debug, Clone)]
pub struct EnergyParams {
    pub max_energy: f64,
    pub energy_cost_fire: f64,
    pub energy_cost_maintenance: f64,
    pub energy_recovery_rate: f64,
    pub plasticity_energy_cost_factor: f64,
}

/// Parâmetros STDP
#[derive(Debug, Clone)]
pub struct STDPParams {
    pub window: i64,
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub a_plus: f64,
    pub a_minus: f64,
}

/// Parâmetros iSTDP
#[derive(Debug, Clone)]
pub struct ISTDPParams {
    pub learning_rate: f64,
    pub target_rate: f64,
}

/// Parâmetros de plasticidade
#[derive(Debug, Clone)]
pub struct PlasticityParams {
    pub base_gain: f64,
    pub min_gain: f64,
    pub energy_threshold_for_full_plasticity: f64,
}

/// Parâmetros homeostáticos
#[derive(Debug, Clone)]
pub struct HomeostaticParams {
    pub refractory_period: i64,
    pub memory_alpha: f64,
    pub homeo_interval: i64,
    pub homeo_eta: f64,
    pub meta_threshold: f64,
    pub meta_alpha: f64,
    pub fr_alpha: f64,
}

/// Parâmetros de entrada
#[derive(Debug, Clone)]
pub struct DerivedInputParams {
    pub recommended_input_density: f64,
    pub recommended_input_amplitude: f64,
}

/// Parâmetros de memória
#[derive(Debug, Clone)]
pub struct MemoryParams {
    pub weight_decay: f64,
    pub weight_clamp: f64,
    pub tag_decay_rate: f64,
    pub tag_multiplier: f64,
    pub capture_threshold: f64,
    pub dopamine_sensitivity: f64,
    pub consolidation_base_rate: f64,
    pub ltm_protection: LTMProtectionParams,
    pub spike_history_capacity: usize,
}

/// Parâmetros de proteção LTM
#[derive(Debug, Clone)]
pub struct LTMProtectionParams {
    pub stability_threshold: f64,
    pub ltm_relevance_threshold: f64,
    pub attraction_strength: f64,
    pub small_change_threshold: f64,
    pub stability_increment: f64,
    pub stability_decay_factor: f64,
    pub tag_consumption_factor: f64,
}

/// Parâmetros de novidade
#[derive(Debug, Clone)]
pub struct NoveltyParams {
    pub alert_decay_rate: f64,
    pub novelty_alert_threshold: f64,
    pub alert_sensitivity: f64,
    pub sleep_alert_level: f64,
    pub initial_priority: f64,
}

/// Parâmetros de sono
#[derive(Debug, Clone)]
pub struct SleepParams {
    pub sleep_interval: u64,
    pub sleep_duration: usize,
    pub sleep_replay_noise: f64,
    pub min_selectivity_to_sleep: f64,
    pub sleep_learning_rate_factor: f64,
    pub sleep_metabolic_factor: f64,
}

/// Parâmetros RL
#[derive(Debug, Clone)]
pub struct RLParams {
    pub initial_exploration_rate: f64,
    pub exploration_decay_rate: f64,
    pub eligibility_trace_window: i64,
    pub spike_classification: SpikeClassificationParams,
}

/// Parâmetros de classificação de spikes
#[derive(Debug, Clone)]
pub struct SpikeClassificationParams {
    pub feedback_excess_factor: f64,
}

/// Parâmetros de Eligibility Traces (v2.0)
#[derive(Debug, Clone)]
pub struct EligibilityParams {
    /// Constante de tempo do trace
    pub trace_tau: f64,
    /// Incremento por correlação pré-pós
    pub trace_increment: f64,
    /// Habilita 3-factor learning
    pub enabled: bool,
}

/// Parâmetros de STP (v2.0)
#[derive(Debug, Clone)]
pub struct STPParams {
    /// Taxa de recuperação de recursos
    pub recovery_tau: f64,
    /// Fração usada por spike
    pub use_fraction: f64,
    /// Habilita STP
    pub enabled: bool,
}

/// Parâmetros de Competição Lateral (v2.0)
#[derive(Debug, Clone)]
pub struct CompetitionParams {
    /// Força da competição [0, 1]
    pub strength: f64,
    /// Intervalo entre aplicações
    pub interval: i64,
    /// Habilita competição
    pub enabled: bool,
}

/// Parâmetros de Working Memory (v2.0)
#[derive(Debug, Clone)]
pub struct WorkingMemoryParams {
    /// Capacidade máxima de slots
    pub capacity: usize,
    /// Força da recorrência
    pub recurrent_strength: f64,
    /// Taxa de decaimento
    pub decay_rate: f64,
    /// Força da inibição lateral
    pub lateral_inhibition: f64,
    /// Habilita WM
    pub enabled: bool,
}

/// Parâmetros de Curiosidade Intrínseca (v2.0)
#[derive(Debug, Clone)]
pub struct CuriosityParams {
    /// Escala da recompensa de curiosidade
    pub curiosity_scale: f64,
    /// Threshold mínimo de surpresa
    pub surprise_threshold: f64,
    /// Taxa de habituação
    pub habituation_rate: f64,
    /// Habilita curiosidade
    pub enabled: bool,
}

/// Métricas runtime
#[derive(Debug, Clone)]
pub struct RuntimeMetrics {
    pub measured_reward_density: f64,
    pub measured_temporal_horizon: f64,
    pub measured_novelty_rate: f64,
    pub measured_avg_energy: f64,
    pub measured_fr_error: f64,
    pub episode_count: usize,
    pub steps_executed: u64,
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self {
            measured_reward_density: 0.0,
            measured_temporal_horizon: 0.0,
            measured_novelty_rate: 0.0,
            measured_avg_energy: 100.0,
            measured_fr_error: 0.0,
            episode_count: 0,
            steps_executed: 0,
        }
    }
}
