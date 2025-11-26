//! AutoConfig v2.0: Configuração Autônoma da Rede Neural Biológica
//!
//! Filosofia: "Minimal Specification, Maximal Autonomy"
//!
//! O usuário especifica apenas:
//! - Quantos sensores (inputs)
//! - Quantos atuadores (outputs)
//! - Tipo de tarefa (RL, classificação, etc.)
//!
//! A rede deriva automaticamente:
//! - Arquitetura (hidden neurons, topologia, E/I ratio, threshold)
//! - Todos os 60+ parâmetros (metabólicos, plasticidade, homeostase, memória)
//! - Ajustes adaptativos durante execução

mod architecture;
mod params;
mod validation;
pub mod adaptive;

pub use adaptive::*;

use crate::network::{ConnectivityType, Network};

// ============================================================================
// STRUCTS PRINCIPAIS
// ============================================================================

/// Especificação mínima da tarefa (INPUT do usuário)
#[derive(Debug, Clone)]
pub struct TaskSpec {
    /// Número de canais de entrada (sensores)
    pub num_sensors: usize,

    /// Número de canais de saída (atuadores)
    pub num_actuators: usize,

    /// Tipo de tarefa e características
    pub task_type: TaskType,
}

/// Tipo de tarefa que a rede vai executar
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Aprendizado por reforço (navegação, controle motor)
    ReinforcementLearning {
        /// Densidade de recompensas esperada
        reward_density: RewardDensity,

        /// Horizonte temporal (steps até recompensa típica)
        temporal_horizon: Option<usize>,
    },

    /// Classificação supervisionada (futuro)
    SupervisedClassification {
        num_classes: usize,
    },

    /// Memória associativa (futuro)
    AssociativeMemory {
        pattern_capacity: usize,
    },
}

/// Densidade de recompensas no ambiente
#[derive(Debug, Clone)]
pub enum RewardDensity {
    /// Rede mede automaticamente durante os primeiros N steps
    Auto,

    /// Recompensas densas (>10% dos steps têm reward)
    Dense,

    /// Recompensas moderadas (1-10% dos steps)
    Moderate,

    /// Recompensas esparsas (<1% dos steps)
    Sparse,
}

// ============================================================================
// AUTO-CONFIG (OUTPUT derivado)
// ============================================================================

/// Configuração completa derivada automaticamente
#[derive(Debug, Clone)]
pub struct AutoConfig {
    /// Especificação original da tarefa
    pub task_spec: TaskSpec,

    /// Arquitetura derivada
    pub architecture: DerivedArchitecture,

    /// Todos os parâmetros da rede (60+ valores)
    pub params: NetworkParams,

    /// Estado adaptativo (ajustado durante execução)
    pub runtime_metrics: RuntimeMetrics,
}

/// Arquitetura derivada automaticamente
#[derive(Debug, Clone)]
pub struct DerivedArchitecture {
    /// Total de neurônios (sensores + hidden + atuadores)
    pub total_neurons: usize,

    /// Índices dos neurônios sensoriais [0, num_sensors)
    pub sensor_indices: std::ops::Range<usize>,

    /// Índices dos neurônios hidden [num_sensors, num_sensors+num_hidden)
    pub hidden_indices: std::ops::Range<usize>,

    /// Índices dos neurônios atuadores (motores)
    pub actuator_indices: std::ops::Range<usize>,

    /// Topologia de conectividade
    pub connectivity: ConnectivityType,

    /// Razão de neurônios inibitórios (0.0-1.0)
    pub inhibitory_ratio: f64,

    /// Threshold de disparo inicial
    pub initial_threshold: f64,
}

/// Todos os parâmetros da rede
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
    pub istdp: iSTDPParams,
    pub plasticity: PlasticityParams,

    // Homeostase
    pub homeostatic: HomeostaticParams,

    // Input (derivado)
    pub input: DerivedInputParams,

    // Memória
    pub memory: MemoryParams,

    // Novidade/Alerta
    pub novelty: NoveltyParams,

    // Sono/Consolidação
    pub sleep: SleepParams,

    // RL-específico (se aplicável)
    pub rl: Option<RLParams>,
}

/// Parâmetros de energia e metabolismo
#[derive(Debug, Clone)]
pub struct EnergyParams {
    pub max_energy: f64,
    pub energy_cost_fire: f64,
    pub energy_cost_maintenance: f64,
    pub energy_recovery_rate: f64,
    pub plasticity_energy_cost_factor: f64,
}

/// Parâmetros de STDP (excitatório)
#[derive(Debug, Clone)]
pub struct STDPParams {
    pub window: i64,
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub a_plus: f64,
    pub a_minus: f64,
}

/// Parâmetros de iSTDP (inibitório)
#[derive(Debug, Clone)]
pub struct iSTDPParams {
    pub learning_rate: f64,
    pub target_rate: f64,
}

/// Parâmetros de plasticidade geral
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
    pub fr_alpha: f64,  // Alpha da EMA de firing rate
}

/// Parâmetros de entrada (input strength) derivados
#[derive(Debug, Clone)]
pub struct DerivedInputParams {
    /// Densidade recomendada de input: % de neurônios estimulados por step
    pub recommended_input_density: f64,

    /// Amplitude recomendada do estímulo de entrada
    pub recommended_input_amplitude: f64,
}

/// Parâmetros de memória (STM/LTM)
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

/// Parâmetros de proteção de LTM
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

/// Parâmetros de novidade e alerta
#[derive(Debug, Clone)]
pub struct NoveltyParams {
    pub alert_decay_rate: f64,
    pub novelty_alert_threshold: f64,
    pub alert_sensitivity: f64,
    pub sleep_alert_level: f64,
    pub initial_priority: f64,
}

/// Parâmetros de sono e consolidação
#[derive(Debug, Clone)]
pub struct SleepParams {
    pub sleep_interval: u64,
    pub sleep_duration: usize,
    pub sleep_replay_noise: f64,
    pub min_selectivity_to_sleep: f64,
    pub sleep_learning_rate_factor: f64,
    pub sleep_metabolic_factor: f64,
}

/// Parâmetros específicos de Reinforcement Learning
#[derive(Debug, Clone)]
pub struct RLParams {
    pub initial_exploration_rate: f64,
    pub exploration_decay_rate: f64,
    pub eligibility_trace_window: i64,
    pub spike_classification: SpikeClassificationParams,
}

/// Parâmetros de classificação de spikes (SpikeOrigin)
#[derive(Debug, Clone)]
pub struct SpikeClassificationParams {
    pub feedback_excess_factor: f64,
}

/// Métricas runtime (atualizado durante execução)
#[derive(Debug, Clone)]
pub struct RuntimeMetrics {
    /// Densidade de reward medida (RL)
    pub measured_reward_density: f64,

    /// Horizonte temporal médio (steps até reward)
    pub measured_temporal_horizon: f64,

    /// Taxa de novidade média do ambiente
    pub measured_novelty_rate: f64,

    /// Energia média da rede
    pub measured_avg_energy: f64,

    /// Erro de firing rate (real - alvo)
    pub measured_fr_error: f64,

    /// Contador de episódios/sucessos
    pub episode_count: usize,

    /// Número de steps executados
    pub steps_executed: u64,
}

// ============================================================================
// IMPLEMENTAÇÃO PRINCIPAL
// ============================================================================

impl AutoConfig {
    /// Cria configuração completa a partir de especificação mínima da tarefa
    ///
    /// # Exemplo
    /// ```
    /// use nenv_visual_sim::autoconfig::*;
    ///
    /// let task = TaskSpec {
    ///     num_sensors: 4,
    ///     num_actuators: 4,
    ///     task_type: TaskType::ReinforcementLearning {
    ///         reward_density: RewardDensity::Auto,
    ///         temporal_horizon: None,
    ///     },
    /// };
    ///
    /// let config = AutoConfig::from_task(task);
    /// ```
    pub fn from_task(task_spec: TaskSpec) -> Self {
        // NÍVEL 0: Deriva arquitetura
        let architecture = DerivedArchitecture::from_task(&task_spec);

        // NÍVEL 1-5: Calcula todos os parâmetros
        let params = NetworkParams::from_architecture(&architecture, &task_spec);

        // Métricas runtime iniciais
        let runtime_metrics = RuntimeMetrics::default();

        AutoConfig {
            task_spec,
            architecture,
            params,
            runtime_metrics,
        }
    }

    /// Cria a rede neural configurada
    pub fn build_network(&self) -> Result<Network, String> {
        // Valida configuração
        if let Err(errors) = self.validate() {
            return Err(errors.join("\n"));
        }

        // Cria a rede usando os parâmetros derivados
        let arch = &self.architecture;
        let params = &self.params;

        let mut network = Network::new(
            arch.total_neurons,
            arch.connectivity,
            arch.inhibitory_ratio,
            arch.initial_threshold,
        );

        // Configura modo de aprendizado (sempre STDP para AutoConfig)
        network.set_learning_mode(crate::network::LearningMode::STDP);

        // Aplica todos os parâmetros derivados aos neurônios
        for neuron in &mut network.neurons {
            // Parâmetros homeostáticos
            neuron.target_firing_rate = params.target_firing_rate;
            neuron.homeo_eta = params.homeostatic.homeo_eta;
            neuron.homeo_interval = params.homeostatic.homeo_interval;
            neuron.set_refractory_period(params.homeostatic.refractory_period);
            neuron.set_memory_alpha(params.homeostatic.memory_alpha);
            neuron.meta_threshold = params.homeostatic.meta_threshold;
            neuron.meta_alpha = params.homeostatic.meta_alpha;
            // Ajusta proporção peso/threshold com base no grid atual (W65T35)
            neuron.homeo_weight_ratio = 0.650;
            neuron.homeo_threshold_ratio = 0.350;

            // Parâmetros do dendritoma
            neuron.dendritoma.set_learning_rate(params.learning_rate);
            neuron.dendritoma.set_stdp_params(
                params.stdp.a_plus,
                params.stdp.a_minus,
                params.stdp.tau_plus,
                params.stdp.tau_minus,
            );
            neuron.dendritoma.istdp_learning_rate = params.istdp.learning_rate;
            neuron.dendritoma.istdp_target_rate = params.istdp.target_rate;
            neuron.dendritoma.set_weight_decay(params.memory.weight_decay);
            neuron.dendritoma.weight_clamp = params.memory.weight_clamp;
            neuron.dendritoma.tag_decay_rate = params.memory.tag_decay_rate;
            neuron.dendritoma.capture_threshold = params.memory.capture_threshold;
            neuron.dendritoma.dopamine_sensitivity = params.memory.dopamine_sensitivity;

            // Parâmetros de energia (glia)
            neuron.glia.max_energy = params.energy.max_energy;
            neuron.glia.energy = params.energy.max_energy; // Inicia com energia cheia
            neuron.glia.energy_cost_fire = params.energy.energy_cost_fire;
            neuron.glia.energy_cost_maintenance = params.energy.energy_cost_maintenance;
            neuron.glia.energy_recovery_rate = params.energy.energy_recovery_rate;

            // Parâmetros de novidade/alerta
            neuron.glia.priority = params.novelty.initial_priority;
        }

        // Configura parâmetros da rede
        network.stdp_window = params.stdp.window;
        network.sleep_learning_rate_factor = params.sleep.sleep_learning_rate_factor;
        network.set_novelty_alert_params(
            params.novelty.novelty_alert_threshold,
            params.novelty.alert_sensitivity,
        );
        network.alert_decay_rate = params.novelty.alert_decay_rate;

        // Inicializa pesos baseado nos parâmetros derivados
        self.initialize_weights(&mut network);

        Ok(network)
    }

    /// Inicializa os pesos da rede baseado nos parâmetros do AutoConfig
    fn initialize_weights(&self, network: &mut Network) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let num_inhibitory = (self.architecture.total_neurons as f64
            * self.architecture.inhibitory_ratio)
            .floor() as usize;

        for (i, neuron) in network.neurons.iter_mut().enumerate() {
            // Inicializa pesos excitatórios com pequena variação aleatória
            for w in &mut neuron.dendritoma.weights {
                *w = rng.gen_range(
                    self.params.initial_excitatory_weight * 0.8
                        ..self.params.initial_excitatory_weight * 1.2,
                );
            }

            // Inicializa pesos inibitórios
            for source_id in 0..self.architecture.total_neurons {
                if source_id < neuron.dendritoma.weights.len() {
                    if network.connectivity_matrix[i][source_id] == 1 {
                        // Se fonte é inibitória e não é auto-conexão
                        if source_id < num_inhibitory && source_id != i {
                            neuron.dendritoma.weights[source_id] =
                                self.params.initial_inhibitory_weight;
                        }
                    }
                }
            }
        }
    }

    /// Imprime relatório detalhado da configuração
    pub fn print_report(&self) {
        println!("╔════════════════════════════════════════╗");
        println!("║  CONFIGURAÇÃO AUTÔNOMA NEN-V v2.0     ║");
        println!("╚════════════════════════════════════════╝\n");

        self.print_task_spec();
        self.print_architecture();
        self.print_parameters();
        self.print_verification();
    }
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

// ============================================================================
// TESTES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_task_basic() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        };

        let config = AutoConfig::from_task(task);

        // Verifica que arquitetura foi derivada
        assert!(config.architecture.total_neurons > 8);
        assert_eq!(config.architecture.sensor_indices.len(), 4);
        assert_eq!(config.architecture.actuator_indices.len(), 4);
    }

    #[test]
    fn test_adaptive_state_default() {
        let state = AdaptiveState::default();

        assert_eq!(state.measured_reward_density, 0.0);
        assert_eq!(state.measured_avg_energy, 100.0);
        assert_eq!(state.episode_count, 0);
    }
}
