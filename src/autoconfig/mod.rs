//! # AutoConfig v2.0: Configuração Autônoma da Rede Neural Biológica
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
//! - Todos os 80+ parâmetros (metabólicos, plasticidade, homeostase, memória)
//! - Ajustes adaptativos durante execução
//!
//! ## Estrutura do Módulo
//!
//! - `task`: Especificação da tarefa (TaskSpec, TaskType)
//! - `architecture`: Derivação de arquitetura (DerivedArchitecture)
//! - `params`: Parâmetros da rede (NetworkParams e sub-structs)
//! - `derivation`: Funções de cálculo de parâmetros
//! - `adaptive`: Sistema adaptativo runtime (AdaptiveState)
//!
//! ## Exemplo
//!
//! ```rust,ignore
//! use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};
//!
//! let task = TaskSpec {
//!     num_sensors: 8,
//!     num_actuators: 4,
//!     task_type: TaskType::ReinforcementLearning {
//!         reward_density: RewardDensity::Auto,
//!         temporal_horizon: Some(100),
//!     },
//! };
//!
//! let config = AutoConfig::from_task(task);
//! config.print_report();
//!
//! let network = config.build_network().expect("Configuração válida");
//! ```

mod task;
mod architecture;
mod params;
mod derivation;
mod adaptive;

// Re-exportações públicas
pub use task::{TaskSpec, TaskType, RewardDensity};
pub use architecture::DerivedArchitecture;
pub use params::{
    NetworkParams, EnergyParams, STDPParams, ISTDPParams, PlasticityParams,
    HomeostaticParams, DerivedInputParams, MemoryParams, LTMProtectionParams,
    NoveltyParams, SleepParams, RLParams, SpikeClassificationParams,
    EligibilityParams, STPParams, CompetitionParams, WorkingMemoryParams,
    CuriosityParams, RuntimeMetrics,
};
pub use adaptive::{AdaptiveState, AdaptiveStats, NetworkIssue, CorrectiveAction, SleepOutcome};

use crate::network::{ConnectivityType, Network, LearningMode};

/// Configuração completa derivada automaticamente
#[derive(Debug, Clone)]
pub struct AutoConfig {
    pub task_spec: TaskSpec,
    pub architecture: DerivedArchitecture,
    pub params: NetworkParams,
    pub runtime_metrics: RuntimeMetrics,
}

impl AutoConfig {
    /// Cria configuração a partir da especificação da tarefa
    pub fn from_task(task_spec: TaskSpec) -> Self {
        let architecture = DerivedArchitecture::from_task(&task_spec);
        let params = NetworkParams::from_architecture(&architecture, &task_spec);
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
        if let Err(errors) = self.validate() {
            return Err(errors.join("\n"));
        }

        let arch = &self.architecture;
        let params = &self.params;

        let mut network = Network::new(
            arch.total_neurons,
            arch.connectivity,
            arch.inhibitory_ratio,
            arch.initial_threshold,
        );

        // Configura modo de aprendizado
        network.set_learning_mode(LearningMode::STDP);

        // Configura índices de camadas
        network.set_layer_indices(
            arch.sensor_indices.clone().collect(),
            arch.hidden_indices.clone().collect(),
            arch.actuator_indices.clone().collect(),
        );

        // Configura competição lateral
        network.lateral_competition_enabled = params.competition.enabled;
        network.competition_strength = params.competition.strength;
        network.competition_interval = params.competition.interval;

        // Aplica parâmetros aos neurônios
        for neuron in &mut network.neurons {
            // Homeostase
            neuron.target_firing_rate = params.target_firing_rate;
            neuron.homeo_eta = params.homeostatic.homeo_eta;
            neuron.homeo_interval = params.homeostatic.homeo_interval;
            neuron.set_refractory_period(params.homeostatic.refractory_period);
            neuron.set_memory_alpha(params.homeostatic.memory_alpha);
            neuron.meta_threshold = params.homeostatic.meta_threshold;
            neuron.meta_alpha = params.homeostatic.meta_alpha;
            neuron.homeo_weight_ratio = 0.650;
            neuron.homeo_threshold_ratio = 0.350;

            // Dendritoma
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

            // Eligibility Traces (v2.0)
            neuron.dendritoma.trace_tau = params.eligibility.trace_tau;
            neuron.dendritoma.trace_increment = params.eligibility.trace_increment;

            // STP (v2.0)
            neuron.dendritoma.stp_recovery_tau = params.stp.recovery_tau;
            neuron.dendritoma.stp_use_fraction = params.stp.use_fraction;

            // Competição
            neuron.dendritoma.competitive_normalization_enabled = params.competition.enabled;

            // Energia
            neuron.glia.max_energy = params.energy.max_energy;
            neuron.glia.energy = params.energy.max_energy;
            neuron.glia.energy_cost_fire = params.energy.energy_cost_fire;
            neuron.glia.energy_cost_maintenance = params.energy.energy_cost_maintenance;
            neuron.glia.energy_recovery_rate = params.energy.energy_recovery_rate;

            // Novidade
            neuron.glia.priority = params.novelty.initial_priority;
        }

        // Parâmetros da rede
        network.stdp_window = params.stdp.window;
        network.sleep_learning_rate_factor = params.sleep.sleep_learning_rate_factor;
        network.set_novelty_alert_params(
            params.novelty.novelty_alert_threshold,
            params.novelty.alert_sensitivity,
        );
        network.alert_decay_rate = params.novelty.alert_decay_rate;

        // Inicializa pesos
        self.initialize_weights(&mut network);

        Ok(network)
    }

    /// Inicializa pesos da rede
    fn initialize_weights(&self, network: &mut Network) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let num_inhibitory = (self.architecture.total_neurons as f64
            * self.architecture.inhibitory_ratio)
            .floor() as usize;

        for (i, neuron) in network.neurons.iter_mut().enumerate() {
            // Pesos excitatórios
            for w in &mut neuron.dendritoma.weights {
                *w = rng.gen_range(
                    self.params.initial_excitatory_weight * 0.8
                        ..self.params.initial_excitatory_weight * 1.2,
                );
            }

            // Pesos inibitórios
            for source_id in 0..self.architecture.total_neurons {
                if source_id < neuron.dendritoma.weights.len() {
                    if network.connectivity_matrix[i][source_id] == 1 {
                        if source_id < num_inhibitory && source_id != i {
                            neuron.dendritoma.weights[source_id] =
                                self.params.initial_inhibitory_weight;
                        }
                    }
                }
            }
        }
    }

    /// Valida a configuração
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validação da arquitetura
        if self.architecture.total_neurons < 3 {
            errors.push("Rede muito pequena (mínimo 3 neurônios)".to_string());
        }

        if self.architecture.inhibitory_ratio < 0.0 || self.architecture.inhibitory_ratio > 1.0 {
            errors.push("Razão inibitória deve estar entre 0.0 e 1.0".to_string());
        }

        if self.architecture.initial_threshold <= 0.0 {
            errors.push("Threshold inicial deve ser positivo".to_string());
        }

        // Validação de energia
        let avg_cost = self.params.energy.energy_cost_fire * self.params.target_firing_rate;
        let avg_gain = self.params.energy.energy_recovery_rate * (1.0 - self.params.target_firing_rate);
        if avg_gain <= avg_cost {
            errors.push(format!(
                "Balanço energético negativo: ganho={:.3}, custo={:.3}",
                avg_gain, avg_cost
            ));
        }

        // Validação STDP
        if self.params.stdp.tau_plus <= 0.0 || self.params.stdp.tau_minus <= 0.0 {
            errors.push("Constantes de tempo STDP devem ser positivas".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Imprime relatório completo
    pub fn print_report(&self) {
        println!("╔═════════════════════════════════════════╗");
        println!("║    CONFIGURAÇÃO AUTÔNOMA NEN-V v2.0     ║");
        println!("╚═════════════════════════════════════════╝\n");

        self.print_task_spec();
        self.print_architecture();
        self.print_parameters();
        self.print_verification();
    }

    fn print_task_spec(&self) {
        println!("┌─────────────────────────────────────────┐");
        println!("│ ESPECIFICAÇÃO DA TAREFA                 │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Sensores:   {:>27} │", self.task_spec.num_sensors);
        println!("│ Atuadores:  {:>27} │", self.task_spec.num_actuators);
        println!("│ Tipo:       {:>27} │", self.task_type_str());
        println!("└─────────────────────────────────────────┘\n");
    }

    fn print_architecture(&self) {
        let arch = &self.architecture;
        println!("┌─────────────────────────────────────────┐");
        println!("│ ARQUITETURA DERIVADA                    │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Total neurônios:    {:>19} │", arch.total_neurons);
        println!("│ Camada hidden:      {:>19} │", arch.hidden_indices.len());
        println!("│ Razão inibitória:   {:>18.1}% │", arch.inhibitory_ratio * 100.0);
        println!("│ Threshold inicial:  {:>19.3} │", arch.initial_threshold);
        println!("│ Conectividade:      {:>19} │", self.connectivity_str());
        println!("└─────────────────────────────────────────┘\n");
    }

    fn print_parameters(&self) {
        let params = &self.params;
        println!("┌─────────────────────────────────────────┐");
        println!("│ PARÂMETROS PRINCIPAIS                   │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Target FR:          {:>18.2}% │", params.target_firing_rate * 100.0);
        println!("│ Learning rate:      {:>19.4} │", params.learning_rate);
        println!("│ STDP window:        {:>19} │", params.stdp.window);
        println!("│ STDP tau+/tau-:     {:>9.1}/{:<8.1} │", params.stdp.tau_plus, params.stdp.tau_minus);
        println!("│ Eligibility tau:    {:>19.1} │", params.eligibility.trace_tau);
        println!("│ STP recovery:       {:>19.1} │", params.stp.recovery_tau);
        println!("│ Competition:        {:>19.2} │", params.competition.strength);
        println!("│ WM capacity:        {:>19} │", params.working_memory.capacity);
        println!("│ Curiosity scale:    {:>19.2} │", params.curiosity.curiosity_scale);
        println!("└─────────────────────────────────────────┘\n");
    }

    fn print_verification(&self) {
        let energy = &self.params.energy;
        let avg_cost = energy.energy_cost_fire * self.params.target_firing_rate;
        let avg_gain = energy.energy_recovery_rate * (1.0 - self.params.target_firing_rate);
        let balance = avg_gain - avg_cost;

        println!("┌─────────────────────────────────────────┐");
        println!("│ VERIFICAÇÃO                             │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Balanço energético: {:>+18.3} │", balance);
        println!("│ LTP/LTD ratio:      {:>19.2} │", self.params.stdp.a_plus / self.params.stdp.a_minus);
        println!("│ Capacidade memória: {:>19} │", self.architecture.estimate_memory_capacity());

        let status = if balance > 0.0 { "✓ OK" } else { "✗ ERRO" };
        println!("│ Status:             {:>19} │", status);
        println!("└─────────────────────────────────────────┘\n");
    }

    fn task_type_str(&self) -> &str {
        match &self.task_spec.task_type {
            TaskType::ReinforcementLearning { .. } => "RL",
            TaskType::SupervisedClassification { .. } => "Classificação",
            TaskType::AssociativeMemory { .. } => "Memória",
        }
    }

    fn connectivity_str(&self) -> &str {
        match self.architecture.connectivity {
            ConnectivityType::FullyConnected => "FullyConnected",
            ConnectivityType::Grid2D => "Grid2D",
            ConnectivityType::Isolated => "Isolated",
        }
    }
}

// ============================================================================
// TESTES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn rl_task() -> TaskType {
        TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        }
    }

    #[test]
    fn test_autoconfig_creation() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: rl_task(),
        };

        let config = AutoConfig::from_task(task);

        assert!(config.architecture.total_neurons > 8);
        assert_eq!(config.architecture.sensor_indices.len(), 4);
        assert_eq!(config.architecture.actuator_indices.len(), 4);
    }

    #[test]
    fn test_validation_passes() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: rl_task(),
        };

        let config = AutoConfig::from_task(task);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_network_building() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: rl_task(),
        };

        let config = AutoConfig::from_task(task);
        let network = config.build_network();

        assert!(network.is_ok());
        let net = network.unwrap();
        assert_eq!(net.num_neurons(), config.architecture.total_neurons);
    }

    #[test]
    fn test_eligibility_params() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Sparse,
                temporal_horizon: Some(100),
            },
        };

        let config = AutoConfig::from_task(task);

        assert!(config.params.eligibility.enabled);
        assert!(config.params.eligibility.trace_tau > 100.0);
    }

    #[test]
    fn test_stdp_params() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: rl_task(),
        };

        let config = AutoConfig::from_task(task);

        // tau_plus > tau_minus (preserva causalidade)
        assert!(config.params.stdp.tau_plus > config.params.stdp.tau_minus);

        // a_plus e a_minus devem ser positivos
        assert!(config.params.stdp.a_plus > 0.0);
        assert!(config.params.stdp.a_minus > 0.0);

        // Após hyperopt: LTP/LTD ratio ~0.97 (quase simétrico)
        let ratio = config.params.stdp.a_plus / config.params.stdp.a_minus;
        assert!(ratio > 0.8 && ratio < 1.2);
    }

    #[test]
    fn test_working_memory_params() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: rl_task(),
        };

        let config = AutoConfig::from_task(task);

        assert!(config.params.working_memory.enabled);
        assert!(config.params.working_memory.capacity >= 5);
    }

    #[test]
    fn test_curiosity_params() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Sparse,
                temporal_horizon: None,
            },
        };

        let config = AutoConfig::from_task(task);

        assert!(config.params.curiosity.enabled);
        assert!(config.params.curiosity.curiosity_scale > 0.1);
    }

    #[test]
    fn test_adaptive_state() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: rl_task(),
        };

        let config = AutoConfig::from_task(task);
        let mut state = AdaptiveState::new(config);

        // Registra métricas
        for _ in 0..100 {
            state.record_metrics(0.1, 80.0, Some(0.5));
        }

        let stats = state.get_stats();
        assert!(stats.avg_firing_rate > 0.0);
        assert!(stats.avg_energy > 0.0);
    }
}
