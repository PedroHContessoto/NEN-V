//! # NEN-V: Neuromorphic Energy-based Neural Virtual Model v2.0
//!
//! Uma implementação biologicamente plausível de rede neural com mecanismos de
//! aprendizado inspirados em neurociência, incluindo STDP, homeostase, consolidação
//! de memória durante o sono, eligibility traces, competição lateral, working memory,
//! codificação preditiva e curiosidade intrínseca.
//!
//! ## Arquitetura do Sistema
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           NEN-V v2.0 ARCHITECTURE                           │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                             │
//! │  ┌─────────────────────────────────────────────────────────────────────┐   │
//! │  │                         PROCESSAMENTO                                │   │
//! │  │                                                                      │   │
//! │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
//! │  │  │  Sensorial  │  │   Hidden    │  │         Atuadores           │  │   │
//! │  │  │  (Input)    │──│   Layer     │──│         (Output)            │  │   │
//! │  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
//! │  │        │                │                        │                   │   │
//! │  │        ▼                ▼                        ▼                   │   │
//! │  │  ┌─────────────────────────────────────────────────────────────┐    │   │
//! │  │  │              WORKING MEMORY POOL (7±2 slots)                │    │   │
//! │  │  └─────────────────────────────────────────────────────────────┘    │   │
//! │  └──────────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! │  ┌──────────────────────────────────────────────────────────────────────┐  │
//! │  │                       PLASTICIDADE                                   │  │
//! │  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐    │  │
//! │  │  │    STDP      │  │ Eligibility  │  │   Predição/Modelo       │    │  │
//! │  │  │  Adaptativo  │◄─┤   Traces     │◄─┤   Interno               │    │  │
//! │  │  └──────────────┘  └──────────────┘  └─────────────────────────┘    │  │
//! │  │                           │                                          │  │
//! │  │                           ▼                                          │  │
//! │  │                 ┌─────────────────────┐                              │  │
//! │  │                 │   Neuromodulação    │                              │  │
//! │  │                 │   Diferencial       │                              │  │
//! │  │                 └─────────────────────┘                              │  │
//! │  └──────────────────────────────────────────────────────────────────────┘  │
//! │                                                                             │
//! │  ┌──────────────────────────────────────────────────────────────────────┐  │
//! │  │                        MOTIVAÇÃO                                     │  │
//! │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │  │
//! │  │  │ Curiosidade │  │ Saciedade/  │  │   Reward Extrínseco        │  │  │
//! │  │  │ Intrínseca  │──┤ Necessidade │──┤   (Ambiente)               │  │  │
//! │  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │  │
//! │  └──────────────────────────────────────────────────────────────────────┘  │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Módulos Core
//!
//! - **nenv**: Neurônio individual com dendritoma, glia e axônio
//! - **dendritoma**: Sistema dendrítico com aprendizado sináptico (STDP/Hebbiano)
//! - **glia**: Modulação metabólica e energética
//! - **network**: Orquestração de múltiplos neurônios e dinâmicas de rede
//! - **neuromodulation**: Sistema de neuromodulação (dopamina, norepinefrina, etc.)
//!
//! ## Módulos Cognitivos (v2.0)
//!
//! - **working_memory**: Pool de memória de trabalho com dinâmica de atrator
//! - **predictive**: Hierarquia preditiva e Active Inference
//! - **intrinsic_motivation**: Curiosidade intrínseca e exploração autônoma
//!
//! ## Módulos de Configuração
//!
//! - **autoconfig**: Configuração automática da rede baseada na tarefa
//!
//! ## Exemplo de Uso Básico
//!
//! ```rust,no_run
//! use nenv_v2::network::{Network, ConnectivityType, LearningMode};
//!
//! // Cria rede manualmente
//! let mut net = Network::new(
//!     20,                            // 20 neurônios
//!     ConnectivityType::FullyConnected,
//!     0.2,                           // 20% inibitórios
//!     0.15,                          // Threshold de disparo
//! );
//!
//! net.set_learning_mode(LearningMode::STDP);
//! net.set_weight_decay(0.002);
//!
//! // Loop de simulação
//! let inputs = vec![0.0; 20];
//! net.update(&inputs);
//! ```
//!
//! ## Exemplo com AutoConfig
//!
//! ```rust,no_run
//! use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};
//!
//! // Define tarefa
//! let task = TaskSpec {
//!     num_sensors: 8,
//!     num_actuators: 4,
//!     task_type: TaskType::ReinforcementLearning {
//!         reward_density: RewardDensity::Auto,
//!         temporal_horizon: Some(100),
//!     },
//! };
//!
//! // AutoConfig deriva todos os 80+ parâmetros automaticamente
//! let config = AutoConfig::from_task(task);
//! config.print_report();
//!
//! // Cria rede otimizada
//! let mut network = config.build_network().expect("Configuração válida");
//! ```
//!
//! ## Exemplo com Working Memory + Curiosidade
//!
//! ```rust,no_run
//! use nenv_v2::working_memory::WorkingMemoryPool;
//! use nenv_v2::intrinsic_motivation::CuriosityModule;
//!
//! // Working memory para manter contexto
//! let mut wm = WorkingMemoryPool::new(7, 64);
//! let pattern = vec![0.5; 64];
//! wm.encode(pattern, 0);
//!
//! // Curiosidade para exploração autônoma
//! let mut curiosity = CuriosityModule::new(64, 4);
//! let state = vec![0.5; 64];
//! let action = vec![1.0, 0.0, 0.0, 0.0];
//! let next_state = vec![0.6; 64];
//! let intrinsic_reward = curiosity.compute_intrinsic_reward(&state, &action, &next_state);
//! ```

// ============================================================================
// CONSTANTES DO SISTEMA
// ============================================================================

pub mod constants;

// ============================================================================
// MÓDULO DE PLASTICIDADE (decomposição do Dendritoma)
// ============================================================================

pub mod plasticity;

// ============================================================================
// MÓDULOS CORE
// ============================================================================

pub mod dendritoma;
pub mod glia;
pub mod lru_cache;
pub mod nenv;
pub mod network;
pub mod neuromodulation;
pub mod sparse;

// ============================================================================
// MÓDULOS COGNITIVOS (v2.0)
// ============================================================================

pub mod working_memory;
pub mod predictive;
pub mod intrinsic_motivation;

// ============================================================================
// MÓDULOS DE CONFIGURAÇÃO
// ============================================================================

pub mod autoconfig;

// ============================================================================
// RE-EXPORTAÇÕES PARA CONVENIÊNCIA
// ============================================================================

// Core
pub use dendritoma::{Dendritoma, DendritomaStats};
pub use glia::{Glia, GliaStats};
pub use nenv::{NeuronType, SpikeOrigin, NENV};
pub use network::{ConnectivityType, LearningMode, Network, NetworkState, NetworkStats};
pub use neuromodulation::{Neuromodulator, NeuromodulatorType, NeuromodulationSystem, NeuromodulationStats};

// Cognitivos
pub use working_memory::{WorkingMemoryPool, WMSlot, WMStats};
pub use predictive::{
    PredictiveHierarchy, PredictiveLayer, PredictiveUnit, PredictiveOutput, HierarchyStats, ActiveInference,
    // Novas estruturas não-lineares (v2.0)
    ActivationFn, DenseLayer, NonLinearGenerativeModel, GenerativeModelStats,
    NonLinearPredictiveLayer, DeepPredictiveHierarchy, DeepHierarchyStats,
};
pub use intrinsic_motivation::{CuriosityModule, CuriosityStats, ForwardModel, RandomNetworkDistillation};

// AutoConfig
pub use autoconfig::{
    AutoConfig, TaskSpec, TaskType, RewardDensity,
    DerivedArchitecture, NetworkParams,
    EnergyParams, STDPParams, EligibilityParams, STPParams, CompetitionParams,
    AdaptiveState, NetworkIssue, CorrectiveAction, SleepOutcome,
};

// ============================================================================
// PRELUDE - Importação Conveniente
// ============================================================================

/// Prelude para importação rápida dos tipos mais comuns
pub mod prelude {
    pub use crate::network::{Network, ConnectivityType, LearningMode, NetworkState};
    pub use crate::nenv::{NENV, NeuronType, SpikeOrigin};
    pub use crate::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};
    pub use crate::working_memory::WorkingMemoryPool;
    pub use crate::predictive::PredictiveHierarchy;
    pub use crate::intrinsic_motivation::CuriosityModule;
    pub use crate::neuromodulation::{NeuromodulationSystem, NeuromodulatorType};
}

// ============================================================================
// VERSÃO E METADADOS (re-exportado de constants)
// ============================================================================

pub use constants::{VERSION, PROJECT_NAME, DESCRIPTION, version_info};

// ============================================================================
// TESTES DE INTEGRAÇÃO
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_basic_network_creation() {
        let network = Network::new(10, ConnectivityType::FullyConnected, 0.2, 0.15);
        assert_eq!(network.num_neurons(), 10);
    }

    #[test]
    fn test_autoconfig_creates_valid_network() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        };

        let config = AutoConfig::from_task(task);
        let network = config.build_network();
        
        assert!(network.is_ok());
    }

    #[test]
    fn test_working_memory_integration() {
        let mut wm = WorkingMemoryPool::new(5, 10);
        let pattern = vec![0.5; 10];
        
        let slot = wm.encode(pattern.clone(), 0);
        assert!(slot.is_some());
        
        let retrieved = wm.retrieve(slot.unwrap());
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), pattern.len());
    }

    #[test]
    fn test_curiosity_module_integration() {
        let mut curiosity = CuriosityModule::new(4, 2);
        
        let state = vec![0.5; 4];
        let action = vec![1.0, 0.0];
        let next_state = vec![0.6, 0.4, 0.5, 0.5];
        
        let reward = curiosity.compute_intrinsic_reward(&state, &action, &next_state);
        assert!(reward >= 0.0);
    }

    #[test]
    fn test_predictive_hierarchy_integration() {
        let mut hierarchy = PredictiveHierarchy::new_three_level(8);
        
        let input = vec![0.5; 8];
        let output = hierarchy.process(&input);
        
        assert_eq!(output.predictions.len(), 8);
        assert!(output.free_energy >= 0.0);
    }

    #[test]
    fn test_neuromodulation_integration() {
        let mut nm = NeuromodulationSystem::new();
        
        let rpe = nm.process_reward(1.0);
        assert!(rpe > 0.0);
        
        let da_level = nm.get_level(NeuromodulatorType::Dopamine);
        assert!(da_level > 0.0);
    }

    #[test]
    fn test_full_pipeline() {
        // 1. Cria rede via AutoConfig
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 2,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Moderate,
                temporal_horizon: Some(50),
            },
        };
        
        let config = AutoConfig::from_task(task);
        let mut network = config.build_network().unwrap();
        
        // 2. Cria working memory
        let mut wm = WorkingMemoryPool::new(5, network.num_neurons());
        
        // 3. Cria módulo de curiosidade
        let mut curiosity = CuriosityModule::new(4, 2);
        
        // 4. Simula alguns passos
        for step in 0..10 {
            let inputs = vec![0.5; network.num_neurons()];
            network.update(&inputs);
            
            // Armazena atividade na WM
            let activity: Vec<f64> = network.neurons.iter()
                .map(|n| if n.is_firing { 1.0 } else { 0.0 })
                .collect();
            wm.encode(activity, step as i64);
            
            // Atualiza WM
            wm.sustain();
        }
        
        // Verifica estado final
        assert!(network.current_time_step > 0);
        assert!(wm.active_count() > 0);
    }
}
