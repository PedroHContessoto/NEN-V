//! # Derivação de Arquitetura
//!
//! Deriva automaticamente a arquitetura da rede a partir da especificação da tarefa.

use crate::network::ConnectivityType;
use super::task::{TaskSpec, TaskType, RewardDensity};

/// Arquitetura derivada automaticamente
#[derive(Debug, Clone)]
pub struct DerivedArchitecture {
    /// Número total de neurônios
    pub total_neurons: usize,

    /// Índices dos neurônios sensores
    pub sensor_indices: std::ops::Range<usize>,

    /// Índices dos neurônios hidden
    pub hidden_indices: std::ops::Range<usize>,

    /// Índices dos neurônios atuadores
    pub actuator_indices: std::ops::Range<usize>,

    /// Tipo de conectividade
    pub connectivity: ConnectivityType,

    /// Razão de neurônios inibitórios [0, 1]
    pub inhibitory_ratio: f64,

    /// Threshold inicial para disparo
    pub initial_threshold: f64,
}

impl DerivedArchitecture {
    /// Deriva arquitetura a partir da especificação da tarefa
    pub fn from_task(task: &TaskSpec) -> Self {
        let num_hidden = derive_num_hidden(
            task.num_sensors,
            task.num_actuators,
            &task.task_type,
        );

        let total_neurons = task.num_sensors + num_hidden + task.num_actuators;

        let sensor_indices = 0..task.num_sensors;
        let hidden_indices = task.num_sensors..(task.num_sensors + num_hidden);
        let actuator_indices = (task.num_sensors + num_hidden)..total_neurons;

        let connectivity = derive_connectivity(total_neurons, &task.task_type);
        let inhibitory_ratio = derive_inhibitory_ratio(&task.task_type);
        let initial_threshold = derive_initial_threshold(connectivity, &task.task_type);

        DerivedArchitecture {
            total_neurons,
            sensor_indices,
            hidden_indices,
            actuator_indices,
            connectivity,
            inhibitory_ratio,
            initial_threshold,
        }
    }

    /// Retorna número de neurônios por camada
    pub fn layer_sizes(&self) -> (usize, usize, usize) {
        (
            self.sensor_indices.len(),
            self.hidden_indices.len(),
            self.actuator_indices.len(),
        )
    }

    /// Estima a capacidade de memória da rede
    ///
    /// Baseado na regra de Hopfield: ~0.14N para redes de memória associativa
    pub fn estimate_memory_capacity(&self) -> usize {
        let excitatory_neurons = (self.total_neurons as f64 * (1.0 - self.inhibitory_ratio)) as usize;
        let base_capacity = (excitatory_neurons as f64 * 0.05) as usize;
        base_capacity.max(1)
    }

    /// Retorna número de neurônios excitatórios
    pub fn num_excitatory(&self) -> usize {
        ((1.0 - self.inhibitory_ratio) * self.total_neurons as f64) as usize
    }

    /// Retorna número de neurônios inibitórios
    pub fn num_inhibitory(&self) -> usize {
        (self.inhibitory_ratio * self.total_neurons as f64) as usize
    }

    /// Verifica se usa conectividade esparsa
    pub fn is_sparse(&self) -> bool {
        matches!(self.connectivity, ConnectivityType::Grid2D)
    }
}

// ============================================================================
// FUNÇÕES DE DERIVAÇÃO
// ============================================================================

/// Deriva número de neurônios hidden
fn derive_num_hidden(num_sensors: usize, num_actuators: usize, task_type: &TaskType) -> usize {
    // Média geométrica como base
    let geometric_mean = ((num_sensors * num_actuators) as f64).sqrt();

    // Fator de expansão baseado no tipo de tarefa
    let expansion_factor = match task_type {
        TaskType::ReinforcementLearning { temporal_horizon, .. } => {
            let base = 2.0;
            match temporal_horizon {
                Some(h) if *h > 200 => base * 1.5,  // Horizonte longo precisa mais memória
                Some(h) if *h > 100 => base * 1.25,
                _ => base,
            }
        }
        TaskType::SupervisedClassification { num_classes } => {
            let base = 1.5;
            if *num_classes > 10 { base * 1.5 } else { base }
        }
        TaskType::AssociativeMemory { pattern_capacity } => {
            // Regra de Hopfield: N >= patterns / 0.14
            let min_neurons = (*pattern_capacity as f64 / 0.14).ceil();
            (min_neurons / geometric_mean).max(3.0)
        }
    };

    let base_hidden = (geometric_mean * expansion_factor) as usize;
    let io_size = num_sensors + num_actuators;

    // Clamp entre IO e 10x IO
    base_hidden.clamp(io_size, io_size * 10)
}

/// Deriva tipo de conectividade
fn derive_connectivity(total_neurons: usize, task_type: &TaskType) -> ConnectivityType {
    match task_type {
        TaskType::ReinforcementLearning { .. } => {
            // RL com muitos neurônios usa Grid2D para eficiência
            if total_neurons < 50 {
                ConnectivityType::FullyConnected
            } else {
                ConnectivityType::Grid2D
            }
        }
        TaskType::SupervisedClassification { .. } => {
            // Classificação geralmente precisa de conexões globais
            ConnectivityType::FullyConnected
        }
        TaskType::AssociativeMemory { .. } => {
            // Memória associativa usa padrões locais
            ConnectivityType::Grid2D
        }
    }
}

/// Deriva razão de neurônios inibitórios
fn derive_inhibitory_ratio(task_type: &TaskType) -> f64 {
    match task_type {
        TaskType::ReinforcementLearning { .. } => 0.20,      // 20% - padrão cortical
        TaskType::SupervisedClassification { .. } => 0.25,   // 25% - mais controle
        TaskType::AssociativeMemory { .. } => 0.15,          // 15% - menos inibição
    }
}

/// Deriva threshold inicial
///
/// Calibrado para inputs ESPARSOS típicos:
/// - 10-30% dos sensores ativos por step
/// - Valores normalizados [0, 1]
/// - Pesos excitatórios ~0.25
///
/// Conta: 3 inputs ativos × 0.25 peso = 0.75 → threshold ~0.5 permite disparo
fn derive_initial_threshold(connectivity: ConnectivityType, task_type: &TaskType) -> f64 {
    // Base calibrada para inputs esparsos com pesos ~0.25
    // Queremos que 2-4 inputs ativos consigam disparar
    let base_threshold = match connectivity {
        ConnectivityType::FullyConnected => 0.20,  // 2 inputs × 0.25 = 0.50 > 0.20 ✓
        ConnectivityType::Grid2D => 0.12,          // Vizinhança local, menos inputs
        ConnectivityType::Isolated => 0.3,         // Debug
    };

    // Ajuste fino por tarefa
    let task_multiplier = match task_type {
        TaskType::ReinforcementLearning { reward_density, .. } => {
            match reward_density {
                RewardDensity::Sparse => 0.85,   // Mais sensível para reward esparso
                RewardDensity::Dense => 1.1,    // Menos sensível
                _ => 1.0,
            }
        }
        TaskType::SupervisedClassification { .. } => 1.2,  // Mais seletivo
        TaskType::AssociativeMemory { .. } => 0.9,         // Menos seletivo para padrões
    };

    base_threshold * task_multiplier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_derivation() {
        let task = TaskSpec {
            num_sensors: 8,
            num_actuators: 4,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        };

        let arch = DerivedArchitecture::from_task(&task);

        assert_eq!(arch.sensor_indices.len(), 8);
        assert_eq!(arch.actuator_indices.len(), 4);
        assert!(arch.hidden_indices.len() >= 12); // >= IO size
        assert!(arch.inhibitory_ratio > 0.0 && arch.inhibitory_ratio < 1.0);
    }

    #[test]
    fn test_memory_capacity() {
        let task = TaskSpec {
            num_sensors: 100,
            num_actuators: 100,
            task_type: TaskType::AssociativeMemory { pattern_capacity: 10 },
        };

        let arch = DerivedArchitecture::from_task(&task);
        let capacity = arch.estimate_memory_capacity();

        assert!(capacity >= 1);
    }

    #[test]
    fn test_sparse_connectivity() {
        let small_task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        };

        let large_task = TaskSpec {
            num_sensors: 50,
            num_actuators: 50,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        };

        let small_arch = DerivedArchitecture::from_task(&small_task);
        let large_arch = DerivedArchitecture::from_task(&large_task);

        // Rede pequena deve ser fully connected
        assert!(!small_arch.is_sparse());

        // Rede grande deve ser esparsa
        assert!(large_arch.is_sparse());
    }
}
