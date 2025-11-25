//! Derivação automática de arquitetura neural
//!
//! NÍVEL 0: Determina a estrutura da rede baseado apenas na interface com o ambiente

use super::*;
use crate::network::ConnectivityType;

impl DerivedArchitecture {
    /// Deriva arquitetura completa a partir da especificação da tarefa
    pub fn from_task(task: &TaskSpec) -> Self {
        // 1. Calcula número de neurônios hidden
        let num_hidden = derive_num_hidden(
            task.num_sensors,
            task.num_actuators,
            &task.task_type,
        );

        // 2. Calcula total de neurônios
        let total_neurons = task.num_sensors + num_hidden + task.num_actuators;

        // 3. Define índices de cada camada
        let sensor_indices = 0..task.num_sensors;
        let hidden_indices = task.num_sensors..(task.num_sensors + num_hidden);
        let actuator_indices = (task.num_sensors + num_hidden)..total_neurons;

        // 4. Determina topologia de conectividade
        let connectivity = derive_connectivity(total_neurons, &task.task_type);

        // 5. Determina razão inibitória
        let inhibitory_ratio = derive_inhibitory_ratio(&task.task_type);

        // 6. Determina threshold inicial
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
}

// ============================================================================
// FUNÇÕES DE DERIVAÇÃO
// ============================================================================

/// Calcula número de neurônios hidden
///
/// Regra biológica: Camada hidden ~ média geométrica de I/O × fator de expansão
/// Similar ao teorema de Kolmogorov-Arnold (1 hidden layer suficiente)
///
/// # Justificativa Biológica
/// Córtex visual: ~100M inputs (retina) → ~1B neurons (V1) → ~10M outputs
/// Ratio: 1:10:0.1 (entrada:hidden:saída)
pub fn derive_num_hidden(
    num_sensors: usize,
    num_actuators: usize,
    task_type: &TaskType,
) -> usize {
    // Média geométrica (valor base)
    let geometric_mean = ((num_sensors * num_actuators) as f64).sqrt();

    // Fator de expansão baseado na complexidade da tarefa
    let expansion_factor = match task_type {
        TaskType::ReinforcementLearning { .. } => 2.0,  // RL precisa exploração
        TaskType::SupervisedClassification { .. } => 1.5,
        TaskType::AssociativeMemory { .. } => 3.0,  // Memória precisa capacidade
    };

    let base_hidden = (geometric_mean * expansion_factor) as usize;

    // Clamp: hidden deve ser pelo menos igual a I/O, no máximo 10× I/O
    let io_size = num_sensors + num_actuators;
    base_hidden.clamp(io_size, io_size * 10)
}

/// Determina topologia de conectividade
///
/// # Regras
/// - RL com rede pequena (<50): FullyConnected (melhor para exploração)
/// - RL com rede grande (≥50): Grid2D (escalabilidade)
/// - Classificação: FullyConnected (features globais)
/// - Memória: Grid2D (localidade topográfica)
pub fn derive_connectivity(
    total_neurons: usize,
    task_type: &TaskType,
) -> ConnectivityType {
    match task_type {
        TaskType::ReinforcementLearning { .. } => {
            if total_neurons < 50 {
                ConnectivityType::FullyConnected
            } else {
                ConnectivityType::Grid2D
            }
        }

        TaskType::SupervisedClassification { .. } => {
            ConnectivityType::FullyConnected
        }

        TaskType::AssociativeMemory { .. } => {
            ConnectivityType::Grid2D
        }
    }
}

/// Determina razão de neurônios inibitórios (Dale's Principle)
///
/// # Justificativa Biológica
/// Córtex cerebral: ~20-30% de neurônios GABAérgicos (inibitórios)
/// Variação por região:
/// - Sensory cortex: ~25% (controle de ganho)
/// - Motor cortex: ~15% (precisão)
/// - Hippocampus: ~10-15% (recall de memória)
pub fn derive_inhibitory_ratio(task_type: &TaskType) -> f64 {
    match task_type {
        TaskType::ReinforcementLearning { .. } => 0.20,  // Balanço padrão (20%)
        TaskType::SupervisedClassification { .. } => 0.25,  // Mais seletividade
        TaskType::AssociativeMemory { .. } => 0.15,  // Facilita recall
    }
}

/// Determina threshold de disparo inicial
///
/// # Regra
/// Threshold deve permitir disparo quando ~10-30% dos inputs estão ativos
///
/// # Justificativa
/// - FullyConnected: Muitos inputs → threshold alto (evita saturação)
/// - Grid2D: Poucos inputs (8 vizinhos) → threshold baixo (sensibilidade)
pub fn derive_initial_threshold(
    connectivity: ConnectivityType,
    task_type: &TaskType,
) -> f64 {
    // Base threshold
    let base_threshold = match connectivity {
        ConnectivityType::FullyConnected => 0.3,  // 30% dos inputs
        ConnectivityType::Grid2D => 0.15,  // 15% (poucos inputs)
        ConnectivityType::Isolated => 0.1,
    };

    // Ajuste por tipo de tarefa
    let task_multiplier = match task_type {
        TaskType::ReinforcementLearning { .. } => 1.0,  // Padrão
        TaskType::SupervisedClassification { .. } => 1.3,  // Mais seletivo
        TaskType::AssociativeMemory { .. } => 0.8,  // Mais sensível (recall)
    };

    base_threshold * task_multiplier
}

// ============================================================================
// TESTES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_num_hidden() {
        // Caso básico: 4 sensors, 4 actuators, RL
        let num_hidden = derive_num_hidden(
            4,
            4,
            &TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        );

        // geometric_mean = sqrt(4*4) = 4
        // base = 4 * 2.0 = 8
        assert_eq!(num_hidden, 8);
    }

    #[test]
    fn test_derive_num_hidden_memory() {
        // Memória precisa mais neurônios (fator 3.0)
        let num_hidden = derive_num_hidden(
            4,
            4,
            &TaskType::AssociativeMemory {
                pattern_capacity: 100,
            },
        );

        // base = 4 * 3.0 = 12
        assert_eq!(num_hidden, 12);
    }

    #[test]
    fn test_derive_connectivity_small_rl() {
        // Rede pequena RL (<50): FullyConnected
        let conn = derive_connectivity(
            20,
            &TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        );

        assert!(matches!(conn, ConnectivityType::FullyConnected));
    }

    #[test]
    fn test_derive_connectivity_large_rl() {
        // Rede grande RL (≥50): Grid2D
        let conn = derive_connectivity(
            100,
            &TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        );

        assert!(matches!(conn, ConnectivityType::Grid2D));
    }

    #[test]
    fn test_derive_inhibitory_ratio() {
        let rl_ratio = derive_inhibitory_ratio(&TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        });

        assert_eq!(rl_ratio, 0.20);

        let memory_ratio = derive_inhibitory_ratio(&TaskType::AssociativeMemory {
            pattern_capacity: 100,
        });

        assert_eq!(memory_ratio, 0.15);
    }

    #[test]
    fn test_derive_initial_threshold() {
        // FullyConnected → threshold mais alto
        let fc_threshold = derive_initial_threshold(
            ConnectivityType::FullyConnected,
            &TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        );

        // Grid2D → threshold mais baixo
        let grid_threshold = derive_initial_threshold(
            ConnectivityType::Grid2D,
            &TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        );

        assert!(fc_threshold > grid_threshold);
        assert_eq!(fc_threshold, 0.3);
        assert_eq!(grid_threshold, 0.15);
    }

    #[test]
    fn test_full_architecture_derivation() {
        let task = TaskSpec {
            num_sensors: 4,
            num_actuators: 4,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Auto,
                temporal_horizon: None,
            },
        };

        let arch = DerivedArchitecture::from_task(&task);

        // Verifica estrutura
        assert_eq!(arch.sensor_indices, 0..4);
        assert_eq!(arch.hidden_indices, 4..12);  // 8 hidden neurons
        assert_eq!(arch.actuator_indices, 12..16);
        assert_eq!(arch.total_neurons, 16);

        // Verifica parâmetros
        assert!(matches!(arch.connectivity, ConnectivityType::FullyConnected));
        assert_eq!(arch.inhibitory_ratio, 0.20);
        assert_eq!(arch.initial_threshold, 0.3);
    }
}
