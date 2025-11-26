//! # Especificação da Tarefa
//!
//! Define os tipos de entrada do usuário para o sistema AutoConfig.

/// Especificação mínima da tarefa
#[derive(Debug, Clone)]
pub struct TaskSpec {
    /// Número de canais de entrada (sensores)
    pub num_sensors: usize,

    /// Número de canais de saída (atuadores)
    pub num_actuators: usize,

    /// Tipo de tarefa
    pub task_type: TaskType,
}

impl TaskSpec {
    /// Cria especificação para tarefa de RL
    pub fn reinforcement_learning(
        num_sensors: usize,
        num_actuators: usize,
        reward_density: RewardDensity,
    ) -> Self {
        Self {
            num_sensors,
            num_actuators,
            task_type: TaskType::ReinforcementLearning {
                reward_density,
                temporal_horizon: None,
            },
        }
    }

    /// Cria especificação para classificação
    pub fn classification(
        num_sensors: usize,
        num_classes: usize,
    ) -> Self {
        Self {
            num_sensors,
            num_actuators: num_classes,
            task_type: TaskType::SupervisedClassification { num_classes },
        }
    }

    /// Cria especificação para memória associativa
    pub fn associative_memory(
        num_sensors: usize,
        pattern_capacity: usize,
    ) -> Self {
        Self {
            num_sensors,
            num_actuators: num_sensors,
            task_type: TaskType::AssociativeMemory { pattern_capacity },
        }
    }

    /// Retorna o tamanho total de I/O
    pub fn io_size(&self) -> usize {
        self.num_sensors + self.num_actuators
    }

    /// Verifica se é tarefa de RL
    pub fn is_reinforcement_learning(&self) -> bool {
        matches!(self.task_type, TaskType::ReinforcementLearning { .. })
    }

    /// Verifica se é tarefa de classificação
    pub fn is_classification(&self) -> bool {
        matches!(self.task_type, TaskType::SupervisedClassification { .. })
    }

    /// Verifica se é tarefa de memória
    pub fn is_memory(&self) -> bool {
        matches!(self.task_type, TaskType::AssociativeMemory { .. })
    }
}

/// Tipo de tarefa
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Aprendizado por reforço
    ReinforcementLearning {
        reward_density: RewardDensity,
        temporal_horizon: Option<usize>,
    },

    /// Classificação supervisionada
    SupervisedClassification {
        num_classes: usize,
    },

    /// Memória associativa
    AssociativeMemory {
        pattern_capacity: usize,
    },
}

impl TaskType {
    /// Retorna nome do tipo de tarefa
    pub fn name(&self) -> &str {
        match self {
            Self::ReinforcementLearning { .. } => "ReinforcementLearning",
            Self::SupervisedClassification { .. } => "SupervisedClassification",
            Self::AssociativeMemory { .. } => "AssociativeMemory",
        }
    }

    /// Retorna densidade de reward (se aplicável)
    pub fn reward_density(&self) -> Option<RewardDensity> {
        match self {
            Self::ReinforcementLearning { reward_density, .. } => Some(*reward_density),
            _ => None,
        }
    }

    /// Retorna horizonte temporal (se aplicável)
    pub fn temporal_horizon(&self) -> Option<usize> {
        match self {
            Self::ReinforcementLearning { temporal_horizon, .. } => *temporal_horizon,
            _ => None,
        }
    }
}

/// Densidade de recompensas
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RewardDensity {
    /// Determinar automaticamente
    Auto,
    /// Recompensas frequentes (a cada passo)
    Dense,
    /// Recompensas moderadas
    Moderate,
    /// Recompensas raras (fim de episódio)
    Sparse,
}

impl RewardDensity {
    /// Retorna valor numérico aproximado da densidade
    pub fn approximate_value(&self) -> f64 {
        match self {
            Self::Dense => 0.15,
            Self::Moderate => 0.05,
            Self::Sparse => 0.01,
            Self::Auto => 0.05, // Default moderado
        }
    }

    /// Determina se é considerada esparsa
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_spec_creation() {
        let task = TaskSpec::reinforcement_learning(8, 4, RewardDensity::Auto);

        assert_eq!(task.num_sensors, 8);
        assert_eq!(task.num_actuators, 4);
        assert!(task.is_reinforcement_learning());
    }

    #[test]
    fn test_classification_task() {
        let task = TaskSpec::classification(64, 10);

        assert_eq!(task.num_sensors, 64);
        assert_eq!(task.num_actuators, 10);
        assert!(task.is_classification());
    }

    #[test]
    fn test_reward_density() {
        assert!(RewardDensity::Sparse.is_sparse());
        assert!(!RewardDensity::Dense.is_sparse());
        assert!(RewardDensity::Sparse.approximate_value() < RewardDensity::Dense.approximate_value());
    }
}
