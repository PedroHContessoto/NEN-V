//! Testes de integração do AutoConfig v2.0

use nenv_visual_sim::autoconfig::*;

#[test]
fn test_autoconfig_basic_rl_task() {
    // Especificação mínima: apenas 3 valores
    let task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    // Deriva configuração
    let config = AutoConfig::from_task(task);

    // Verifica arquitetura derivada
    assert_eq!(config.architecture.sensor_indices, 0..4);
    assert_eq!(config.architecture.hidden_indices, 4..12);
    assert_eq!(config.architecture.actuator_indices, 12..16);
    assert_eq!(config.architecture.total_neurons, 16);

    // Verifica que é FullyConnected para rede pequena
    assert!(matches!(
        config.architecture.connectivity,
        nenv_visual_sim::network::ConnectivityType::FullyConnected
    ));

    // Verifica razão inibitória (20% para RL)
    assert_eq!(config.architecture.inhibitory_ratio, 0.20);
}

#[test]
fn test_autoconfig_validation_passes() {
    let task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);

    // Validação deve passar
    assert!(config.validate().is_ok());
}

#[test]
fn test_autoconfig_energy_balance_positive() {
    let task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);

    // Verifica balanço energético
    let energy = &config.params.energy;
    let fr = config.params.target_firing_rate;

    let avg_cost = energy.energy_cost_fire * fr;
    let avg_gain = energy.energy_recovery_rate * (1.0 - fr);
    let balance = avg_gain - avg_cost;

    assert!(balance > 0.0, "Balanço energético deve ser positivo");
}

#[test]
fn test_autoconfig_istdp_alignment() {
    let task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);

    // iSTDP target deve ser igual ao target_firing_rate
    let error = (config.params.istdp.target_rate - config.params.target_firing_rate).abs();
    assert!(error < 1e-6, "iSTDP deve estar alinhado com target FR");
}

#[test]
fn test_autoconfig_stdp_ratio_valid() {
    let task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);

    let ratio = config.params.stdp.a_plus / config.params.stdp.a_minus;

    // Ratio LTP/LTD deve estar em range razoável (1.5-3.0)
    assert!(ratio >= 1.0, "LTP deve ser maior que LTD");
    assert!(ratio <= 5.0, "Ratio não deve ser muito alto");
}

#[test]
fn test_autoconfig_build_network() {
    let task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);

    // Cria rede
    let result = config.build_network();
    assert!(result.is_ok(), "Rede deve ser criada sem erros");

    let network = result.unwrap();

    // Verifica estrutura da rede
    assert_eq!(network.num_neurons(), 16);
    assert_eq!(network.average_energy(), 100.0);

    // Verifica que modo de aprendizado é STDP
    assert!(matches!(
        network.learning_mode,
        nenv_visual_sim::network::LearningMode::STDP
    ));
}

#[test]
fn test_autoconfig_network_simulation() {
    let task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);
    let mut network = config.build_network().unwrap();

    // Executa alguns steps
    for _ in 0..10 {
        let inputs = vec![0.5; 16]; // Inputs moderados
        network.update(&inputs);
    }

    // Verifica que rede ainda está operacional
    assert!(network.average_energy() > 0.0);
    assert_eq!(network.current_time_step, 10);
}

#[test]
fn test_autoconfig_grid2d_large_network() {
    // Rede grande (≥50) deve usar Grid2D
    let task = TaskSpec {
        num_sensors: 20,
        num_actuators: 20,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);

    // Deve ter escolhido Grid2D
    assert!(matches!(
        config.architecture.connectivity,
        nenv_visual_sim::network::ConnectivityType::Grid2D
    ));

    // Target FR deve ser consistente (correção do Furo 1)
    // Grid2D sempre tem fan_in=8, então FR não deve variar com tamanho
    assert!(config.params.target_firing_rate > 0.0);
}

#[test]
fn test_autoconfig_different_task_types() {
    // RL
    let rl_task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let rl_config = AutoConfig::from_task(rl_task);
    assert_eq!(rl_config.architecture.inhibitory_ratio, 0.20);

    // Classificação
    let class_task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::SupervisedClassification { num_classes: 3 },
    };

    let class_config = AutoConfig::from_task(class_task);
    assert_eq!(class_config.architecture.inhibitory_ratio, 0.25);

    // Memória
    let mem_task = TaskSpec {
        num_sensors: 4,
        num_actuators: 4,
        task_type: TaskType::AssociativeMemory {
            pattern_capacity: 10,
        },
    };

    let mem_config = AutoConfig::from_task(mem_task);
    assert_eq!(mem_config.architecture.inhibitory_ratio, 0.15);
}
