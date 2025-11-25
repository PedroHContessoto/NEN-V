//! Testes do sistema adaptativo runtime

use nenv_visual_sim::autoconfig::*;

#[test]
fn test_adaptive_system_dead_network_recovery() {
    // Cria uma rede pequena
    let task = TaskSpec {
        num_sensors: 2,
        num_actuators: 2,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);
    let mut network = config.build_network().unwrap();
    let mut adaptive = AdaptiveState::new();

    // Força threshold muito alto (rede vai "morrer")
    for neuron in &mut network.neurons {
        neuron.threshold = 5.0; // Impossível disparar
    }

    // Simula rede por 100 steps sem adaptação
    let mut firing_rates_before = Vec::new();
    for _ in 0..100 {
        let inputs = vec![1.0; network.num_neurons()];
        network.update(&inputs);

        let fr = network.num_firing() as f64 / network.num_neurons() as f64;
        firing_rates_before.push(fr);
    }

    // FR média deve ser muito baixa (rede morta)
    let avg_fr_before: f64 = firing_rates_before.iter().sum::<f64>() / firing_rates_before.len() as f64;
    assert!(avg_fr_before < 0.01, "Rede deveria estar morta");

    // Agora ativa sistema adaptativo
    println!("\n🔧 Ativando sistema adaptativo...");

    for step in 100..300 {
        let inputs = vec![1.0; network.num_neurons()];
        network.update(&inputs);

        // Monitora e adapta
        let adapted = monitor_and_adapt(
            &mut network,
            &mut adaptive,
            config.params.target_firing_rate,
            step,
            step % 50 == 0, // verbose a cada 50 steps
        );

        if adapted && step % 50 == 0 {
            println!("  Step {}: Adaptação aplicada", step);
        }
    }

    // Coleta FR final
    let mut firing_rates_after = Vec::new();
    for _ in 300..350 {
        let inputs = vec![1.0; network.num_neurons()];
        network.update(&inputs);

        let fr = network.num_firing() as f64 / network.num_neurons() as f64;
        firing_rates_after.push(fr);
    }

    let avg_fr_after: f64 = firing_rates_after.iter().sum::<f64>() / firing_rates_after.len() as f64;

    println!("\n📊 Resultados:");
    println!("  • FR antes: {:.4}", avg_fr_before);
    println!("  • FR depois: {:.4}", avg_fr_after);
    println!("  • Adaptações aplicadas: {}", adaptive.adaptation_count());

    // Sistema deve ter recuperado a rede
    assert!(avg_fr_after > avg_fr_before * 5.0, "FR deveria aumentar significativamente");
    assert!(adaptive.adaptation_count() > 0, "Deve ter aplicado adaptações");
}

#[test]
fn test_adaptive_system_energy_depletion() {
    let task = TaskSpec {
        num_sensors: 2,
        num_actuators: 2,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);
    let mut network = config.build_network().unwrap();
    let mut adaptive = AdaptiveState::new();

    // Força depleção energética (recovery muito baixo)
    for neuron in &mut network.neurons {
        neuron.glia.energy_recovery_rate = 0.1; // Quase não recupera
        neuron.threshold = 0.1; // Dispara muito (gasta energia)
    }

    // Simula sem adaptação
    for step in 0..50 {
        let inputs = vec![1.0; network.num_neurons()];
        network.update(&inputs);

        adaptive.record_metrics(
            network.num_firing() as f64 / network.num_neurons() as f64,
            network.average_energy(),
        );
    }

    let energy_before = network.average_energy();
    assert!(energy_before < 50.0, "Energia deveria estar baixa");

    // Ativa adaptação
    println!("\n🔧 Detectando depleção energética...");

    for step in 50..150 {
        let inputs = vec![1.0; network.num_neurons()];
        network.update(&inputs);

        monitor_and_adapt(
            &mut network,
            &mut adaptive,
            config.params.target_firing_rate,
            step,
            step == 100, // verbose apenas no step 100
        );
    }

    let energy_after = network.average_energy();

    println!("\n📊 Resultados:");
    println!("  • Energia antes: {:.1}", energy_before);
    println!("  • Energia depois: {:.1}", energy_after);
    println!("  • Adaptações: {}", adaptive.adaptation_count());

    // Energia deve ter melhorado
    assert!(energy_after > energy_before + 10.0, "Energia deveria ter recuperado");
}

#[test]
fn test_adaptive_cooldown_prevents_oscillation() {
    let task = TaskSpec {
        num_sensors: 2,
        num_actuators: 2,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: None,
        },
    };

    let config = AutoConfig::from_task(task);
    let mut network = config.build_network().unwrap();
    let mut adaptive = AdaptiveState::new();

    // Força problema contínuo
    for neuron in &mut network.neurons {
        neuron.threshold = 5.0; // Muito alto
    }

    let mut adaptation_steps = Vec::new();

    // Simula com monitoramento
    for step in 0..500 {
        let inputs = vec![1.0; network.num_neurons()];
        network.update(&inputs);

        let adapted = monitor_and_adapt(
            &mut network,
            &mut adaptive,
            config.params.target_firing_rate,
            step,
            false,
        );

        if adapted {
            adaptation_steps.push(step);
        }
    }

    println!("\n📊 Adaptações aplicadas em steps: {:?}", adaptation_steps);
    println!("Total de adaptações: {}", adaptive.adaptation_count());

    // Deve ter aplicado poucas adaptações devido ao cooldown
    assert!(
        adaptive.adaptation_count() < 10,
        "Cooldown deve limitar adaptações"
    );

    // Adaptações devem estar espaçadas
    if adaptation_steps.len() >= 2 {
        for i in 1..adaptation_steps.len() {
            let gap = adaptation_steps[i] - adaptation_steps[i - 1];
            assert!(
                gap >= 100,
                "Adaptações devem estar espaçadas por cooldown (gap={}, mínimo=100)",
                gap
            );
        }
    }
}

#[test]
fn test_adaptive_detection_accuracy() {
    let mut adaptive = AdaptiveState::new();

    // Simula rede morta
    for _ in 0..60 {
        adaptive.record_metrics(0.0, 80.0);
    }

    let issues = adaptive.detect_issues(0.15, 60);
    assert!(issues.contains(&NetworkIssue::DeadNetwork));

    // Reseta e simula runaway
    let mut adaptive = AdaptiveState::new();
    for _ in 0..20 {
        adaptive.record_metrics(0.98, 50.0);
    }

    let issues = adaptive.detect_issues(0.15, 20);
    assert!(issues.contains(&NetworkIssue::RunawayExcitation));

    // Reseta e simula baixa energia
    let mut adaptive = AdaptiveState::new();
    for _ in 0..20 {
        adaptive.record_metrics(0.15, 25.0);
    }

    let issues = adaptive.detect_issues(0.15, 20);
    assert!(issues.contains(&NetworkIssue::EnergyDepletionRisk));
}
