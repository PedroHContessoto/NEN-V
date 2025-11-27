//! Diagnóstico: Por que a rede morre (FR = 0.0)?
//!
//! Este exemplo investiga as causas da morte da rede neural,
//! testando diferentes hipóteses sobre o problema.

use nenv_v2::autoconfig::{AutoConfig, RewardDensity, TaskSpec, TaskType};
use nenv_v2::network::LearningMode;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     DIAGNÓSTICO: Por que a rede morre? (FR = 0.0)          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Cria uma rede usando AutoConfig
    let task = TaskSpec {
        num_sensors: 20,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task.clone());
    let mut network = config.build_network().expect("Falha ao construir rede");

    println!("Rede criada:");
    println!("  Total de neurônios: {}", network.num_neurons());
    println!("  Sensores: {:?}", config.architecture.sensor_indices);
    println!("  Target FR: {:.4}", config.params.target_firing_rate);
    println!("  Learning mode: STDP");
    println!("  Homeo_eta: {:.4}\n", network.neurons[0].homeo_eta);

    // Liga STDP
    network.set_learning_mode(LearningMode::STDP);

    let mut rng = StdRng::seed_from_u64(42);
    let sensor_indices = config.architecture.sensor_indices.clone();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  TESTE 1: Input constante forte (todos sensores = 1.0)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Usa a rede original
    for step in 0..100 {
        // Input forte constante
        let mut inputs = vec![0.0; network.num_neurons()];
        for idx in sensor_indices.clone() {
            inputs[idx] = 1.0;
        }

        network.update(&inputs);

        if step % 20 == 0 {
            let fr = network.num_firing() as f64 / network.num_neurons() as f64;
            let avg_threshold = network.neurons.iter()
                .map(|n| n.threshold)
                .sum::<f64>() / network.num_neurons() as f64;
            let avg_energy = network.neurons.iter()
                .map(|n| n.glia.energy)
                .sum::<f64>() / network.num_neurons() as f64;

            println!("  Step {:3}: FR={:.3} | Threshold={:.4} | Energy={:.1}",
                step, fr, avg_threshold, avg_energy);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  TESTE 2: Input aleatório (10% dos sensores ativos)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Recria a rede
    let mut network2 = config.build_network().expect("Falha ao construir rede");
    network2.set_learning_mode(LearningMode::STDP);

    for step in 0..100 {
        // Input aleatório
        let mut inputs = vec![0.0; network2.num_neurons()];
        for idx in sensor_indices.clone() {
            if rng.gen_bool(0.1) {
                inputs[idx] = 1.0;
            }
        }

        network2.update(&inputs);

        if step % 20 == 0 {
            let fr = network2.num_firing() as f64 / network2.num_neurons() as f64;
            let avg_threshold = network2.neurons.iter()
                .map(|n| n.threshold)
                .sum::<f64>() / network2.num_neurons() as f64;
            let avg_energy = network2.neurons.iter()
                .map(|n| n.glia.energy)
                .sum::<f64>() / network2.num_neurons() as f64;

            println!("  Step {:3}: FR={:.3} | Threshold={:.4} | Energy={:.1}",
                step, fr, avg_threshold, avg_energy);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  TESTE 3: Inspeção detalhada de um neurônio sensor");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Recria rede e foca em um neurônio sensor
    let mut network3 = config.build_network().expect("Falha ao construir rede");
    network3.set_learning_mode(LearningMode::STDP);
    let sensor_idx = sensor_indices.start;

    println!("Neurônio sensor #{} ANTES de qualquer update:", sensor_idx);
    let neuron = &network3.neurons[sensor_idx];
    println!("  Threshold: {:.6}", neuron.threshold);
    println!("  Peso[{}] (auto-conexão): {:.6}", sensor_idx, neuron.dendritoma.weights[sensor_idx]);
    println!("  Soma total de pesos: {:.6}", neuron.dendritoma.total_weight());
    println!("  Energia: {:.1}", neuron.glia.energy);
    println!("  Homeo_eta: {:.4}\n", neuron.homeo_eta);

    // Aplica 10 updates com input direto nesse sensor
    for step in 0..10 {
        let mut inputs = vec![0.0; network3.num_neurons()];
        inputs[sensor_idx] = 1.0;

        network3.update(&inputs);

        let neuron = &network3.neurons[sensor_idx];
        println!("  Step {}: firing={} | threshold={:.6} | energy={:.1} | weight[{}]={:.6}",
            step,
            neuron.is_firing,
            neuron.threshold,
            neuron.glia.energy,
            sensor_idx,
            neuron.dendritoma.weights[sensor_idx]);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  TESTE 4: Verificação da matriz de conectividade");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Verifica auto-conexões
    let mut num_self_connections = 0;
    let mut num_total_connections = 0;

    for i in 0..network3.num_neurons() {
        let connections: Vec<usize> = network3.connectivity_matrix[i]
            .iter()
            .enumerate()
            .filter(|(_, &val)| val == 1)
            .map(|(idx, _)| idx)
            .collect();

        num_total_connections += connections.len();

        if network3.connectivity_matrix[i][i] == 1 {
            num_self_connections += 1;
        }

        if i < 3 {
            println!("  Neurônio {}: {} conexões, auto-conexão={}",
                i, connections.len(),
                network3.connectivity_matrix[i][i] == 1);
        }
    }

    println!("  ...");
    println!("  Total de auto-conexões: {}/{}", num_self_connections, network3.num_neurons());
    println!("  Média de conexões por neurônio: {:.1}",
        num_total_connections as f64 / network3.num_neurons() as f64);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  TESTE 5: Período refratário e energy gating");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Recria rede
    let mut network4 = config.build_network().expect("Falha ao construir rede");
    network4.set_learning_mode(LearningMode::STDP);
    let test_idx = sensor_indices.start;

    println!("Testando neurônio #{} com input constante:", test_idx);
    let mut inputs = vec![0.0; network4.num_neurons()];
    inputs[test_idx] = 1.0;

    for step in 0..20 {
        network4.update(&inputs);

        let neuron = &network4.neurons[test_idx];
        let time_since_fire = if neuron.last_fire_time < 0 {
            999
        } else {
            network4.current_time_step - neuron.last_fire_time
        };

        println!("  Step {:2}: fire={} | last_fire={:3} | time_since={:3} | energy={:.1}",
            step,
            neuron.is_firing,
            neuron.last_fire_time,
            time_since_fire,
            neuron.glia.energy);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  CONCLUSÕES");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("  Use os dados acima para identificar:");
    println!("  1. Se os neurônios disparam inicialmente");
    println!("  2. Se param de disparar, quando e por quê (energia/threshold)");
    println!("  3. Se há auto-conexões suficientes para propagação de input");
    println!("  4. Se o período refratário está bloqueando disparos");
}
