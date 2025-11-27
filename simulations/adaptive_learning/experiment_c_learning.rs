//! EXPERIMENTO C: Validação de Aprendizado e Estabilidade
//!
//! Protocolo:
//! - Fase 1 (0-20k): Ruído (baseline)
//! - Fase 2 (20k-60k): Padrão repetido (80%) + ruído (20%)
//! - Fase 3 (60k-80k): Ruído (washout)

use nenv_v2::autoconfig::{AutoConfig, RewardDensity, TaskSpec, TaskType};
use nenv_v2::network::LearningMode;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("===============================================================");
    println!("  EXPERIMENTO C: Validação de Aprendizado e Estabilidade");
    println!("===============================================================\n");

    // Configura rede usando AutoConfig default
    let task = TaskSpec {
        num_sensors: 10,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task.clone());
    let mut network = config.build_network().expect("Falha ao construir rede");

    // Liga aprendizado STDP e ajusta LR para um cenário desafiador
    network.set_learning_mode(LearningMode::STDP);
    for n in &mut network.neurons {
        n.dendritoma.set_learning_rate(0.005);
    }

    // Padrão: sensores pares ativos
    let sensor_indices = config.architecture.sensor_indices.clone();
    let mut pattern_input = vec![0.0; network.num_neurons()];
    for i in sensor_indices.clone() {
        if i % 2 == 0 {
            pattern_input[i] = 1.0;
        }
    }

    let total_steps = 80_000;
    let phase_1_end = 20_000;
    let phase_2_end = 60_000;

    let mut rng = StdRng::seed_from_u64(1234);
    let mut inputs = vec![0.0; network.num_neurons()];

    let mut file = File::create("learning_log.csv").expect("Erro ao criar log");
    writeln!(
        file,
        "step,phase,fr,max_fr,avg_threshold,avg_weight_pattern,avg_weight_noise"
    )
    .unwrap();

    println!("🚀 Iniciando simulação (80k steps)...");
    println!("   Fase 1 (0-20k): Ruído (Baseline)");
    println!("   Fase 2 (20k-60k): Padrão Repetido (Aprendizado)");
    println!("   Fase 3 (60k-80k): Ruído (Teste de Memória/Estabilidade)\n");

    let mut max_fr_seen = 0.0;

    for step in 0..total_steps {
        inputs.fill(0.0);

        let phase_name;
        if step < phase_1_end {
            phase_name = "BASELINE";
            for i in sensor_indices.clone() {
                if rng.gen_bool(0.1) {
                    inputs[i] = 1.0;
                }
            }
        } else if step < phase_2_end {
            phase_name = "TRAINING";
            if rng.gen_bool(0.8) {
                for (i, &val) in pattern_input.iter().enumerate() {
                    if val > 0.0 {
                        inputs[i] = val;
                    }
                }
            } else {
                for i in sensor_indices.clone() {
                    if rng.gen_bool(0.1) {
                        inputs[i] = 1.0;
                    }
                }
            }
        } else {
            phase_name = "WASHOUT";
            for i in sensor_indices.clone() {
                if rng.gen_bool(0.1) {
                    inputs[i] = 1.0;
                }
            }
        }

        network.update(&inputs);

        // Métricas a cada 100 steps
        if step % 100 == 0 {
            let fr = network.num_firing() as f64 / network.num_neurons() as f64;
            if fr > max_fr_seen {
                max_fr_seen = fr;
            }

            let avg_thresh: f64 = network
                .neurons
                .iter()
                .map(|n| n.threshold)
                .sum::<f64>()
                / network.num_neurons() as f64;

            // Pesos: média das conexões de sensores pares (padrão) vs ímpares (ruído)
            let mut w_pattern_sum = 0.0;
            let mut w_pattern_count = 0;
            let mut w_noise_sum = 0.0;
            let mut w_noise_count = 0;

            for n in &network.neurons {
                if config.architecture.hidden_indices.contains(&n.id) {
                    for (idx, &w) in n.dendritoma.weights.iter().enumerate() {
                        if sensor_indices.contains(&idx) {
                            if idx % 2 == 0 {
                                w_pattern_sum += w;
                                w_pattern_count += 1;
                            } else {
                                w_noise_sum += w;
                                w_noise_count += 1;
                            }
                        }
                    }
                }
            }

            let avg_w_pattern = if w_pattern_count > 0 {
                w_pattern_sum / w_pattern_count as f64
            } else {
                0.0
            };
            let avg_w_noise = if w_noise_count > 0 {
                w_noise_sum / w_noise_count as f64
            } else {
                0.0
            };

            writeln!(
                file,
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                step, phase_name, fr, max_fr_seen, avg_thresh, avg_w_pattern, avg_w_noise
            )
            .unwrap();

            if step % 5000 == 0 {
                println!(
                    "Step {:5} [{}] FR={:.3} (max {:.3}) | Th={:.3} | W_Patt={:.3} vs W_Noise={:.3}",
                    step, phase_name, fr, max_fr_seen, avg_thresh, avg_w_pattern, avg_w_noise
                );
            }

            if fr > 0.8 || max_fr_seen > 1.0 {
                println!(
                    "❌ FALHA CRÍTICA: Runaway Excitation detectada (FR={:.2}, max {:.2}) no step {}",
                    fr, max_fr_seen, step
                );
                writeln!(
                    file,
                    "{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    step, "FAIL", fr, max_fr_seen, avg_thresh, avg_w_pattern, avg_w_noise
                )
                .unwrap();
                return;
            }
        }
    }

    println!("\nExperimento concluído. Verifique 'learning_log.csv' para plotar os dados.");
}
