//! EXPERIMENTO: Weight Decay vs STDP Learning
//!
//! Este experimento demonstra o problema do weight decay duplo
//! quando STDP está ativo, usando a rede completa com AutoConfig.

use nenv_v2::autoconfig::{AutoConfig, RewardDensity, TaskSpec, TaskType};
use nenv_v2::network::LearningMode;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENTO: Weight Decay vs STDP Learning                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Objetivo: Demonstrar que weight decay domina sobre STDP            ║");
    println!("║            em redes com atividade esparsa                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Configuração da tarefa
    let task = TaskSpec {
        num_sensors: 10,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Moderate,
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task.clone());

    println!("Configuração da Rede:");
    println!("  Sensores: {}", task.num_sensors);
    println!("  Actuators: {}", task.num_actuators);
    println!("  Target FR: {:.4}", config.params.target_firing_rate);
    println!("  Weight decay (dendritoma): 0.0001");
    println!("  STDP a_plus: 0.015");
    println!("  STDP a_minus: 0.006\n");

    println!("═════════════════════════════════════════════════════════════════════");
    println!("  EXPERIMENTO A: Rede com STDP (input aleatório 10%)");
    println!("═════════════════════════════════════════════════════════════════════\n");

    // Cria rede com STDP
    let mut network_stdp = config.build_network().expect("Falha ao construir rede");
    network_stdp.set_learning_mode(LearningMode::STDP);

    let total_neurons = network_stdp.num_neurons();
    let sensor_indices = config.architecture.sensor_indices.clone();
    let hidden_indices = config.architecture.hidden_indices.clone();

    println!("Rede criada: {} neurônios ({} sensores, {} hidden, {} actuators)\n",
        total_neurons,
        sensor_indices.len(),
        hidden_indices.len(),
        config.architecture.actuator_indices.len());

    // Foca em um neurônio hidden específico para tracking
    let tracked_neuron_idx = hidden_indices.start;
    let neuron_tracked = &network_stdp.neurons[tracked_neuron_idx];
    let initial_weight_sum = neuron_tracked.dendritoma.total_weight();
    let initial_weight_avg = initial_weight_sum / neuron_tracked.dendritoma.weights.len() as f64;

    println!("Neurônio rastreado: #{} (hidden layer)", tracked_neuron_idx);
    println!("  Pesos iniciais:");
    println!("    Soma total: {:.6}", initial_weight_sum);
    println!("    Média: {:.6}", initial_weight_avg);
    println!("    Threshold: {:.6}\n", neuron_tracked.threshold);

    let mut rng = StdRng::seed_from_u64(42);
    let steps = 10_000;

    println!("Rodando {} steps com input aleatório (10% dos sensores ativos)...\n", steps);

    // Métricas
    let mut total_stdp_events = 0;
    let mut total_fires = 0;

    for step in 0..steps {
        // Input aleatório esparso
        let mut inputs = vec![0.0; total_neurons];
        for idx in sensor_indices.clone() {
            if rng.gen_bool(0.1) {
                inputs[idx] = 1.0;
            }
        }

        network_stdp.update(&inputs);

        // Rastreia atividade do neurônio
        let neuron = &network_stdp.neurons[tracked_neuron_idx];
        if neuron.is_firing {
            total_fires += 1;
        }

        // Log periódico
        if (step + 1) % 1000 == 0 {
            let neuron = &network_stdp.neurons[tracked_neuron_idx];
            let current_weight_sum = neuron.dendritoma.total_weight();
            let current_weight_avg = current_weight_sum / neuron.dendritoma.weights.len() as f64;
            let weight_change_pct = ((current_weight_sum - initial_weight_sum) / initial_weight_sum) * 100.0;

            let fr = network_stdp.num_firing() as f64 / total_neurons as f64;

            // Calcula número de eventos STDP (aproximação)
            let trace_sum: f64 = neuron.dendritoma.eligibility_trace.iter().sum();

            println!("  Step {:5}: FR={:.3} | Weight_sum={:.4} ({:+.2}%) | Avg={:.6} | Trace={:.3} | Fires={}",
                step + 1,
                fr,
                current_weight_sum,
                weight_change_pct,
                current_weight_avg,
                trace_sum,
                total_fires);
        }
    }

    // Análise final
    println!("\n─────────────────────────────────────────────────────────────────────");
    println!("  RESULTADOS DO EXPERIMENTO A (STDP ativo)");
    println!("─────────────────────────────────────────────────────────────────────\n");

    let neuron = &network_stdp.neurons[tracked_neuron_idx];
    let final_weight_sum = neuron.dendritoma.total_weight();
    let final_weight_avg = final_weight_sum / neuron.dendritoma.weights.len() as f64;
    let total_weight_change = final_weight_sum - initial_weight_sum;
    let weight_change_pct = (total_weight_change / initial_weight_sum) * 100.0;

    println!("Neurônio #{}", tracked_neuron_idx);
    println!("  Pesos iniciais: soma={:.6}, média={:.6}", initial_weight_sum, initial_weight_avg);
    println!("  Pesos finais:   soma={:.6}, média={:.6}", final_weight_sum, final_weight_avg);
    println!("  Mudança:        {:+.6} ({:+.2}%)", total_weight_change, weight_change_pct);
    println!("  Disparos totais: {} ({:.2}% dos steps)", total_fires, (total_fires as f64 / steps as f64) * 100.0);
    println!("  Threshold final: {:.6}", neuron.threshold);

    let final_fr = network_stdp.num_firing() as f64 / total_neurons as f64;
    println!("\nRede completa:");
    println!("  FR final: {:.3}", final_fr);
    println!("  Target FR: {:.3}", config.params.target_firing_rate);

    println!("\n═════════════════════════════════════════════════════════════════════");
    println!("  EXPERIMENTO B: Rede com Hebbian (sem STDP, para comparação)");
    println!("═════════════════════════════════════════════════════════════════════\n");

    // Cria nova rede com Hebbian (sem STDP) para comparação
    let mut network_no_learning = config.build_network().expect("Falha ao construir rede");
    network_no_learning.set_learning_mode(LearningMode::Hebbian);

    let neuron_b = &network_no_learning.neurons[tracked_neuron_idx];
    let initial_weight_sum_b = neuron_b.dendritoma.total_weight();

    println!("Rodando {} steps SEM aprendizado (apenas decay)...\n", steps);

    let mut rng2 = StdRng::seed_from_u64(42); // mesma seed para inputs idênticos

    for step in 0..steps {
        let mut inputs = vec![0.0; total_neurons];
        for idx in sensor_indices.clone() {
            if rng2.gen_bool(0.1) {
                inputs[idx] = 1.0;
            }
        }

        network_no_learning.update(&inputs);

        if (step + 1) % 1000 == 0 {
            let neuron = &network_no_learning.neurons[tracked_neuron_idx];
            let current_weight_sum = neuron.dendritoma.total_weight();
            let weight_change_pct = ((current_weight_sum - initial_weight_sum_b) / initial_weight_sum_b) * 100.0;
            let fr = network_no_learning.num_firing() as f64 / total_neurons as f64;

            println!("  Step {:5}: FR={:.3} | Weight_sum={:.4} ({:+.2}%)",
                step + 1, fr, current_weight_sum, weight_change_pct);
        }
    }

    println!("\n─────────────────────────────────────────────────────────────────────");
    println!("  RESULTADOS DO EXPERIMENTO B (Hebbian, sem STDP)");
    println!("─────────────────────────────────────────────────────────────────────\n");

    let neuron_b = &network_no_learning.neurons[tracked_neuron_idx];
    let final_weight_sum_b = neuron_b.dendritoma.total_weight();
    let total_weight_change_b = final_weight_sum_b - initial_weight_sum_b;
    let weight_change_pct_b = (total_weight_change_b / initial_weight_sum_b) * 100.0;

    println!("Neurônio #{}", tracked_neuron_idx);
    println!("  Pesos iniciais: {:.6}", initial_weight_sum_b);
    println!("  Pesos finais:   {:.6}", final_weight_sum_b);
    println!("  Mudança:        {:+.6} ({:+.2}%)", total_weight_change_b, weight_change_pct_b);

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                          CONCLUSÕES                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    println!("Comparação de weight decay:");
    println!("  COM STDP:  {:+.2}% ({:+.6})", weight_change_pct, total_weight_change);
    println!("  SEM STDP:  {:+.2}% ({:+.6})", weight_change_pct_b, total_weight_change_b);

    if weight_change_pct < 0.0 && weight_change_pct_b < 0.0 {
        println!("\n⚠️  PROBLEMA DETECTADO:");
        println!("  Ambas as redes tiveram decay de pesos!");

        if weight_change_pct.abs() > weight_change_pct_b.abs() {
            println!("  Rede COM STDP teve MAIS decay que sem STDP!");
            println!("  Isso indica que STDP não está compensando o weight decay.");
        } else if weight_change_pct.abs() < weight_change_pct_b.abs() {
            println!("  STDP ajudou parcialmente, mas não foi suficiente.");
        }

        println!("\n  Causas prováveis:");
        println!("  1. Weight decay aplicado DUAS vezes em apply_stdp_learning:");
        println!("     - Uma vez em apply_stdp_pair (linha 471-472)");
        println!("     - Outra vez em apply_stdp_learning (linha 503)");
        println!("  2. Atividade esparsa (FR={:.3}) limita eventos STDP", final_fr);
        println!("  3. Weight decay SEM proteção de atividade durante STDP");
    } else if weight_change_pct > 0.0 {
        println!("\n✅ STDP está funcionando: pesos aumentaram!");
    }

    println!("\n  Decay esperado teoricamente:");
    println!("    Por step: 0.0001 (0.01%)");
    println!("    Em {} steps: ~{:.2}% (sem aprendizado)", steps, (1.0 - (1.0 - 0.0001f64).powi(steps)) * 100.0);
    println!("\n  Decay observado:");
    println!("    COM STDP: {:.2}%", weight_change_pct.abs());
    println!("    SEM STDP: {:.2}%", weight_change_pct_b.abs());

    if weight_change_pct_b.abs() > 1.0 {
        println!("\n⚠️  ALERTA: Decay observado ({:.2}%) é muito maior que o esperado (~1.0%)",
            weight_change_pct_b.abs());
        println!("  Isso confirma que weight decay está sendo aplicado múltiplas vezes!");
    }
}
