//! Exemplo básico de uso da rede NEN-V
//!
//! Este exemplo demonstra:
//! - Criação de rede via AutoConfig
//! - Simulação básica
//! - Monitoramento de métricas

use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};

fn main() {
    println!("╔═════════════════════════════════════════╗");
    println!("║       NEN-V v2.0 - Exemplo Básico       ║");
    println!("╚═════════════════════════════════════════╝\n");

    // Define tarefa
    let task = TaskSpec {
        num_sensors: 8,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,
            temporal_horizon: Some(100),
        },
    };

    // Cria configuração automática
    let config = AutoConfig::from_task(task);
    config.print_report();

    // Constrói a rede
    let mut network = match config.build_network() {
        Ok(net) => net,
        Err(e) => {
            eprintln!("Erro ao criar rede: {}", e);
            return;
        }
    };

    println!("\n┌─────────────────────────────────────────┐");
    println!("│ INICIANDO SIMULAÇÃO                     │");
    println!("└─────────────────────────────────────────┘\n");

    // Simula 1000 passos
    let num_steps = 1000;
    let num_neurons = network.num_neurons();
    
    for step in 0..num_steps {
        // Gera inputs (padrão simples)
        let inputs: Vec<f64> = (0..num_neurons)
            .map(|i| {
                if i < 8 {
                    // Sensores recebem input variável
                    ((step as f64 / 50.0 + i as f64).sin() + 1.0) / 2.0
                } else {
                    0.0
                }
            })
            .collect();

        // Atualiza rede
        network.update(&inputs);

        // Mostra estatísticas a cada 100 passos
        if step % 100 == 0 {
            let stats = network.get_stats();
            println!(
                "Step {:>4}: FR={:>5.1}% | Energy={:>5.1}% | Threshold={:.3} | DA={:.2}",
                step,
                stats.firing_rate * 100.0,
                stats.avg_energy,
                stats.avg_threshold,
                stats.dopamine
            );
        }

        // Simula reward esporádico
        if step % 200 == 100 {
            network.propagate_reward(1.0);
        }
    }

    // Estatísticas finais
    let final_stats = network.get_stats();
    println!("\n┌─────────────────────────────────────────┐");
    println!("│ ESTATÍSTICAS FINAIS                     │");
    println!("├─────────────────────────────────────────┤");
    println!("│ Tempo total:        {:>18} │", final_stats.time_step);
    println!("│ Firing rate:        {:>17.2}% │", final_stats.firing_rate * 100.0);
    println!("│ Energia média:      {:>17.1}% │", final_stats.avg_energy);
    println!("│ Threshold médio:    {:>18.3} │", final_stats.avg_threshold);
    println!("│ Eligibility médio:  {:>18.4} │", final_stats.avg_eligibility);
    println!("└─────────────────────────────────────────┘");

    println!("\n✓ Simulação concluída com sucesso!");
}
