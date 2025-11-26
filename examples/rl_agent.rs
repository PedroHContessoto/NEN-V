//! Exemplo de Agente de Reinforcement Learning com NEN-V
//!
//! Este exemplo demonstra:
//! - Configuração para RL
//! - Integração com Working Memory
//! - Propagação de reward
//! - Adaptação em tempo real

use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity, AdaptiveState};
use nenv_v2::working_memory::WorkingMemoryPool;
use nenv_v2::neuromodulation::NeuromodulatorType;

fn main() {
    println!("╔═════════════════════════════════════════╗");
    println!("║     NEN-V v2.0 - Agente RL              ║");
    println!("╚═════════════════════════════════════════╝\n");

    // Configuração para RL com reward esparso
    let task = TaskSpec {
        num_sensors: 16,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Sparse,
            temporal_horizon: Some(200),
        },
    };

    let config = AutoConfig::from_task(task);
    println!("Configuração criada:");
    println!("  - Neurônios: {}", config.architecture.total_neurons);
    println!("  - Eligibility tau: {:.1}", config.params.eligibility.trace_tau);
    println!("  - Curiosidade: {}", if config.params.curiosity.enabled { "ativada" } else { "desativada" });
    println!();

    let mut network = config.build_network().expect("Configuração válida");
    
    // Working Memory para manter contexto
    let wm_pattern_size = network.num_neurons();
    let mut working_memory = WorkingMemoryPool::new(7, wm_pattern_size);
    
    // Estado adaptativo para auto-regulação
    let mut adaptive_state = AdaptiveState::new(config.clone());

    println!("┌─────────────────────────────────────────┐");
    println!("│ TREINAMENTO DO AGENTE                   │");
    println!("└─────────────────────────────────────────┘\n");

    let num_episodes = 10;
    let steps_per_episode = 500;
    let mut total_reward = 0.0;
    let mut episode_rewards = Vec::new();

    for episode in 0..num_episodes {
        let mut episode_reward = 0.0;

        for step in 0..steps_per_episode {
            // Estado do ambiente (simplificado)
            let env_state: Vec<f64> = (0..16)
                .map(|i| {
                    let phase = (step as f64 / 20.0 + i as f64 * 0.5).sin();
                    (phase + 1.0) / 2.0
                })
                .collect();

            // Prepara inputs (estado + contexto da WM)
            let mut inputs = vec![0.0; network.num_neurons()];
            for (i, &s) in env_state.iter().enumerate() {
                if i < inputs.len() {
                    inputs[i] = s;
                }
            }

            // Adiciona contexto da Working Memory
            let wm_context = working_memory.get_combined_representation();
            for (i, &ctx) in wm_context.iter().enumerate() {
                let target_idx = 16 + i;
                if target_idx < inputs.len() {
                    inputs[target_idx] += ctx * 0.3;
                }
            }

            // Atualiza rede
            network.update(&inputs);

            // Armazena padrão de atividade na WM
            let activity: Vec<f64> = network.neurons.iter()
                .map(|n| if n.is_firing { 1.0 } else { 0.0 })
                .collect();
            working_memory.encode(activity, (episode * steps_per_episode + step) as i64);
            working_memory.sustain();

            // Simula recompensa do ambiente
            let reward = if step % 100 == 50 && rand::random::<f64>() > 0.3 {
                1.0
            } else {
                0.0
            };

            if reward > 0.0 {
                // Propaga reward com eligibility traces
                network.propagate_reward(reward);
                episode_reward += reward;
                
                // Boost de dopamina
                network.neuromodulation.release(NeuromodulatorType::Dopamine, 0.5);
            }

            // Adaptação em tempo real (a cada 50 passos)
            if step % 50 == 0 {
                adaptive_state.monitor_and_adapt(&mut network);
            }
        }

        episode_rewards.push(episode_reward);
        total_reward += episode_reward;

        let stats = network.get_stats();
        let wm_stats = working_memory.get_stats();
        
        println!(
            "Episódio {:>2}: Reward={:>5.1} | FR={:>5.1}% | WM slots={}/{} | DA={:.2}",
            episode + 1,
            episode_reward,
            stats.firing_rate * 100.0,
            wm_stats.active_slots,
            wm_stats.max_capacity,
            stats.dopamine
        );
    }

    // Análise final
    println!("\n┌─────────────────────────────────────────┐");
    println!("│ RESULTADOS DO TREINAMENTO               │");
    println!("├─────────────────────────────────────────┤");
    println!("│ Reward total:       {:>18.1} │", total_reward);
    println!("│ Reward médio/ep:    {:>18.2} │", total_reward / num_episodes as f64);
    
    // Tendência de melhoria
    let first_half: f64 = episode_rewards.iter().take(5).sum();
    let second_half: f64 = episode_rewards.iter().skip(5).sum();
    let improvement = if first_half > 0.0 {
        ((second_half - first_half) / first_half) * 100.0
    } else {
        0.0
    };
    
    println!("│ Melhoria 2a metade: {:>+17.1}% │", improvement);
    
    let adaptive_stats = adaptive_state.get_stats();
    println!("│ Problemas detectados: {:>16} │", adaptive_stats.issues_detected);
    println!("│ Correções aplicadas:  {:>16} │", adaptive_stats.actions_taken);
    println!("└─────────────────────────────────────────┘");

    println!("\n✓ Treinamento concluído!");
}
