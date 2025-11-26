//! Exemplo de Exploração Autônoma com Curiosidade Intrínseca
//!
//! Este exemplo demonstra:
//! - Módulo de curiosidade para recompensa intrínseca
//! - Codificação preditiva para modelar o ambiente
//! - Exploração guiada por surpresa
//! - Integração completa dos componentes cognitivos

use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};
use nenv_v2::intrinsic_motivation::{CuriosityModule, RandomNetworkDistillation};
use nenv_v2::predictive::PredictiveHierarchy;
use nenv_v2::working_memory::WorkingMemoryPool;

fn main() {
    println!("╔═════════════════════════════════════════╗");
    println!("║  NEN-V v2.0 - Curiosidade Intrínseca    ║");
    println!("╚═════════════════════════════════════════╝\n");

    // Configuração para exploração autônoma
    let task = TaskSpec {
        num_sensors: 8,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Sparse, // Sem reward externo frequente
            temporal_horizon: Some(100),
        },
    };

    let config = AutoConfig::from_task(task);
    let mut network = config.build_network().expect("Configuração válida");

    // Componentes cognitivos
    let state_size = 8;
    let action_size = 4;
    
    // Módulo de Curiosidade (ICM-style)
    let mut curiosity = CuriosityModule::new(state_size, action_size);
    curiosity.curiosity_scale = 0.12; // Escala intermediária da recompensa intrínseca
    curiosity.surprise_threshold = 0.03; // Evita reforçar surpresas muito fracas
    curiosity.habituation_rate = 0.997; // Habitua devagar para manter curiosidade
    
    // Random Network Distillation (alternativa)
    let mut rnd = RandomNetworkDistillation::new(state_size, 16);
    
    // Hierarquia Preditiva
    let mut predictor = PredictiveHierarchy::new_three_level(state_size);
    
    // Working Memory
    let mut working_memory = WorkingMemoryPool::new(7, state_size);

    println!("Configuração:");
    println!("  - Estado: {} dimensões", state_size);
    println!("  - Ações: {} dimensões", action_size);
    println!("  - WM capacidade: {} slots", working_memory.max_capacity);
    println!();

    println!("┌─────────────────────────────────────────┐");
    println!("│ EXPLORAÇÃO AUTÔNOMA                     │");
    println!("└─────────────────────────────────────────┘\n");

    let num_steps = 2000;
    let mut total_intrinsic_reward = 0.0;
    let mut total_rnd_reward = 0.0;
    let mut total_prediction_error = 0.0;

    // Estado inicial do "ambiente"
    let mut current_state: Vec<f64> = (0..state_size)
        .map(|i| (i as f64 / state_size as f64))
        .collect();

    for step in 0..num_steps {
        // Seleciona ação (exploração guiada por curiosidade)
        let action: Vec<f64> = select_curious_action(
            &current_state,
            &network,
            action_size,
            step,
        );

        // Simula transição do ambiente
        let next_state = simulate_environment(&current_state, &action, step);

        // === CURIOSIDADE ICM ===
        let icm_reward = curiosity.compute_intrinsic_reward(
            &current_state,
            &action,
            &next_state,
        );
        total_intrinsic_reward += icm_reward;

        // === RANDOM NETWORK DISTILLATION ===
        let rnd_reward = rnd.compute_reward(&next_state);
        total_rnd_reward += rnd_reward;

        // === CODIFICAÇÃO PREDITIVA ===
        let pred_output = predictor.process(&next_state);
        total_prediction_error += pred_output.free_energy;

        // === WORKING MEMORY ===
        working_memory.encode(next_state.clone(), step as i64);
        working_memory.sustain();

        // Combina recompensas intrínsecas priorizando ICM e reduzindo RND
        let combined_reward = icm_reward * 0.8 + rnd_reward * 0.15;

        // Propaga para a rede se reward significativo
        if combined_reward > 0.05 {
            network.propagate_reward(combined_reward);
        }

        // Atualiza rede com o estado
        let mut inputs = vec![0.0; network.num_neurons()];
        for (i, &s) in next_state.iter().enumerate() {
            if i < inputs.len() {
                inputs[i] = s;
            }
        }
        network.update(&inputs);

        // Logs periódicos
        if step % 200 == 0 {
            let curiosity_stats = curiosity.get_stats();
            let pred_stats = predictor.get_stats();
            let wm_stats = working_memory.get_stats();

            println!(
                "Step {:>4}: ICM={:.3} | RND={:.3} | FE={:.3} | WM={}/{} | Prog={:+.1}%",
                step,
                icm_reward,
                rnd_reward,
                pred_output.free_energy,
                wm_stats.active_slots,
                wm_stats.max_capacity,
                curiosity_stats.learning_progress * 100.0
            );
        }

        // Avança estado
        current_state = next_state;
    }

    // Estatísticas finais
    let final_curiosity_stats = curiosity.get_stats();
    let final_pred_stats = predictor.get_stats();
    let final_wm_stats = working_memory.get_stats();
    let final_network_stats = network.get_stats();

    println!("\n┌─────────────────────────────────────────┐");
    println!("│ RESULTADOS DA EXPLORAÇÃO                │");
    println!("├─────────────────────────────────────────┤");
    println!("│ Reward ICM total:   {:>18.2} │", total_intrinsic_reward);
    println!("│ Reward RND total:   {:>18.2} │", total_rnd_reward);
    println!("│ Free Energy média:  {:>18.4} │", total_prediction_error / num_steps as f64);
    println!("├─────────────────────────────────────────┤");
    println!("│ CURIOSIDADE                             │");
    println!("│   Experiências:     {:>18} │", final_curiosity_stats.experience_count);
    println!("│   Erro predição:    {:>18.4} │", final_curiosity_stats.avg_prediction_error);
    println!("│   Progresso:        {:>+17.1}% │", final_curiosity_stats.learning_progress * 100.0);
    println!("├─────────────────────────────────────────┤");
    println!("│ HIERARQUIA PREDITIVA                    │");
    println!("│   Níveis:           {:>18} │", final_pred_stats.num_levels);
    println!("│   Free Energy:      {:>18.4} │", final_pred_stats.total_free_energy);
    println!("│   Precisão média:   {:>18.4} │", final_pred_stats.average_precision);
    println!("├─────────────────────────────────────────┤");
    println!("│ WORKING MEMORY                          │");
    println!("│   Slots ativos:     {:>18} │", final_wm_stats.active_slots);
    println!("│   Utilização:       {:>17.1}% │", final_wm_stats.utilization * 100.0);
    println!("│   Operações:        {:>18} │", final_wm_stats.operation_count);
    println!("├─────────────────────────────────────────┤");
    println!("│ REDE NEURAL                             │");
    println!("│   Firing rate:      {:>17.1}% │", final_network_stats.firing_rate * 100.0);
    println!("│   Energia média:    {:>17.1}% │", final_network_stats.avg_energy);
    println!("│   Eligibility:      {:>18.4} │", final_network_stats.avg_eligibility);
    println!("└─────────────────────────────────────────┘");

    // Análise da exploração
    println!("\n┌─────────────────────────────────────────┐");
    println!("│ ANÁLISE DA EXPLORAÇÃO                   │");
    println!("├─────────────────────────────────────────┤");
    
    let exploration_efficiency = if final_curiosity_stats.experience_count > 0 {
        total_intrinsic_reward / final_curiosity_stats.experience_count as f64
    } else {
        0.0
    };
    
    println!("│ Eficiência:         {:>18.4} │", exploration_efficiency);
    
    let health = final_curiosity_stats.health;
    let health_str = match health {
        nenv_v2::intrinsic_motivation::CuriosityHealth::Healthy => "Saudável",
        nenv_v2::intrinsic_motivation::CuriosityHealth::TooLow => "Muito Baixa",
        nenv_v2::intrinsic_motivation::CuriosityHealth::TooHigh => "Muito Alta",
    };
    println!("│ Saúde curiosidade:  {:>18} │", health_str);
    
    let model_quality = if final_curiosity_stats.avg_prediction_error > 0.5 {
        "Precisa melhorar"
    } else if final_curiosity_stats.avg_prediction_error > 0.2 {
        "Razoável"
    } else {
        "Bom"
    };
    println!("│ Qualidade modelo:   {:>18} │", model_quality);
    println!("└─────────────────────────────────────────┘");

    println!("\n✓ Exploração autônoma concluída!");
}

/// Seleciona ação baseada em curiosidade (epsilon-curiosity)
fn select_curious_action(
    _state: &[f64],
    _network: &nenv_v2::network::Network,
    action_size: usize,
    _step: usize,
) -> Vec<f64> {
    // Estratégia simples: ação aleatória com bias
    let mut action = vec![0.0; action_size];
    let selected = rand::random::<usize>() % action_size;
    action[selected] = 1.0;
    action
}

/// Simula transição do ambiente (ambiente sintético)
fn simulate_environment(state: &[f64], action: &[f64], step: usize) -> Vec<f64> {
    let mut next_state = state.to_vec();
    
    // Dinâmica não-linear simples
    for (i, val) in next_state.iter_mut().enumerate() {
        let action_effect = action.get(i % action.len()).unwrap_or(&0.0);
        let noise = (rand::random::<f64>() - 0.5) * 0.1;
        let temporal = (step as f64 / 100.0 + i as f64 * 0.3).sin() * 0.1;
        
        *val = (*val * 0.9 + action_effect * 0.3 + temporal + noise).clamp(0.0, 1.0);
    }
    
    next_state
}
