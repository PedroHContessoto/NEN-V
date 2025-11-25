mod dendritoma;
mod glia;
mod nenv;
mod network;
mod environment;
mod visuals;

use macroquad::prelude::*;
use network::{ConnectivityType, LearningMode, Network};
use ::rand::Rng;

// Importa nossos módulos novos
use environment::{Environment, ActionResult, Metrics};
use visuals::{compute_neuron_positions, draw_grid_environment, draw_neural_network, draw_metrics, draw_legend_panel};

// === CONFIGURAÇÃO DE HARDWARE DA REDE ===
const NUM_NEURONS: usize = 20;

const SENSOR_UP: usize = 0;
const SENSOR_DOWN: usize = 1;
const SENSOR_LEFT: usize = 2;
const SENSOR_RIGHT: usize = 3;
const MOTOR_UP: usize = 16;
const MOTOR_DOWN: usize = 17;
const MOTOR_LEFT: usize = 18;
const MOTOR_RIGHT: usize = 19;

// === CONFIGURAÇÃO DE CICLOS DE SONO ===
const SLEEP_INTERVAL: u64 = 3000;      // Dorme a cada 3000 steps (mais frequente)
const SLEEP_DURATION: usize = 500;     // Duração do sono: 500 steps (mais rápido)
const SLEEP_REPLAY_NOISE: f64 = 0.05;   // Probabilidade de replay espontâneo
const MIN_SELECTIVITY_TO_SLEEP: f64 = 0.03; // Dorme se tiver algum aprendizado (seletividade > 0.03)
const MIN_SUCCESSES_TO_SLEEP: u32 = 3;      // Precisa ter comido pelo menos 3 vezes

// Função para calcular seletividade média da rede
fn compute_average_selectivity(net: &Network) -> f64 {
    let pairs = [
        (SENSOR_UP, MOTOR_UP),
        (SENSOR_DOWN, MOTOR_DOWN),
        (SENSOR_LEFT, MOTOR_LEFT),
        (SENSOR_RIGHT, MOTOR_RIGHT),
    ];

    let mut selectivities = Vec::new();

    for (sensor, motor) in pairs {
        let w_stm = net.neurons[motor].dendritoma.weights[sensor];
        let w_ltm = net.neurons[motor].dendritoma.weights_ltm[sensor];
        let correct_weight = w_stm + w_ltm;

        // Calcula peso médio das conexões incorretas
        let mut incorrect_weights = 0.0;
        let mut count = 0;
        for (other_s, _) in &pairs {
            if *other_s != sensor {
                incorrect_weights += net.neurons[motor].dendritoma.weights[*other_s]
                                   + net.neurons[motor].dendritoma.weights_ltm[*other_s];
                count += 1;
            }
        }
        let avg_incorrect = incorrect_weights / count as f64;

        // Selectivity Index: quanto o peso correto é maior que o ruído
        let selectivity = if correct_weight > 0.01 {
            ((correct_weight - avg_incorrect) / correct_weight).max(0.0)
        } else {
            0.0
        };

        selectivities.push(selectivity);
    }

    selectivities.iter().sum::<f64>() / selectivities.len() as f64
}

// Função auxiliar de diagnóstico (Imprime no terminal)
fn diagnose_selectivity(net: &Network) {
    let pairs = [
        (SENSOR_UP, MOTOR_UP, "UP"),
        (SENSOR_DOWN, MOTOR_DOWN, "DOWN"),
        (SENSOR_LEFT, MOTOR_LEFT, "LEFT"),
        (SENSOR_RIGHT, MOTOR_RIGHT, "RIGHT"),
    ];

    println!("\n--- Diagnóstico de Seletividade (Pesos) ---");
    for (sensor, motor, name) in pairs {
        let w_stm = net.neurons[motor].dendritoma.weights[sensor];
        let w_ltm = net.neurons[motor].dendritoma.weights_ltm[sensor];
        let total = w_stm + w_ltm;

        // Calcula ruído (conexões erradas)
        let mut noise = 0.0;
        let mut count = 0;
        for (other_s, _, _) in &pairs {
            if *other_s != sensor {
                noise += net.neurons[motor].dendritoma.weights[*other_s];
                count += 1;
            }
        }
        let avg_noise = noise / count as f64;

        // Ratio > 1.0 indica que aprendeu a conexão correta
        let ratio = if avg_noise > 0.001 { total / avg_noise } else { 0.0 };

        println!("{}: Peso={:.3} | Ruído={:.3} | Selectividade={:.2}",
            name, total, avg_noise, ratio);
    }

    let avg_sel = compute_average_selectivity(net);
    println!("Seletividade Média: {:.3}", avg_sel);
    println!("-------------------------------------------");
}

fn window_conf() -> macroquad::window::Conf {
    macroquad::window::Conf {
        window_title: "NENV GridWorld - Rede Neural Biológica".to_owned(),
        window_width: 1280,
        window_height: 720,
        window_resizable: false,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // 1. INICIALIZAÇÃO DA REDE (Parâmetros Biológicos)
    let mut net = Network::new(
        NUM_NEURONS,
        ConnectivityType::FullyConnected,
        0.2,  // 20% inibitórios (equilíbrio E/I)
        0.15, // Threshold de disparo
    );

    net.set_learning_mode(LearningMode::STDP);
    net.set_weight_decay(0.002);  // Decay moderado - meio termo entre preservar e esquecer

    // 2. INICIALIZAÇÃO DO AMBIENTE
    let mut env = Environment::new(5); // Grid 5x5 inicial
    let mut metrics = Metrics::new();
    let mut rng = ::rand::thread_rng();
    
    // Dados para Curriculum Learning (aumentar dificuldade)
    let mut food_count_level = 0;
    const FOODS_PER_LEVEL: u32 = 10;

    // Controle de Ciclos de Sono
    let mut last_sleep_step = 0u64;

    // Timeout para episódios (considera falha se demorar muito)
    const MAX_STEPS_PER_EPISODE: u32 = 200;

    // Prepara visualização (posições fixas dos neurônios)
    let neuron_positions = compute_neuron_positions(NUM_NEURONS);

    // Listas para passar para o visualizador
    let sensors_idx = [SENSOR_UP, SENSOR_DOWN, SENSOR_LEFT, SENSOR_RIGHT];
    let motors_idx = [MOTOR_UP, MOTOR_DOWN, MOTOR_LEFT, MOTOR_RIGHT];

    // === LOOP PRINCIPAL ===
    loop {
        // Limpa a tela
        clear_background(BLACK);

        // --- FASE 1: PERCEPÇÃO ---
        let sensors = env.get_sensor_inputs();
        let mut inputs = vec![0.0; NUM_NEURONS];
        
        // Injeta sinal nos neurônios sensoriais
        inputs[SENSOR_UP] = sensors[0];
        inputs[SENSOR_DOWN] = sensors[1];
        inputs[SENSOR_LEFT] = sensors[2];
        inputs[SENSOR_RIGHT] = sensors[3];

        // --- FASE 2: EXPLORAÇÃO INTELIGENTE (Estratégia B) ---
        // Se tiver poucos sucessos, explora muito (20%). Se já sabe jogar, explora pouco (2%).
        let exploration_rate = if metrics.successes < 15 { 0.20 } 
            else if metrics.successes < 40 { 0.10 } 
            else { 0.02 };

        // Aplica "impulso" aleatório em um motor para descobrir novas ações
        if rng.gen_bool(exploration_rate) {
            let random_motor = motors_idx[rng.gen_range(0..4)];
            inputs[random_motor] += 1.5; // Bootstrap forte
        }

        // --- FASE 3: PROCESSAMENTO ---
        net.update(&inputs);

        // --- FASE 4: DECISÃO (Winner-Takes-All) ---
        let mut best_motor = None;
        let mut max_potential = -100.0; // Valor muito baixo inicial

        for &motor_id in &motors_idx {
            let neuron = &net.neurons[motor_id];
            
            // Calculamos o potencial "vontade de disparar"
            let integrated = neuron.dendritoma.integrate(&inputs);
            let potential = neuron.glia.modulate(integrated);
            
            // Só considera motores que estão ativos ou quase disparando
            if neuron.is_firing || potential > 0.3 { 
                if potential > max_potential {
                    max_potential = potential;
                    best_motor = Some(motor_id);
                }
            }
        }

        // --- FASE 5: AÇÃO E CONSEQUÊNCIA ---
        let mut action_result = ActionResult::None;
        
        if let Some(motor_id) = best_motor {
            // Executa movimento
            action_result = env.execute_motor(motor_id, &motors_idx);

            // Registra estatística
            metrics.total_movements += 1;
            if let Some(idx) = motors_idx.iter().position(|&m| m == motor_id) {
                metrics.motor_fires[idx] += 1;
            }
        }

        // --- FASE 6: FEEDBACK (Dopamina/Energia) ---
        match action_result {
            ActionResult::AteFood => {
                // 1. Dopamina: Reforça conexões que levaram a isso (LTP)
                net.global_reward_signal = 1.0; 
                
                // 2. Metabolismo: RECARREGA ENERGIA (Sobreviveu!)
                for n in &mut net.neurons { 
                    n.glia.energy = n.glia.max_energy; 
                }
                
                // Lógica do jogo
                metrics.record_success(env.steps_current_episode);
                env.steps_current_episode = 0;
                env.respawn_food();
                
                // Curriculum Learning (aumenta grid se estiver fácil)
                food_count_level += 1;
                if food_count_level >= FOODS_PER_LEVEL {
                    env.expand_grid();
                    food_count_level = 0;
                }
            },
            ActionResult::HitWall => {
                // Punição moderada: Enfraquece conexões erradas sem destruir o aprendizado
                net.global_reward_signal = -1.0;
                metrics.record_wall_hit();
            },
            ActionResult::Moved => {
                // Movimento neutro (apenas custo metabólico natural)
                net.global_reward_signal = 0.0;
            },
            ActionResult::None => {
                 net.global_reward_signal = 0.0;
            }
        }

        metrics.total_steps += 1;
        env.steps_current_episode += 1;

        // --- FASE 7: SOBREVIVÊNCIA (MORTE REAL) ---
        // Sem Fail-Safe. Se energia acabar, reseta o corpo (mantém cérebro/pesos).
        if net.average_energy() < 5.0 {
            // Punição severa ao morrer (para evitar suicídio energético)
            net.global_reward_signal = -1.0;
            
            // Reseta posição do agente e comida
            env.agent.x = env.grid_size / 2;
            env.agent.y = env.grid_size / 2;
            env.respawn_food();
            env.steps_current_episode = 0;

            // "Reencarna": Recupera energia para tentar de novo
            for n in &mut net.neurons { 
                n.glia.energy = n.glia.max_energy; 
            }
            
            // Opcional: imprimir aviso de morte
            // println!("Agente morreu de fome no passo {}", metrics.total_steps);
        }

        // --- DIAGNÓSTICO PERIÓDICO ---
        if metrics.total_steps % 1000 == 0 {
            println!("Steps: {} | Score: {} | Energy: {:.1}%",
                metrics.total_steps, metrics.score, net.average_energy());
            diagnose_selectivity(&net);
        }

        // --- CICLOS DE SONO (Consolidação STM→LTM) ---
        if metrics.total_steps > 0 && metrics.total_steps - last_sleep_step >= SLEEP_INTERVAL {
            // Calcula seletividade para decidir se vale a pena dormir
            let selectivity = compute_average_selectivity(&net);

            // Critérios para dormir: tem algum aprendizado E já teve alguns sucessos
            let has_learning = selectivity > MIN_SELECTIVITY_TO_SLEEP;
            let has_experience = metrics.successes >= MIN_SUCCESSES_TO_SLEEP;

            if has_learning && has_experience {
                println!("\n[SONO] Entrando em modo sono (Step {})...", metrics.total_steps);
                println!("   Seletividade: {:.3} | Sucessos: {} (bom aprendizado, vale a pena consolidar)",
                         selectivity, metrics.successes);

                // Entra em modo sono
                net.enter_sleep(SLEEP_REPLAY_NOISE, SLEEP_DURATION);

                // Executa ciclo de sono (replay + consolidação)
                for sleep_step in 0..SLEEP_DURATION {
                    net.update(&vec![0.0; NUM_NEURONS]);

                    // Visualiza durante sono (opcional - mostra replay)
                    if sleep_step % 50 == 0 {
                        clear_background(Color::new(0.05, 0.0, 0.1, 1.0)); // Fundo roxo = sono

                        draw_neural_network(&net, &neuron_positions, &sensors_idx, &motors_idx);

                        // Texto indicando sono
                        draw_text(&format!("[SONO] Consolidando memorias ({}/{})",
                                  sleep_step, SLEEP_DURATION),
                                  50.0, 30.0, 24.0, PURPLE);
                        draw_text("(Replay espontaneo + Transferencia STM->LTM)",
                                  50.0, 60.0, 18.0, VIOLET);

                        next_frame().await;
                    }
                }

                // Acorda
                net.wake_up();
                last_sleep_step = metrics.total_steps;

                println!("[SONO] Acordou! Memorias consolidadas.");
                println!("   Proximo sono em {} steps\n", SLEEP_INTERVAL);
            } else {
                // Diagnostica por que não dormiu
                if !has_learning {
                    println!("\n[SONO] Pulou sono (Step {}): Seletividade {:.3} < {:.2} (aprendizado insuficiente)",
                            metrics.total_steps, selectivity, MIN_SELECTIVITY_TO_SLEEP);
                } else if !has_experience {
                    println!("\n[SONO] Pulou sono (Step {}): Apenas {} sucessos < {} (precisa de mais experiencia)",
                            metrics.total_steps, metrics.successes, MIN_SUCCESSES_TO_SLEEP);
                }
                last_sleep_step = metrics.total_steps; // Reseta timer mesmo sem dormir
            }
        }

        // --- VISUALIZAÇÃO ---
        // Calcula tamanho da célula dinamicamente para caber na tela
        // Espaço disponível: ~400px de altura, deixa margem para crescer até grid 15x15
        let max_grid_dimension = 400.0;
        let cell_size = (max_grid_dimension / env.grid_size as f32).min(40.0);

        draw_grid_environment(&env, 50.0, 100.0, cell_size);

        draw_neural_network(&net, &neuron_positions, &sensors_idx, &motors_idx);

        // Calcula seletividade para display
        let current_selectivity = compute_average_selectivity(&net);

        draw_metrics(
            &metrics,
            &net,
            &env,
            exploration_rate,
            current_selectivity,
            last_sleep_step,
            SLEEP_INTERVAL
        );

        // Painel lateral direito com legenda
        draw_legend_panel();

        // Controla FPS para não fritar a CPU
        next_frame().await;
    }
}