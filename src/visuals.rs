use macroquad::prelude::*;
use crate::network::{Network, NetworkState};
use crate::nenv::NeuronType;
use crate::environment::{Environment, Metrics};

// Constantes de layout
const NETWORK_CENTER_X: f32 = 600.0;
const NETWORK_CENTER_Y: f32 = 300.0;
const NETWORK_RADIUS: f32 = 200.0;

/// Calcula posições dos neurônios em círculo para visualização
pub fn compute_neuron_positions(num_neurons: usize) -> Vec<(f32, f32)> {
    let mut positions = Vec::new();
    for i in 0..num_neurons {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (num_neurons as f32);
        let x = NETWORK_CENTER_X + NETWORK_RADIUS * angle.cos();
        let y = NETWORK_CENTER_Y + NETWORK_RADIUS * angle.sin();
        positions.push((x, y));
    }
    positions
}

/// Desenha o GridWorld
pub fn draw_grid_environment(env: &Environment, x_offset: f32, y_offset: f32, cell_size: f32) {
    draw_text("Ambiente (GridWorld)", x_offset, y_offset - 15.0, 30.0, WHITE);

    // Grade
    for i in 0..env.grid_size {
        for j in 0..env.grid_size {
            let x = x_offset + (j as f32) * cell_size;
            let y = y_offset + (i as f32) * cell_size;
            draw_rectangle_lines(x, y, cell_size, cell_size, 1.0, DARKGRAY);
        }
    }

    // Comida
    let food_x = x_offset + (env.food.x as f32) * cell_size + cell_size / 2.0;
    let food_y = y_offset + (env.food.y as f32) * cell_size + cell_size / 2.0;
    draw_circle(food_x, food_y, cell_size * 0.3, GREEN);

    // Agente
    let agent_x = x_offset + (env.agent.x as f32) * cell_size + cell_size / 2.0;
    let agent_y = y_offset + (env.agent.y as f32) * cell_size + cell_size / 2.0;
    draw_circle(agent_x, agent_y, cell_size * 0.4, BLUE);
}

/// Desenha a Rede Neural
pub fn draw_neural_network(
    net: &Network,
    positions: &[(f32, f32)],
    sensors: &[usize],
    motors: &[usize]
) {
    draw_text("Rede Neural NEN-V (40 neurônios)", NETWORK_CENTER_X - 150.0, 50.0, 26.0, WHITE);
    draw_text("S_UP/DOWN/LEFT/RIGHT = Sensores | M_UP/DOWN/LEFT/RIGHT = Motores",
              NETWORK_CENTER_X - 250.0, 75.0, 14.0, LIGHTGRAY);

    // 1. Desenha Sinapses (Conexões)
    for i in 0..net.neurons.len() {
        for j in 0..net.neurons.len() {
            // Peso Total = Curto Prazo (STM) + Longo Prazo (LTM)
            let w = net.neurons[i].dendritoma.weights[j] + net.neurons[i].dendritoma.weights_ltm[j];

            if w > 0.1 {
                let (x1, y1) = positions[j]; // Origem
                let (x2, y2) = positions[i]; // Destino

                // Verde = Origem Excitatória | Vermelho = Origem Inibitória
                let color = if net.neurons[j].neuron_type == NeuronType::Excitatory {
                    Color::new(0.0, 1.0, 0.0, (w * 0.4).min(0.8) as f32)
                } else {
                    Color::new(1.0, 0.0, 0.0, (w * 0.4).min(0.8) as f32)
                };

                // Espessura = Força da Memória
                draw_line(x1, y1, x2, y2, (w as f32).min(5.0), color);
            }
        }
    }

    // 2. Desenha Neurônios
    for (i, n) in net.neurons.iter().enumerate() {
        let (x, y) = positions[i];

        // Cor baseada no estado
        let color = if n.is_firing {
            YELLOW // Disparando (Spike)
        } else if n.neuron_type == NeuronType::Excitatory {
            BLUE   // Excitatório
        } else {
            RED    // Inibitório
        };

        // Tamanho = Nível de Energia (Glia)
        let radius = 4.0 + n.glia.energy_fraction() as f32 * 6.0;

        draw_circle(x, y, radius, color);

        // Highlight e Labels para Sensores e Motores
        if sensors.contains(&i) {
            draw_circle_lines(x, y, radius + 4.0, 2.0, ORANGE);
            // Label descritivo
            let label = match sensors.iter().position(|&s| s == i) {
                Some(0) => "S_UP",
                Some(1) => "S_DOWN",
                Some(2) => "S_LEFT",
                Some(3) => "S_RIGHT",
                _ => ""
            };
            draw_text(label, x - 18.0, y - radius - 8.0, 13.0, ORANGE);
        }

        if motors.contains(&i) {
            draw_circle_lines(x, y, radius + 4.0, 2.0, PURPLE);
            // Label descritivo
            let label = match motors.iter().position(|&m| m == i) {
                Some(0) => "M_UP",
                Some(1) => "M_DOWN",
                Some(2) => "M_LEFT",
                Some(3) => "M_RIGHT",
                _ => ""
            };
            draw_text(label, x - 18.0, y + radius + 18.0, 13.0, PURPLE);
        }

        // ID do Neurônio (somente para não-I/O)
        if !sensors.contains(&i) && !motors.contains(&i) {
            draw_text(&format!("{}", i), x - 4.0, y + 3.0, 11.0, DARKGRAY);
        }
    }
}

/// Desenha Painel de Métricas PROFISSIONAL E COMPLETO
pub fn draw_metrics(
    metrics: &Metrics,
    net: &Network,
    env: &Environment,
    exploration_rate: f64,
    selectivity: f64,
    last_sleep_step: u64,
    sleep_interval: u64
) {
    let panel_y = 520.0;  // Movido para cima
    let panel_height = screen_height() - panel_y;

    // Fundo do painel (mais escuro)
    draw_rectangle(0.0, panel_y, screen_width(), panel_height, Color::new(0.05, 0.05, 0.05, 0.95));
    draw_line(0.0, panel_y, screen_width(), panel_y, 3.0, Color::new(0.3, 0.6, 0.9, 1.0));

    // === COLUNA 1: PERFORMANCE ===
    let col1_x = 20.0;
    let col1_y = panel_y + 20.0;

    draw_text("PERFORMANCE", col1_x, col1_y, 16.0, Color::new(0.2, 1.0, 0.5, 1.0));
    draw_rectangle(col1_x, col1_y + 3.0, 120.0, 1.0, Color::new(0.2, 1.0, 0.5, 0.5));

    let mut y = col1_y + 20.0;

    draw_text(&format!("Grid:{}x{} Exp:{:.0}%", env.grid_size, env.grid_size, exploration_rate*100.0),
              col1_x, y, 12.0, WHITE);
    y += 16.0;
    draw_text(&format!("Score:{} Sel:{:.2}", metrics.score, selectivity),
              col1_x, y, 12.0, LIME);
    y += 16.0;
    draw_text(&format!("Avg:{:.1} Taxa:{:.0}%", metrics.average_steps(), metrics.success_rate_recent()),
              col1_x, y, 12.0, SKYBLUE);
    y += 16.0;
    draw_text(&format!("Colisoes:{} Best:{}", metrics.wall_collisions, metrics.best_episode_steps),
              col1_x, y, 12.0, LIGHTGRAY);

    // === COLUNA 2: ESTADO DA REDE ===
    let col2_x = 200.0;
    let col2_y = panel_y + 20.0;

    draw_text("REDE NEURAL", col2_x, col2_y, 16.0, Color::new(0.5, 0.8, 1.0, 1.0));
    draw_rectangle(col2_x, col2_y + 3.0, 120.0, 1.0, Color::new(0.5, 0.8, 1.0, 0.5));

    y = col2_y + 20.0;

    let firing_count = net.neurons.iter().filter(|n| n.is_firing).count();
    let avg_energy = net.average_energy();

    draw_text(&format!("E:{:.0}% Fire:{}/{}",
                       avg_energy, firing_count, net.neurons.len()),
              col2_x, y, 12.0, WHITE);
    y += 16.0;

    draw_text(&format!("Exc:{} Inh:{}",
                       net.neurons.iter().filter(|n| matches!(n.neuron_type, NeuronType::Excitatory)).count(),
                       net.neurons.iter().filter(|n| matches!(n.neuron_type, NeuronType::Inhibitory)).count()),
              col2_x, y, 12.0, SKYBLUE);

    // === COLUNA 3: MEMÓRIA E SONO ===
    let col3_x = 380.0;
    let col3_y = panel_y + 20.0;

    let state = net.get_state();
    let state_text = match state {
        NetworkState::Awake => "ACORDADO",
        NetworkState::Sleep { .. } => "DORMINDO"
    };
    let state_color = match state {
        NetworkState::Awake => Color::new(1.0, 0.9, 0.3, 1.0),
        NetworkState::Sleep { .. } => Color::new(0.7, 0.3, 1.0, 1.0)
    };

    draw_text(&format!("MEMORIA [{}]", state_text), col3_x, col3_y, 16.0, state_color);

    y = col3_y + 20.0;

    // Calcula peso STM vs LTM médio
    let mut total_stm = 0.0;
    let mut total_ltm = 0.0;
    let mut count = 0;
    for neuron in &net.neurons {
        for i in 0..neuron.dendritoma.weights.len() {
            total_stm += neuron.dendritoma.weights[i];
            total_ltm += neuron.dendritoma.weights_ltm[i];
            count += 1;
        }
    }
    let avg_stm = if count > 0 { total_stm / count as f64 } else { 0.0 };
    let avg_ltm = if count > 0 { total_ltm / count as f64 } else { 0.0 };

    draw_text(&format!("STM:{:.3} LTM:{:.3}", avg_stm, avg_ltm), col3_x, y, 12.0, SKYBLUE);
    y += 16.0;

    if metrics.total_steps > last_sleep_step {
        let steps_until_sleep = sleep_interval - (metrics.total_steps - last_sleep_step);
        draw_text(&format!("Sono em: {}s", steps_until_sleep), col3_x, y, 12.0, PINK);
    }

    // === COLUNA 4: ATIVIDADE MOTORA ===
    let col4_x = 560.0;
    let col4_y = panel_y + 20.0;

    draw_text("MOTORES", col4_x, col4_y, 16.0, Color::new(1.0, 0.5, 0.0, 1.0));
    draw_rectangle(col4_x, col4_y + 3.0, 100.0, 1.0, Color::new(1.0, 0.5, 0.0, 0.5));

    y = col4_y + 20.0;

    let motor_names = ["UP", "DN", "LF", "RT"];
    let total_fires: u32 = metrics.motor_fires.iter().sum();

    for (i, name) in motor_names.iter().enumerate() {
        let fires = metrics.motor_fires[i];
        let percentage = if total_fires > 0 {
            (fires as f64 / total_fires as f64) * 100.0
        } else {
            0.0
        };

        // Barra de progresso compacta
        let bar_width = 80.0 * (percentage / 100.0) as f32;
        draw_rectangle(col4_x + 30.0, y - 10.0, bar_width, 12.0, Color::new(1.0, 0.6, 0.0, 0.7));
        draw_rectangle_lines(col4_x + 30.0, y - 10.0, 80.0, 12.0, 1.0, ORANGE);

        draw_text(&format!("{}:{:.0}%", name, percentage), col4_x, y, 12.0, WHITE);
        y += 16.0;
    }

    // === GRÁFICO DE SCORE ===
    let graph_x = 720.0;
    let graph_y = panel_y + 20.0;
    let graph_width = 200.0;
    let graph_height = 90.0;

    draw_text("HISTORICO SCORE", graph_x, graph_y, 16.0, Color::new(0.3, 0.9, 0.9, 1.0));

    // Fundo do gráfico
    draw_rectangle(graph_x, graph_y + 8.0, graph_width, graph_height, Color::new(0.1, 0.1, 0.15, 0.8));
    draw_rectangle_lines(graph_x, graph_y + 8.0, graph_width, graph_height, 1.0, SKYBLUE);

    // Desenha linha de score
    if metrics.score_history.len() > 1 {
        let max_score = *metrics.score_history.iter().max().unwrap_or(&1);
        for i in 1..metrics.score_history.len() {
            let x1 = graph_x + (i - 1) as f32 * (graph_width / 100.0);
            let y1 = graph_y + graph_height + 8.0 - (metrics.score_history[i - 1] as f32 / max_score as f32 * graph_height);

            let x2 = graph_x + i as f32 * (graph_width / 100.0);
            let y2 = graph_y + graph_height + 8.0 - (metrics.score_history[i] as f32 / max_score as f32 * graph_height);

            draw_line(x1, y1, x2, y2, 2.0, LIME);
        }
    }

    // Label
    draw_text(&format!("Max:{}", metrics.score), graph_x + 8.0, graph_y + 24.0, 12.0, LIGHTGRAY);

    // === DIFERENÇA STEPS vs MOVIMENTOS ===
    let col5_x = 960.0;
    let col5_y = panel_y + 20.0;
    draw_text("STEPS vs MOVIMENTOS", col5_x, col5_y, 16.0, Color::new(0.9, 0.9, 0.3, 1.0));
    y = col5_y + 20.0;
    draw_text(&format!("Total Steps: {}", metrics.total_steps), col5_x, y, 12.0, SKYBLUE);
    y += 16.0;
    draw_text(&format!("Total Movimentos: {}", metrics.total_movements), col5_x, y, 12.0, LIME);
    y += 16.0;
    let idle_rate = if metrics.total_steps > 0 {
        ((metrics.total_steps - metrics.total_movements) as f64 / metrics.total_steps as f64) * 100.0
    } else {
        0.0
    };
    draw_text(&format!("Taxa Inativo: {:.1}%", idle_rate), col5_x, y, 12.0, ORANGE);
}

/// Desenha legenda explicativa da visualização
pub fn draw_legend_panel() {
    let panel_x = 1050.0;
    let panel_y = 60.0;
    let panel_width = 200.0;
    let panel_height = 400.0;

    // Fundo do painel
    draw_rectangle(panel_x, panel_y, panel_width, panel_height,
                   Color::new(0.08, 0.08, 0.12, 0.95));
    draw_rectangle_lines(panel_x, panel_y, panel_width, panel_height, 2.0,
                        Color::new(0.9, 0.7, 0.2, 1.0));

    let mut y = panel_y + 20.0;
    draw_text("LEGENDA", panel_x + 10.0, y, 20.0, Color::new(1.0, 0.9, 0.3, 1.0));
    y += 30.0;

    // === NEURÔNIOS ===
    draw_text("NEURONIOS (Circulos):", panel_x + 10.0, y, 15.0, WHITE);
    y += 22.0;

    // Excitatório
    draw_circle(panel_x + 18.0, y - 4.0, 5.0, BLUE);
    draw_text("Azul = Excitatorio", panel_x + 30.0, y, 13.0, LIGHTGRAY);
    y += 16.0;
    draw_text("  (estimula outros)", panel_x + 30.0, y, 10.0, DARKGRAY);
    y += 18.0;

    // Inibitório
    draw_circle(panel_x + 18.0, y - 4.0, 5.0, RED);
    draw_text("Vermelho = Inibitorio", panel_x + 30.0, y, 13.0, LIGHTGRAY);
    y += 16.0;
    draw_text("  (inibe outros)", panel_x + 30.0, y, 10.0, DARKGRAY);
    y += 18.0;

    // Disparando
    draw_circle(panel_x + 18.0, y - 4.0, 5.0, YELLOW);
    draw_text("Amarelo = Disparando", panel_x + 30.0, y, 13.0, LIGHTGRAY);
    y += 16.0;
    draw_text("  (spike ativo)", panel_x + 30.0, y, 10.0, DARKGRAY);
    y += 18.0;

    // Tamanho = Energia
    draw_circle(panel_x + 16.0, y - 4.0, 3.0, GRAY);
    draw_circle(panel_x + 28.0, y - 4.0, 7.0, GRAY);
    draw_text("Tamanho = Energia", panel_x + 40.0, y, 13.0, LIGHTGRAY);
    y += 22.0;

    // === CONEXÕES (SINAPSES) ===
    draw_text("SINAPSES (Linhas):", panel_x + 10.0, y, 15.0, WHITE);
    y += 22.0;

    // Verde
    draw_line(panel_x + 13.0, y - 4.0, panel_x + 40.0, y - 4.0, 2.5, GREEN);
    draw_text("Verde = Excitatoria", panel_x + 45.0, y, 13.0, LIGHTGRAY);
    y += 16.0;
    draw_text("  (passa ativacao)", panel_x + 45.0, y, 10.0, DARKGRAY);
    y += 18.0;

    // Vermelha
    draw_line(panel_x + 13.0, y - 4.0, panel_x + 40.0, y - 4.0, 2.5, Color::new(1.0, 0.0, 0.0, 0.6));
    draw_text("Vermelha = Inibitoria", panel_x + 45.0, y, 13.0, LIGHTGRAY);
    y += 16.0;
    draw_text("  (bloqueia ativacao)", panel_x + 45.0, y, 10.0, DARKGRAY);
    y += 18.0;

    // Espessura
    draw_line(panel_x + 13.0, y - 7.0, panel_x + 32.0, y - 7.0, 1.0, GRAY);
    draw_line(panel_x + 13.0, y - 2.0, panel_x + 32.0, y - 2.0, 4.0, GRAY);
    draw_text("Grossura = Forca", panel_x + 36.0, y, 13.0, LIGHTGRAY);
    y += 16.0;
    draw_text("  (peso aprendido)", panel_x + 36.0, y, 10.0, DARKGRAY);
    y += 22.0;

    // === NEURÔNIOS ESPECIAIS ===
    draw_text("NEURONIOS I/O:", panel_x + 10.0, y, 15.0, WHITE);
    y += 22.0;

    // Sensores
    draw_circle(panel_x + 18.0, y - 4.0, 5.0, BLUE);
    draw_circle_lines(panel_x + 18.0, y - 4.0, 8.0, 1.5, ORANGE);
    draw_text("Laranja = Sensor", panel_x + 30.0, y, 13.0, ORANGE);
    y += 16.0;
    draw_text("  (entrada)", panel_x + 30.0, y, 10.0, DARKGRAY);
    y += 18.0;

    // Motores
    draw_circle(panel_x + 18.0, y - 4.0, 5.0, BLUE);
    draw_circle_lines(panel_x + 18.0, y - 4.0, 8.0, 1.5, PURPLE);
    draw_text("Roxo = Motor", panel_x + 30.0, y, 13.0, PURPLE);
    y += 16.0;
    draw_text("  (saida/acao)", panel_x + 30.0, y, 10.0, DARKGRAY);
    y += 22.0;

    // === MEMÓRIA ===
    draw_text("MEMORIA:", panel_x + 10.0, y, 15.0, WHITE);
    y += 20.0;
    draw_text("STM = Curto Prazo", panel_x + 13.0, y, 12.0, SKYBLUE);
    y += 14.0;
    draw_text("  (volatil)", panel_x + 13.0, y, 10.0, DARKGRAY);
    y += 18.0;
    draw_text("LTM = Longo Prazo", panel_x + 13.0, y, 12.0, VIOLET);
    y += 14.0;
    draw_text("  (consolidado no sono)", panel_x + 13.0, y, 9.0, DARKGRAY);
}

/// Desenha painel lateral direito com detalhes de neurônios específicos
pub fn draw_neuron_details_panel(net: &Network, sensors: &[usize], motors: &[usize]) {
    let panel_x = 1050.0;
    let panel_y = 100.0;
    let panel_width = 220.0;
    let panel_height = 440.0;

    // Fundo do painel
    draw_rectangle(panel_x, panel_y, panel_width, panel_height,
                   Color::new(0.08, 0.08, 0.12, 0.95));
    draw_rectangle_lines(panel_x, panel_y, panel_width, panel_height, 2.0,
                        Color::new(0.4, 0.6, 1.0, 1.0));

    let mut y = panel_y + 20.0;
    draw_text("DETALHES", panel_x + 10.0, y, 20.0, Color::new(0.5, 0.8, 1.0, 1.0));
    y += 30.0;

    // === SENSORES ===
    draw_text("SENSORES (Entrada)", panel_x + 10.0, y, 16.0, ORANGE);
    y += 20.0;

    for (i, &sensor_id) in sensors.iter().enumerate() {
        let neuron = &net.neurons[sensor_id];
        let label = match i {
            0 => "S_UP",
            1 => "S_DOWN",
            2 => "S_LEFT",
            3 => "S_RIGHT",
            _ => "???"
        };

        let status = if neuron.is_firing { "[*]" } else { "[ ]" };
        let e = neuron.glia.energy_fraction() * 100.0;

        draw_text(&format!("{} #{}: E{:.0}%", status, sensor_id, e),
                  panel_x + 15.0, y, 13.0, if neuron.is_firing { YELLOW } else { LIGHTGRAY });
        y += 16.0;
    }

    y += 10.0;

    // === MOTORES COM SINAPSES ===
    draw_text("MOTORES (Saída)", panel_x + 10.0, y, 16.0, PURPLE);
    y += 20.0;

    for (i, &motor_id) in motors.iter().enumerate() {
        let neuron = &net.neurons[motor_id];
        let label = match i {
            0 => "M_UP",
            1 => "M_DOWN",
            2 => "M_LEFT",
            3 => "M_RIGHT",
            _ => "???"
        };

        let status = if neuron.is_firing { "[*]" } else { "[ ]" };

        draw_text(&format!("{} #{}:", status, motor_id),
                  panel_x + 15.0, y, 13.0, if neuron.is_firing { YELLOW } else { WHITE });
        y += 16.0;

        // Mostra sinapse principal (sensor correspondente)
        let sensor_id = sensors[i];
        let w_stm = neuron.dendritoma.weights[sensor_id];
        let w_ltm = neuron.dendritoma.weights_ltm[sensor_id];
        let w_total = w_stm + w_ltm;

        draw_text(&format!("  ←S{} W:{:.3}", sensor_id, w_total),
                  panel_x + 20.0, y, 11.0, SKYBLUE);
        y += 14.0;
        draw_text(&format!("  STM:{:.3} LTM:{:.3}", w_stm, w_ltm),
                  panel_x + 20.0, y, 10.0, VIOLET);
        y += 18.0;
    }

    y += 10.0;

    // === ESTATÍSTICAS GERAIS ===
    draw_text("REDE GERAL", panel_x + 10.0, y, 16.0, Color::new(0.5, 1.0, 0.5, 1.0));
    y += 20.0;

    let firing_count = net.num_firing();
    let firing_pct = (firing_count as f64 / 40.0) * 100.0;
    draw_text(&format!("Disparando: {}/40", firing_count),
              panel_x + 15.0, y, 12.0, YELLOW);
    y += 15.0;
    draw_text(&format!("({:.0}% ativo)", firing_pct),
              panel_x + 15.0, y, 11.0, GOLD);
    y += 18.0;

    // Média de pesos STM/LTM
    let mut total_stm = 0.0;
    let mut total_ltm = 0.0;
    let mut count = 0;
    for neuron in &net.neurons {
        for i in 0..neuron.dendritoma.weights.len() {
            total_stm += neuron.dendritoma.weights[i];
            total_ltm += neuron.dendritoma.weights_ltm[i];
            count += 1;
        }
    }
    let avg_stm = if count > 0 { total_stm / count as f64 } else { 0.0 };
    let avg_ltm = if count > 0 { total_ltm / count as f64 } else { 0.0 };

    draw_text("Peso Médio Global:", panel_x + 15.0, y, 12.0, WHITE);
    y += 15.0;
    draw_text(&format!("STM: {:.4}", avg_stm), panel_x + 20.0, y, 11.0, SKYBLUE);
    y += 14.0;
    draw_text(&format!("LTM: {:.4}", avg_ltm), panel_x + 20.0, y, 11.0, VIOLET);
}
