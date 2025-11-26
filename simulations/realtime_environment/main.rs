//! # Simulação Realtime: Ambiente de Navegação com Aprendizado
//!
//! Uma simulação completa que testa todos os componentes da rede NEN-V:
//! - Rede neural spiking com STDP e homeostase
//! - Working Memory para contexto
//! - Predictive Coding para antecipação
//! - Curiosidade Intrínseca para exploração
//! - Neuromodulação (dopamina, norepinefrina)
//! - Eligibility Traces para credit assignment
//!
//! ## Ambiente
//!
//! Um agente 2D navega em um grid world com:
//! - Comida (reward positivo)
//! - Perigo (reward negativo)
//! - Obstáculos
//! - Pontos de interesse (novidade)
//!
//! ## Execução
//!
//! ```bash
//! cargo run --release --bin realtime_sim
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};
use nenv_v2::network::{Network, LearningMode};
use nenv_v2::working_memory::WorkingMemoryPool;
use nenv_v2::predictive::DeepPredictiveHierarchy;
use nenv_v2::intrinsic_motivation::CuriosityModule;
use nenv_v2::neuromodulation::{NeuromodulationSystem, NeuromodulatorType};

// ============================================================================
// CONFIGURAÇÃO DA SIMULAÇÃO
// ============================================================================

/// Configuração do ambiente
#[derive(Clone)]
pub struct SimConfig {
    /// Tamanho do grid (width x height)
    pub grid_size: (usize, usize),
    /// Número de itens de comida
    pub num_food: usize,
    /// Número de zonas de perigo
    pub num_danger: usize,
    /// Número de obstáculos
    pub num_obstacles: usize,
    /// Reward por coletar comida
    pub food_reward: f64,
    /// Reward por entrar em zona de perigo
    pub danger_penalty: f64,
    /// Custo por movimento (incentiva eficiência)
    pub movement_cost: f64,
    /// Intervalo de regeneração de comida (steps)
    pub food_respawn_interval: u64,
    /// Máximo de steps por episódio
    pub max_steps_per_episode: u64,
    /// Número de episódios
    pub num_episodes: u64,
    /// Intervalo de relatório (steps)
    pub report_interval: u64,
    /// Habilita visualização no terminal
    pub enable_visualization: bool,
    /// Delay entre frames (ms) para visualização
    pub frame_delay_ms: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            grid_size: (15, 15),
            num_food: 5,
            num_danger: 3,
            num_obstacles: 10,
            food_reward: 1.0,
            danger_penalty: -0.5,
            movement_cost: -0.01,
            food_respawn_interval: 50,
            max_steps_per_episode: 500,
            num_episodes: 100,
            report_interval: 100,
            enable_visualization: true,
            frame_delay_ms: 50,
        }
    }
}

// ============================================================================
// AMBIENTE
// ============================================================================

/// Tipos de célula no grid
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CellType {
    Empty,
    Food,
    Danger,
    Obstacle,
    Agent,
}

/// Direções de movimento
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
    Stay,
}

impl Direction {
    fn to_delta(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, -1),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
            Direction::Stay => (0, 0),
        }
    }

    fn from_index(idx: usize) -> Self {
        match idx {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => Direction::Stay,
        }
    }
}

/// Posição no grid
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

impl Position {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    pub fn move_in(&self, dir: Direction, max_x: usize, max_y: usize) -> Self {
        let (dx, dy) = dir.to_delta();
        let new_x = (self.x as i32 + dx).clamp(0, max_x as i32 - 1) as usize;
        let new_y = (self.y as i32 + dy).clamp(0, max_y as i32 - 1) as usize;
        Position::new(new_x, new_y)
    }
}

/// Estado do ambiente
pub struct Environment {
    pub config: SimConfig,
    pub grid: Vec<Vec<CellType>>,
    pub agent_pos: Position,
    pub food_positions: Vec<Position>,
    pub danger_positions: Vec<Position>,
    pub obstacle_positions: Vec<Position>,
    pub step_count: u64,
    pub episode_count: u64,
    pub total_reward: f64,
    pub episode_reward: f64,
    pub food_collected: u64,
    pub danger_hits: u64,
    rng_seed: u64,
}

impl Environment {
    pub fn new(config: SimConfig) -> Self {
        let mut env = Self {
            config: config.clone(),
            grid: vec![vec![CellType::Empty; config.grid_size.0]; config.grid_size.1],
            agent_pos: Position::new(config.grid_size.0 / 2, config.grid_size.1 / 2),
            food_positions: Vec::new(),
            danger_positions: Vec::new(),
            obstacle_positions: Vec::new(),
            step_count: 0,
            episode_count: 0,
            total_reward: 0.0,
            episode_reward: 0.0,
            food_collected: 0,
            danger_hits: 0,
            rng_seed: 42,
        };
        env.reset();
        env
    }

    /// Reset do ambiente para novo episódio
    pub fn reset(&mut self) {
        self.episode_count += 1;
        self.episode_reward = 0.0;

        // Limpa grid
        for row in &mut self.grid {
            for cell in row {
                *cell = CellType::Empty;
            }
        }

        self.food_positions.clear();
        self.danger_positions.clear();
        self.obstacle_positions.clear();

        // Posiciona agente no centro
        self.agent_pos = Position::new(
            self.config.grid_size.0 / 2,
            self.config.grid_size.1 / 2,
        );

        // Adiciona obstáculos
        for _ in 0..self.config.num_obstacles {
            let pos = self.random_empty_position();
            self.grid[pos.y][pos.x] = CellType::Obstacle;
            self.obstacle_positions.push(pos);
        }

        // Adiciona comida
        for _ in 0..self.config.num_food {
            self.spawn_food();
        }

        // Adiciona perigo
        for _ in 0..self.config.num_danger {
            let pos = self.random_empty_position();
            self.grid[pos.y][pos.x] = CellType::Danger;
            self.danger_positions.push(pos);
        }
    }

    /// Gera posição aleatória vazia
    fn random_empty_position(&mut self) -> Position {
        loop {
            let x = self.lcg_random() % self.config.grid_size.0;
            let y = self.lcg_random() % self.config.grid_size.1;
            let pos = Position::new(x, y);

            if self.grid[y][x] == CellType::Empty && pos != self.agent_pos {
                return pos;
            }
        }
    }

    /// LCG simples para reprodutibilidade
    fn lcg_random(&mut self) -> usize {
        self.rng_seed = self.rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_seed >> 16) & 0x7fff) as usize
    }

    /// Spawn de nova comida
    fn spawn_food(&mut self) {
        let pos = self.random_empty_position();
        self.grid[pos.y][pos.x] = CellType::Food;
        self.food_positions.push(pos);
    }

    /// Executa ação e retorna (reward, done)
    pub fn step(&mut self, action: Direction) -> (f64, bool) {
        self.step_count += 1;
        let mut reward = self.config.movement_cost;

        // Calcula nova posição
        let new_pos = self.agent_pos.move_in(
            action,
            self.config.grid_size.0,
            self.config.grid_size.1,
        );

        // Verifica colisão com obstáculo
        if self.grid[new_pos.y][new_pos.x] == CellType::Obstacle {
            // Não move - bateu no obstáculo
        } else {
            self.agent_pos = new_pos;

            // Verifica interações
            match self.grid[new_pos.y][new_pos.x] {
                CellType::Food => {
                    reward += self.config.food_reward;
                    self.food_collected += 1;
                    self.grid[new_pos.y][new_pos.x] = CellType::Empty;
                    self.food_positions.retain(|p| *p != new_pos);
                }
                CellType::Danger => {
                    reward += self.config.danger_penalty;
                    self.danger_hits += 1;
                }
                _ => {}
            }
        }

        // Respawn de comida
        if self.step_count % self.config.food_respawn_interval == 0 {
            if self.food_positions.len() < self.config.num_food {
                self.spawn_food();
            }
        }

        self.episode_reward += reward;
        self.total_reward += reward;

        // Verifica fim de episódio
        let done = self.step_count % self.config.max_steps_per_episode == 0;

        (reward, done)
    }

    /// Gera observação sensorial para a rede
    pub fn get_observation(&self) -> Vec<f64> {
        // Sensores:
        // - 8 direções (raycasting) x 3 canais (comida, perigo, obstáculo)
        // - Posição normalizada (x, y)
        // - Proximidade da comida mais próxima
        // - Proximidade do perigo mais próximo

        let mut obs = Vec::with_capacity(28);

        // Raycasting em 8 direções
        let directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ];

        for (dx, dy) in &directions {
            let (food, danger, obstacle) = self.raycast(*dx, *dy);
            obs.push(food);
            obs.push(danger);
            obs.push(obstacle);
        }

        // Posição normalizada
        obs.push(self.agent_pos.x as f64 / self.config.grid_size.0 as f64);
        obs.push(self.agent_pos.y as f64 / self.config.grid_size.1 as f64);

        // Proximidade da comida mais próxima
        obs.push(self.nearest_distance(&self.food_positions));

        // Proximidade do perigo mais próximo
        obs.push(self.nearest_distance(&self.danger_positions));

        obs
    }

    /// Raycast em uma direção, retorna (food_signal, danger_signal, obstacle_signal)
    fn raycast(&self, dx: i32, dy: i32) -> (f64, f64, f64) {
        let max_dist = self.config.grid_size.0.max(self.config.grid_size.1) as f64;
        let mut food: f64 = 0.0;
        let mut danger: f64 = 0.0;
        let mut obstacle: f64 = 0.0;

        let mut x = self.agent_pos.x as i32;
        let mut y = self.agent_pos.y as i32;

        for dist in 1..=max_dist as i32 {
            x += dx;
            y += dy;

            if x < 0 || x >= self.config.grid_size.0 as i32 ||
               y < 0 || y >= self.config.grid_size.1 as i32 {
                break;
            }

            let signal = 1.0 / (dist as f64 + 1.0);

            match self.grid[y as usize][x as usize] {
                CellType::Food => food = food.max(signal),
                CellType::Danger => danger = danger.max(signal),
                CellType::Obstacle => {
                    obstacle = obstacle.max(signal);
                    break; // Obstáculo bloqueia visão
                }
                _ => {}
            }
        }

        (food, danger, obstacle)
    }

    /// Distância normalizada ao objeto mais próximo
    fn nearest_distance(&self, positions: &[Position]) -> f64 {
        if positions.is_empty() {
            return 0.0;
        }

        let max_dist = ((self.config.grid_size.0.pow(2) + self.config.grid_size.1.pow(2)) as f64).sqrt();

        let min_dist = positions.iter()
            .map(|p| {
                let dx = p.x as f64 - self.agent_pos.x as f64;
                let dy = p.y as f64 - self.agent_pos.y as f64;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(f64::MAX, f64::min);

        1.0 - (min_dist / max_dist).min(1.0)
    }

    /// Renderiza estado atual no terminal
    pub fn render(&self) {
        // Limpa tela
        print!("\x1B[2J\x1B[1;1H");

        println!("╔{}╗", "═".repeat(self.config.grid_size.0 * 2 + 1));

        for y in 0..self.config.grid_size.1 {
            print!("║ ");
            for x in 0..self.config.grid_size.0 {
                let pos = Position::new(x, y);
                let symbol = if pos == self.agent_pos {
                    "🤖"
                } else {
                    match self.grid[y][x] {
                        CellType::Empty => "· ",
                        CellType::Food => "🍎",
                        CellType::Danger => "💀",
                        CellType::Obstacle => "🧱",
                        CellType::Agent => "🤖",
                    }
                };
                print!("{}", symbol);
            }
            println!("║");
        }

        println!("╚{}╝", "═".repeat(self.config.grid_size.0 * 2 + 1));
    }
}

// ============================================================================
// AGENTE NEURAL
// ============================================================================

/// Estatísticas do agente
#[derive(Default, Clone)]
pub struct AgentStats {
    pub avg_firing_rate: f64,
    pub avg_energy: f64,
    pub dopamine_level: f64,
    pub norepinephrine_level: f64,
    pub free_energy: f64,
    pub curiosity_reward: f64,
    pub exploration_rate: f64,
    pub wm_active_slots: usize,
}

/// Agente neural completo
pub struct NeuralAgent {
    /// Rede neural principal
    pub network: Network,
    /// Sistema de neuromodulação
    pub neuromod: NeuromodulationSystem,
    /// Working Memory
    pub working_memory: WorkingMemoryPool,
    /// Predictive Coding
    pub predictive: DeepPredictiveHierarchy,
    /// Curiosidade Intrínseca
    pub curiosity: CuriosityModule,
    /// Taxa de exploração atual
    pub exploration_rate: f64,
    /// Decaimento da exploração
    pub exploration_decay: f64,
    /// Histórico de estados recentes
    pub state_history: VecDeque<Vec<f64>>,
    /// Histórico de ações recentes
    pub action_history: VecDeque<usize>,
    /// Histórico de rewards
    pub reward_history: VecDeque<f64>,
    /// Configuração
    pub config: AutoConfig,
}

impl NeuralAgent {
    pub fn new(observation_size: usize, num_actions: usize) -> Self {
        // Configura rede via AutoConfig
        let task = TaskSpec {
            num_sensors: observation_size,
            num_actuators: num_actions,
            task_type: TaskType::ReinforcementLearning {
                reward_density: RewardDensity::Moderate,
                temporal_horizon: Some(50),
            },
        };

        let config = AutoConfig::from_task(task);
        let mut network = config.build_network().expect("Configuração válida");
        network.set_learning_mode(LearningMode::STDP);

        // Working Memory
        let wm_capacity = config.params.working_memory.capacity;
        let working_memory = WorkingMemoryPool::new(wm_capacity, observation_size);

        // Predictive Coding
        let predictive = DeepPredictiveHierarchy::new_three_level_deep(observation_size);

        // Curiosidade
        let curiosity = CuriosityModule::new(observation_size, num_actions);

        Self {
            network,
            neuromod: NeuromodulationSystem::new(),
            working_memory,
            predictive,
            curiosity,
            exploration_rate: 0.3,
            exploration_decay: 0.995,
            state_history: VecDeque::with_capacity(100),
            action_history: VecDeque::with_capacity(100),
            reward_history: VecDeque::with_capacity(100),
            config,
        }
    }

    /// Seleciona ação baseada na observação
    pub fn select_action(&mut self, observation: &[f64]) -> usize {
        // 1. Processa através da hierarquia preditiva
        let pred_output = self.predictive.process(observation);

        // 2. Armazena contexto na working memory
        self.working_memory.encode(observation.to_vec(), self.network.current_time_step);
        self.working_memory.sustain();

        // 3. Prepara input para rede (observação + contexto WM + predições)
        let mut network_input = vec![0.0; self.network.num_neurons()];

        // Mapeia observação para neurônios sensores
        for (i, &val) in observation.iter().enumerate() {
            if i < network_input.len() {
                network_input[i] = val;
            }
        }

        // 4. Processa na rede spiking
        self.network.update(&network_input);

        // 5. Extrai atividade dos neurônios atuadores
        let actuator_start = self.config.architecture.actuator_indices.start;
        let actuator_end = self.config.architecture.actuator_indices.end;
        let num_actions = actuator_end - actuator_start;

        let mut action_values: Vec<f64> = (actuator_start..actuator_end)
            .map(|i| {
                if self.network.neurons[i].is_firing {
                    1.0
                } else {
                    // Usa firing rate recente como proxy do potencial
                    self.network.neurons[i].recent_firing_rate
                }
            })
            .collect();

        // 6. Adiciona bônus de curiosidade
        if let Some(last_state) = self.state_history.back() {
            for (i, action_val) in action_values.iter_mut().enumerate() {
                let action_vec = self.action_to_vec(i, num_actions);
                // Simula próximo estado (simplificado)
                let expected_next = observation.iter()
                    .enumerate()
                    .map(|(j, &v)| v + action_vec.get(j % action_vec.len()).unwrap_or(&0.0) * 0.1)
                    .collect::<Vec<_>>();

                let curiosity_bonus = self.curiosity.compute_intrinsic_reward(
                    last_state,
                    &action_vec,
                    &expected_next,
                );
                *action_val += curiosity_bonus * 0.5;
            }
        }

        // 7. Exploração epsilon-greedy com decaimento
        let lcg_rand = || -> f64 {
            static mut SEED: u64 = 12345;
            unsafe {
                SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
                ((SEED >> 16) & 0x7fff) as f64 / 32768.0
            }
        };

        let action = if lcg_rand() < self.exploration_rate {
            // Exploração: ação aleatória
            (lcg_rand() * num_actions as f64) as usize % num_actions
        } else {
            // Exploitation: ação com maior valor
            action_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        // 8. Atualiza históricos
        self.state_history.push_back(observation.to_vec());
        if self.state_history.len() > 100 {
            self.state_history.pop_front();
        }
        self.action_history.push_back(action);
        if self.action_history.len() > 100 {
            self.action_history.pop_front();
        }

        action
    }

    /// Processa reward e atualiza rede
    pub fn receive_reward(&mut self, reward: f64, next_observation: &[f64]) {
        // 1. Processa reward através do sistema de neuromodulação
        let _rpe = self.neuromod.process_reward(reward);

        // 2. Modula plasticidade baseado em dopamina
        let dopamine = self.neuromod.get_level(NeuromodulatorType::Dopamine);
        // Aplica modulação global via learning rate
        let base_lr = 0.01;
        let modulated_lr = base_lr * (1.0 + dopamine * reward.signum());
        for neuron in &mut self.network.neurons {
            neuron.dendritoma.set_learning_rate(modulated_lr);
        }

        // 3. Atualiza modelo de curiosidade
        // O modelo é atualizado implicitamente ao chamar compute_intrinsic_reward
        // que internamente treina o forward model
        if let (Some(last_state), Some(&last_action)) =
            (self.state_history.back(), self.action_history.back())
        {
            let action_vec = self.action_to_vec(last_action, 4);
            // Computa reward intrínseco que também atualiza o modelo interno
            let _ = self.curiosity.compute_intrinsic_reward(last_state, &action_vec, next_observation);
        }

        // 4. Atualiza taxa de exploração
        self.exploration_rate *= self.exploration_decay;
        self.exploration_rate = self.exploration_rate.max(0.05);

        // 5. Registra reward
        self.reward_history.push_back(reward);
        if self.reward_history.len() > 100 {
            self.reward_history.pop_front();
        }

        // 6. Atualiza neuromodulação
        self.neuromod.update();
    }

    /// Converte índice de ação para vetor one-hot
    fn action_to_vec(&self, action: usize, num_actions: usize) -> Vec<f64> {
        let mut vec = vec![0.0; num_actions];
        if action < num_actions {
            vec[action] = 1.0;
        }
        vec
    }

    /// Retorna estatísticas atuais
    pub fn get_stats(&self) -> AgentStats {
        let firing_count = self.network.neurons.iter()
            .filter(|n| n.is_firing)
            .count();

        let avg_energy = self.network.neurons.iter()
            .map(|n| n.glia.energy)
            .sum::<f64>() / self.network.num_neurons() as f64;

        AgentStats {
            avg_firing_rate: firing_count as f64 / self.network.num_neurons() as f64,
            avg_energy,
            dopamine_level: self.neuromod.get_level(NeuromodulatorType::Dopamine),
            norepinephrine_level: self.neuromod.get_level(NeuromodulatorType::Norepinephrine),
            free_energy: self.predictive.total_free_energy(),
            curiosity_reward: self.curiosity.get_stats().avg_intrinsic_reward,
            exploration_rate: self.exploration_rate,
            wm_active_slots: self.working_memory.active_count(),
        }
    }

    /// Reset do agente para novo episódio
    pub fn reset_episode(&mut self) {
        self.state_history.clear();
        self.action_history.clear();
        self.reward_history.clear();
        self.neuromod.reset();
    }
}

// ============================================================================
// MÉTRICAS E RELATÓRIOS
// ============================================================================

/// Métricas de performance
#[derive(Default)]
pub struct PerformanceMetrics {
    pub episode_rewards: Vec<f64>,
    pub episode_food: Vec<u64>,
    pub episode_danger: Vec<u64>,
    pub episode_steps: Vec<u64>,
    pub avg_firing_rates: Vec<f64>,
    pub avg_energies: Vec<f64>,
    pub exploration_rates: Vec<f64>,
}

impl PerformanceMetrics {
    pub fn record_episode(
        &mut self,
        reward: f64,
        food: u64,
        danger: u64,
        steps: u64,
        agent_stats: &AgentStats,
    ) {
        self.episode_rewards.push(reward);
        self.episode_food.push(food);
        self.episode_danger.push(danger);
        self.episode_steps.push(steps);
        self.avg_firing_rates.push(agent_stats.avg_firing_rate);
        self.avg_energies.push(agent_stats.avg_energy);
        self.exploration_rates.push(agent_stats.exploration_rate);
    }

    pub fn print_summary(&self, last_n: usize) {
        let n = last_n.min(self.episode_rewards.len());
        if n == 0 {
            return;
        }

        let recent_rewards: Vec<_> = self.episode_rewards.iter().rev().take(n).collect();
        let avg_reward: f64 = recent_rewards.iter().copied().sum::<f64>() / n as f64;

        let recent_food: Vec<_> = self.episode_food.iter().rev().take(n).collect();
        let avg_food: f64 = recent_food.iter().map(|&&x| x as f64).sum::<f64>() / n as f64;

        let recent_danger: Vec<_> = self.episode_danger.iter().rev().take(n).collect();
        let avg_danger: f64 = recent_danger.iter().map(|&&x| x as f64).sum::<f64>() / n as f64;

        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║              RESUMO DE PERFORMANCE (últimos {})            ║", n);
        println!("╠════════════════════════════════════════════════════════════╣");
        println!("║ Reward Médio:      {:>38.3} ║", avg_reward);
        println!("║ Comida Coletada:   {:>38.1} ║", avg_food);
        println!("║ Perigos Atingidos: {:>38.1} ║", avg_danger);
        println!("║ Episódios Totais:  {:>38} ║", self.episode_rewards.len());
        println!("╚════════════════════════════════════════════════════════════╝");
    }
}

// ============================================================================
// LOOP PRINCIPAL DA SIMULAÇÃO
// ============================================================================

/// Executa simulação completa
pub fn run_simulation(config: SimConfig) {
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     NEN-V v2.0 - SIMULAÇÃO DE AMBIENTE DE NAVEGAÇÃO       ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Grid: {}x{}                                                 ║",
             config.grid_size.0, config.grid_size.1);
    println!("║ Episódios: {}                                             ║",
             config.num_episodes);
    println!("║ Steps/Episódio: {}                                       ║",
             config.max_steps_per_episode);
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Inicializa ambiente
    let mut env = Environment::new(config.clone());

    // Tamanho da observação e ações
    let obs_size = 28; // 8 direções * 3 canais + 2 pos + 2 proximidades
    let num_actions = 4; // Up, Down, Left, Right

    // Inicializa agente
    let mut agent = NeuralAgent::new(obs_size, num_actions);
    let mut metrics = PerformanceMetrics::default();

    println!("Rede neural configurada:");
    println!("  - Neurônios: {}", agent.network.num_neurons());
    println!("  - Sensores: {}", obs_size);
    println!("  - Atuadores: {}", num_actions);
    println!("  - WM Capacity: {}", agent.config.params.working_memory.capacity);
    println!();

    let start_time = Instant::now();
    let mut episode_start_food = 0u64;
    let mut episode_start_danger = 0u64;

    // Loop de episódios
    for episode in 0..config.num_episodes {
        env.reset();
        agent.reset_episode();

        episode_start_food = env.food_collected;
        episode_start_danger = env.danger_hits;

        let episode_start = env.step_count;

        // Loop de steps
        loop {
            // Obtém observação
            let observation = env.get_observation();

            // Seleciona ação
            let action_idx = agent.select_action(&observation);
            let action = Direction::from_index(action_idx);

            // Executa ação no ambiente
            let (reward, done) = env.step(action);

            // Obtém nova observação
            let next_observation = env.get_observation();

            // Processa reward
            agent.receive_reward(reward, &next_observation);

            // Visualização (se habilitada)
            if config.enable_visualization && env.step_count % 10 == 0 {
                env.render();

                let stats = agent.get_stats();
                println!();
                println!("┌──────────────────────────────────────────┐");
                println!("│ Episode: {:>4} | Step: {:>6}             │",
                         episode + 1, env.step_count - episode_start);
                println!("│ Reward: {:>+7.3} | Total: {:>+8.2}        │",
                         reward, env.episode_reward);
                println!("│ Food: {:>3} | Danger: {:>3}                 │",
                         env.food_collected - episode_start_food,
                         env.danger_hits - episode_start_danger);
                println!("├──────────────────────────────────────────┤");
                println!("│ Firing Rate: {:>6.2}% | Energy: {:>5.1}%  │",
                         stats.avg_firing_rate * 100.0,
                         stats.avg_energy);
                println!("│ Dopamine: {:>6.3} | NE: {:>6.3}         │",
                         stats.dopamine_level,
                         stats.norepinephrine_level);
                println!("│ Exploration: {:>5.1}% | WM Slots: {:>2}     │",
                         stats.exploration_rate * 100.0,
                         stats.wm_active_slots);
                println!("│ Free Energy: {:>8.2}                   │",
                         stats.free_energy);
                println!("└──────────────────────────────────────────┘");

                std::thread::sleep(Duration::from_millis(config.frame_delay_ms));
            }

            if done {
                break;
            }
        }

        // Registra métricas do episódio
        let stats = agent.get_stats();
        metrics.record_episode(
            env.episode_reward,
            env.food_collected - episode_start_food,
            env.danger_hits - episode_start_danger,
            env.step_count - episode_start,
            &stats,
        );

        // Relatório periódico
        if (episode + 1) % 10 == 0 {
            metrics.print_summary(10);
            println!("Tempo decorrido: {:.1}s", start_time.elapsed().as_secs_f64());
        }
    }

    // Relatório final
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                    SIMULAÇÃO CONCLUÍDA                     ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    metrics.print_summary(config.num_episodes as usize);

    println!("\nTempo total: {:.2}s", start_time.elapsed().as_secs_f64());
    println!("Steps totais: {}", env.step_count);
    println!("Steps/segundo: {:.1}", env.step_count as f64 / start_time.elapsed().as_secs_f64());
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    // Configuração padrão
    let mut config = SimConfig::default();

    // Modo interativo ou benchmark
    let args: Vec<String> = std::env::args().collect();

    if args.contains(&"--benchmark".to_string()) {
        // Modo benchmark: sem visualização, mais episódios
        config.enable_visualization = false;
        config.num_episodes = 500;
        config.report_interval = 50;
    } else if args.contains(&"--fast".to_string()) {
        // Modo rápido
        config.enable_visualization = true;
        config.frame_delay_ms = 10;
        config.num_episodes = 50;
    } else if args.contains(&"--demo".to_string()) {
        // Modo demonstração
        config.enable_visualization = true;
        config.frame_delay_ms = 100;
        config.num_episodes = 20;
    }

    run_simulation(config);
}

// ============================================================================
// TESTES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_creation() {
        let config = SimConfig::default();
        let env = Environment::new(config);

        assert!(env.food_positions.len() > 0);
        assert!(env.danger_positions.len() > 0);
    }

    #[test]
    fn test_agent_creation() {
        let agent = NeuralAgent::new(28, 4);
        assert!(agent.network.num_neurons() > 32);
    }

    #[test]
    fn test_observation_generation() {
        let config = SimConfig::default();
        let env = Environment::new(config);

        let obs = env.get_observation();
        assert_eq!(obs.len(), 28);
    }

    #[test]
    fn test_action_selection() {
        let mut agent = NeuralAgent::new(28, 4);
        let observation = vec![0.5; 28];

        let action = agent.select_action(&observation);
        assert!(action < 4);
    }

    #[test]
    fn test_simulation_step() {
        let config = SimConfig::default();
        let mut env = Environment::new(config);

        let initial_step = env.step_count;
        let (_, _) = env.step(Direction::Up);

        assert_eq!(env.step_count, initial_step + 1);
    }

    #[test]
    fn test_full_episode() {
        let mut config = SimConfig::default();
        config.max_steps_per_episode = 50;
        config.enable_visualization = false;

        let mut env = Environment::new(config);
        let mut agent = NeuralAgent::new(28, 4);

        for _ in 0..50 {
            let obs = env.get_observation();
            let action_idx = agent.select_action(&obs);
            let action = Direction::from_index(action_idx);
            let (reward, done) = env.step(action);
            let next_obs = env.get_observation();
            agent.receive_reward(reward, &next_obs);

            if done {
                break;
            }
        }

        assert!(env.step_count > 0);
    }
}
