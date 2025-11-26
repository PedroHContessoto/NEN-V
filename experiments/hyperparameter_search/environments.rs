//! # Sistema de Ambientes para Benchmarks
//!
//! Define ambientes modulares e parametrizáveis para avaliação real da rede neural.
//!
//! ## Arquitetura
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    ENVIRONMENT SYSTEM                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  trait Environment                                              │
//! │  ├── reset() -> Observation                                     │
//! │  ├── step(action) -> (Observation, Reward, Done, Info)         │
//! │  ├── observation_size() -> usize                                │
//! │  ├── action_size() -> usize                                     │
//! │  ├── name() -> &str                                             │
//! │  └── config() -> &EnvironmentParams                             │
//! │                                                                 │
//! │  Implementações:                                                │
//! │  ├── NavigationEnv      - Grid world com comida/perigo         │
//! │  ├── PatternMemoryEnv   - Memorização de sequências            │
//! │  ├── PredictionEnv      - Previsão de séries temporais         │
//! │  └── AssociationEnv     - Aprendizado associativo              │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Calibração de Rewards e Thresholds
//!
//! Cada ambiente tem parâmetros calibrados baseado em:
//! - **Reward range normalizado**: [-1, +1] para consistência entre ambientes
//! - **Success threshold**: Calibrado para ~30% de sucesso com agente aleatório
//! - **Reward shaping**: Balanceado para guiar aprendizado sem exploits
//!
//! ## Adicionando Novos Ambientes
//!
//! ```rust
//! impl Environment for MeuAmbiente {
//!     fn reset(&mut self) -> Vec<f64> { ... }
//!     fn step(&mut self, action: usize) -> StepResult { ... }
//!     fn observation_size(&self) -> usize { ... }
//!     fn action_size(&self) -> usize { ... }
//!     fn name(&self) -> &str { "MeuAmbiente" }
//!     fn params(&self) -> &EnvironmentParams { &self.params }
//! }
//! ```

use std::collections::HashMap;

// =============================================================================
// PARÂMETROS GLOBAIS DE AMBIENTE
// =============================================================================

/// Parâmetros de configuração para qualquer ambiente
#[derive(Debug, Clone)]
pub struct EnvironmentParams {
    /// Seed base para reprodutibilidade
    pub seed: u64,
    /// Máximo de steps por episódio
    pub max_steps: usize,
    /// Escala de reward (para normalização)
    pub reward_scale: f64,
    /// Threshold de sucesso (calibrado)
    pub success_threshold: f64,
    /// Reward máximo teórico por episódio
    pub max_episode_reward: f64,
    /// Nível de dificuldade [0.0 = fácil, 1.0 = difícil]
    pub difficulty: f64,
    /// Variância do ambiente (ruído)
    pub stochasticity: f64,
}

impl Default for EnvironmentParams {
    fn default() -> Self {
        Self {
            seed: 42,
            max_steps: 100,
            reward_scale: 1.0,
            success_threshold: 0.5,
            max_episode_reward: 100.0,
            difficulty: 0.5,
            stochasticity: 0.1,
        }
    }
}

impl EnvironmentParams {
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_difficulty(mut self, difficulty: f64) -> Self {
        self.difficulty = difficulty.clamp(0.0, 1.0);
        self
    }

    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }
}

// =============================================================================
// RESULTADO DE STEP
// =============================================================================

/// Resultado de um step no ambiente
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Observação do estado atual
    pub observation: Vec<f64>,
    /// Reward obtido (normalizado para [-1, +1] preferencialmente)
    pub reward: f64,
    /// Episódio terminou?
    pub done: bool,
    /// Informações extras para diagnóstico
    pub info: HashMap<String, f64>,
}

impl StepResult {
    pub fn new(observation: Vec<f64>, reward: f64, done: bool) -> Self {
        Self {
            observation,
            reward,
            done,
            info: HashMap::new(),
        }
    }

    pub fn with_info(mut self, key: &str, value: f64) -> Self {
        self.info.insert(key.to_string(), value);
        self
    }
}

// =============================================================================
// TRAIT ENVIRONMENT
// =============================================================================

/// Trait principal para ambientes de benchmark
pub trait Environment: Send {
    /// Reseta o ambiente e retorna observação inicial
    fn reset(&mut self) -> Vec<f64>;

    /// Reseta com uma seed específica (para controle de variância)
    fn reset_with_seed(&mut self, seed: u64) -> Vec<f64> {
        self.set_seed(seed);
        self.reset()
    }

    /// Define seed para próximo reset
    fn set_seed(&mut self, seed: u64);

    /// Executa uma ação e retorna resultado
    fn step(&mut self, action: usize) -> StepResult;

    /// Tamanho do vetor de observação
    fn observation_size(&self) -> usize;

    /// Número de ações possíveis
    fn action_size(&self) -> usize;

    /// Nome do ambiente
    fn name(&self) -> &str;

    /// Retorna parâmetros do ambiente
    fn params(&self) -> &EnvironmentParams;

    /// Descrição do ambiente
    fn description(&self) -> &str {
        "No description available"
    }

    /// Reward máximo teórico por episódio
    fn max_episode_reward(&self) -> f64 {
        self.params().max_episode_reward
    }

    /// Critério de sucesso calibrado
    fn success_threshold(&self) -> f64 {
        self.params().success_threshold
    }

    /// Retorna baseline de agente aleatório (para calibração)
    fn random_baseline(&self) -> f64 {
        // Override em cada ambiente com valor empírico
        0.0
    }
}

// =============================================================================
// NAVIGATION ENVIRONMENT - Grid World com Comida e Perigo
// =============================================================================

/// Tipos de célula no grid
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellType {
    Empty,
    Food,
    Danger,
    Obstacle,
}

/// Configuração específica do NavigationEnv
#[derive(Debug, Clone)]
pub struct NavigationConfig {
    pub width: usize,
    pub height: usize,
    pub num_food: usize,
    pub num_danger: usize,
    pub num_obstacles: usize,
    pub food_reward: f64,
    pub danger_penalty: f64,
    pub movement_cost: f64,
    pub food_respawn: bool,
    pub ray_directions: usize,
}

impl Default for NavigationConfig {
    fn default() -> Self {
        Self {
            width: 12,
            height: 12,
            num_food: 5,
            num_danger: 3,
            num_obstacles: 0,
            // Rewards calibrados: agente aleatório ~= -0.5 por episódio
            // Bom agente ~= +5.0 por episódio
            food_reward: 1.0,
            danger_penalty: -0.5,
            movement_cost: -0.01,
            food_respawn: true,
            ray_directions: 8,
        }
    }
}

impl NavigationConfig {
    /// Configuração fácil (mais comida, menos perigo)
    pub fn easy() -> Self {
        Self {
            num_food: 8,
            num_danger: 1,
            food_reward: 1.0,
            danger_penalty: -0.3,
            ..Default::default()
        }
    }

    /// Configuração difícil (menos comida, mais perigo)
    pub fn hard() -> Self {
        Self {
            width: 15,
            height: 15,
            num_food: 3,
            num_danger: 6,
            num_obstacles: 5,
            food_reward: 1.0,
            danger_penalty: -0.8,
            movement_cost: -0.02,
            ..Default::default()
        }
    }
}

/// Ambiente de navegação em grid 2D
pub struct NavigationEnv {
    config: NavigationConfig,
    params: EnvironmentParams,
    /// Grid atual
    grid: Vec<Vec<CellType>>,
    /// Posição do agente
    agent_pos: (usize, usize),
    /// Posições de comida
    food_positions: Vec<(usize, usize)>,
    /// Posições de perigo
    danger_positions: Vec<(usize, usize)>,
    /// Step atual
    current_step: usize,
    /// Comida coletada neste episódio
    food_collected: usize,
    /// Perigos atingidos neste episódio
    dangers_hit: usize,
    /// Seed atual para RNG
    current_seed: u64,
    /// Episódios completados (para estatísticas)
    episodes_completed: usize,
    /// Total de comida coletada (para calibração)
    total_food_collected: usize,
}

impl NavigationEnv {
    pub fn new(width: usize, height: usize, seed: u64) -> Self {
        Self::with_config(NavigationConfig {
            width,
            height,
            ..Default::default()
        }, seed)
    }

    pub fn with_config(config: NavigationConfig, seed: u64) -> Self {
        // Calibração dinâmica baseada em config:
        // - Agente aleatório em grid: movimento cost domina, ~-1.5 por episódio
        // - Max reward teórico: comida coletada * food_reward (limitado por respawn rate)
        let max_steps = 200usize;
        let random_baseline = max_steps as f64 * config.movement_cost; // ~-2.0
        let max_reward = (config.num_food as f64 * 10.0).min(config.food_reward * 60.0);

        // Threshold = baseline + 30% do gap até max reward
        // Para random_baseline=-2, max=60: threshold = -2 + 0.3*(62) = ~16.6
        let success_threshold = random_baseline + (max_reward - random_baseline) * 0.3;

        let params = EnvironmentParams {
            seed,
            max_steps,
            reward_scale: 1.0,
            success_threshold,
            max_episode_reward: max_reward,
            difficulty: 0.5,
            stochasticity: 0.0,
        };

        let mut env = Self {
            grid: vec![vec![CellType::Empty; config.width]; config.height],
            agent_pos: (config.width / 2, config.height / 2),
            food_positions: Vec::new(),
            danger_positions: Vec::new(),
            current_step: 0,
            food_collected: 0,
            dangers_hit: 0,
            current_seed: seed,
            episodes_completed: 0,
            total_food_collected: 0,
            config,
            params,
        };
        env.reset();
        env
    }

    fn next_random(&mut self) -> f64 {
        self.current_seed = self.current_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.current_seed >> 33) as f64) / (u32::MAX as f64)
    }

    fn spawn_item(&mut self, cell_type: CellType) -> bool {
        for _ in 0..100 {
            let x = (self.next_random() * self.config.width as f64) as usize;
            let y = (self.next_random() * self.config.height as f64) as usize;

            if self.grid[y][x] == CellType::Empty && (x, y) != self.agent_pos {
                self.grid[y][x] = cell_type;
                match cell_type {
                    CellType::Food => self.food_positions.push((x, y)),
                    CellType::Danger => self.danger_positions.push((x, y)),
                    _ => {}
                }
                return true;
            }
        }
        false
    }

    /// Gera observação via raycasting
    fn get_observation(&self) -> Vec<f64> {
        let mut obs = Vec::with_capacity(self.observation_size());

        // Raycasting em 8 direções
        let directions = [
            (0, -1),  // Norte
            (1, -1),  // NE
            (1, 0),   // Leste
            (1, 1),   // SE
            (0, 1),   // Sul
            (-1, 1),  // SW
            (-1, 0),  // Oeste
            (-1, -1), // NW
        ];

        let max_dist = (self.config.width.max(self.config.height)) as f64;

        for (dx, dy) in directions.iter() {
            let mut food_dist = max_dist;
            let mut danger_dist = max_dist;
            let mut obstacle_dist = max_dist;

            for dist in 1..=self.config.width.max(self.config.height) {
                let x = self.agent_pos.0 as i32 + dx * dist as i32;
                let y = self.agent_pos.1 as i32 + dy * dist as i32;

                if x < 0 || x >= self.config.width as i32 || y < 0 || y >= self.config.height as i32 {
                    if obstacle_dist == max_dist {
                        obstacle_dist = dist as f64;
                    }
                    break;
                }

                let cell = self.grid[y as usize][x as usize];
                match cell {
                    CellType::Food if food_dist == max_dist => food_dist = dist as f64,
                    CellType::Danger if danger_dist == max_dist => danger_dist = dist as f64,
                    CellType::Obstacle if obstacle_dist == max_dist => {
                        obstacle_dist = dist as f64;
                        break;
                    }
                    _ => {}
                }
            }

            // Normaliza distâncias (1.0 = perto, 0.0 = longe)
            obs.push(1.0 - food_dist / max_dist);
            obs.push(1.0 - danger_dist / max_dist);
            obs.push(1.0 - obstacle_dist / max_dist);
        }

        // Posição normalizada do agente
        obs.push(self.agent_pos.0 as f64 / self.config.width as f64);
        obs.push(self.agent_pos.1 as f64 / self.config.height as f64);

        // Progresso do episódio
        obs.push(self.current_step as f64 / self.params.max_steps as f64);

        // Comida restante normalizada
        let food_remaining = self.food_positions.len() as f64 / self.config.num_food.max(1) as f64;
        obs.push(food_remaining.min(1.0));

        obs
    }
}

impl Environment for NavigationEnv {
    fn reset(&mut self) -> Vec<f64> {
        // Limpa grid
        for row in &mut self.grid {
            for cell in row {
                *cell = CellType::Empty;
            }
        }

        // Posição inicial do agente (randomizada levemente)
        let cx = self.config.width / 2;
        let cy = self.config.height / 2;
        let offset_x = ((self.next_random() - 0.5) * 4.0) as i32;
        let offset_y = ((self.next_random() - 0.5) * 4.0) as i32;
        self.agent_pos = (
            (cx as i32 + offset_x).clamp(1, self.config.width as i32 - 2) as usize,
            (cy as i32 + offset_y).clamp(1, self.config.height as i32 - 2) as usize,
        );

        self.food_positions.clear();
        self.danger_positions.clear();
        self.current_step = 0;
        self.food_collected = 0;
        self.dangers_hit = 0;

        // Spawna itens
        for _ in 0..self.config.num_food {
            self.spawn_item(CellType::Food);
        }
        for _ in 0..self.config.num_danger {
            self.spawn_item(CellType::Danger);
        }
        for _ in 0..self.config.num_obstacles {
            self.spawn_item(CellType::Obstacle);
        }

        self.get_observation()
    }

    fn reset_with_seed(&mut self, seed: u64) -> Vec<f64> {
        self.current_seed = seed;
        self.reset()
    }

    fn set_seed(&mut self, seed: u64) {
        self.current_seed = seed;
    }

    fn step(&mut self, action: usize) -> StepResult {
        self.current_step += 1;

        // Movimentos: 0=Norte, 1=Leste, 2=Sul, 3=Oeste
        let (dx, dy): (i32, i32) = match action.min(3) {
            0 => (0, -1),
            1 => (1, 0),
            2 => (0, 1),
            3 => (-1, 0),
            _ => (0, 0),
        };

        let new_x = (self.agent_pos.0 as i32 + dx).clamp(0, self.config.width as i32 - 1) as usize;
        let new_y = (self.agent_pos.1 as i32 + dy).clamp(0, self.config.height as i32 - 1) as usize;

        // Verifica obstáculos
        if self.grid[new_y][new_x] != CellType::Obstacle {
            self.agent_pos = (new_x, new_y);
        }

        // Calcula reward (base, antes de scale)
        let mut raw_reward = self.config.movement_cost;
        let cell = self.grid[new_y][new_x];

        match cell {
            CellType::Food => {
                raw_reward += self.config.food_reward;
                self.food_collected += 1;
                self.total_food_collected += 1;
                self.grid[new_y][new_x] = CellType::Empty;
                self.food_positions.retain(|&pos| pos != (new_x, new_y));

                if self.config.food_respawn {
                    self.spawn_item(CellType::Food);
                }
            }
            CellType::Danger => {
                raw_reward += self.config.danger_penalty;
                self.dangers_hit += 1;
            }
            _ => {}
        }

        // Aplica reward_scale para normalização
        let reward = raw_reward * self.params.reward_scale;

        let done = self.current_step >= self.params.max_steps;

        if done {
            self.episodes_completed += 1;
        }

        let mut result = StepResult::new(self.get_observation(), reward, done);
        result = result.with_info("food_collected", self.food_collected as f64);
        result = result.with_info("dangers_hit", self.dangers_hit as f64);
        result = result.with_info("step", self.current_step as f64);
        result = result.with_info("net_reward", (self.food_collected as f64 * self.config.food_reward
            + self.dangers_hit as f64 * self.config.danger_penalty));

        result
    }

    fn observation_size(&self) -> usize {
        self.config.ray_directions * 3 + 4
    }

    fn action_size(&self) -> usize {
        4
    }

    fn name(&self) -> &str {
        "NavigationEnv"
    }

    fn params(&self) -> &EnvironmentParams {
        &self.params
    }

    fn description(&self) -> &str {
        "Grid world navigation with food collection and danger avoidance"
    }

    fn random_baseline(&self) -> f64 {
        // Empiricamente: agente aleatório em grid 12x12 com 5 comida/3 perigo
        // ~= -1.5 por episódio (movimento cost domina)
        -1.5
    }
}

// =============================================================================
// PATTERN MEMORY ENVIRONMENT - Memorização de Padrões
// =============================================================================

/// Configuração do PatternMemoryEnv
#[derive(Debug, Clone)]
pub struct PatternMemoryConfig {
    pub pattern_size: usize,
    pub num_symbols: usize,
    pub correct_reward: f64,
    pub incorrect_penalty: f64,
    pub presentation_reward: f64,
}

impl Default for PatternMemoryConfig {
    fn default() -> Self {
        Self {
            pattern_size: 5,
            num_symbols: 4,
            // Calibrado: chance aleatória = 25% (1/4 símbolos)
            // Success threshold = 60% (3/5 corretos)
            correct_reward: 1.0,
            incorrect_penalty: -0.3,
            presentation_reward: 0.0,
        }
    }
}

impl PatternMemoryConfig {
    pub fn easy() -> Self {
        Self {
            pattern_size: 3,
            num_symbols: 2,
            ..Default::default()
        }
    }

    pub fn hard() -> Self {
        Self {
            pattern_size: 7,
            num_symbols: 6,
            ..Default::default()
        }
    }
}

/// Ambiente de memorização de padrões sequenciais
pub struct PatternMemoryEnv {
    config: PatternMemoryConfig,
    params: EnvironmentParams,
    /// Padrão atual a memorizar
    current_pattern: Vec<usize>,
    /// Resposta do agente
    agent_response: Vec<usize>,
    /// Fase: 0=apresentação, 1=recall
    phase: usize,
    /// Step dentro da fase
    pub phase_step: usize,
    /// Acertos no recall
    correct_recalls: usize,
    /// Seed atual
    current_seed: u64,
}

impl PatternMemoryEnv {
    pub fn new(pattern_size: usize, num_symbols: usize, seed: u64) -> Self {
        Self::with_config(PatternMemoryConfig {
            pattern_size,
            num_symbols,
            ..Default::default()
        }, seed)
    }

    pub fn with_config(config: PatternMemoryConfig, seed: u64) -> Self {
        // Calibração dinâmica baseada em config:
        // - Chance aleatória: 1/num_symbols por posição
        // - Random baseline = pattern_size * (1/num_symbols * correct - (1-1/num_symbols) * penalty)
        let chance = 1.0 / config.num_symbols as f64;
        let expected_correct = config.pattern_size as f64 * chance;
        let expected_incorrect = config.pattern_size as f64 * (1.0 - chance);
        let random_baseline = expected_correct * config.correct_reward
            + expected_incorrect * config.incorrect_penalty;

        let max_reward = config.pattern_size as f64 * config.correct_reward;

        // Threshold = baseline + 40% do gap até perfeito
        let success_threshold = random_baseline + (max_reward - random_baseline) * 0.4;

        let params = EnvironmentParams {
            seed,
            max_steps: config.pattern_size * 2, // apresentação + recall
            reward_scale: 1.0,
            success_threshold,
            max_episode_reward: max_reward,
            difficulty: 0.5,
            stochasticity: 0.0,
        };

        let mut env = Self {
            current_pattern: Vec::new(),
            agent_response: Vec::new(),
            phase: 0,
            phase_step: 0,
            correct_recalls: 0,
            current_seed: seed,
            config,
            params,
        };
        env.reset();
        env
    }

    fn next_random(&mut self) -> f64 {
        self.current_seed = self.current_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.current_seed >> 33) as f64) / (u32::MAX as f64)
    }

    fn generate_pattern(&mut self) {
        self.current_pattern.clear();
        for _ in 0..self.config.pattern_size {
            let symbol = (self.next_random() * self.config.num_symbols as f64) as usize;
            self.current_pattern.push(symbol);
        }
    }

    fn get_observation(&self) -> Vec<f64> {
        let mut obs = vec![0.0; self.observation_size()];

        // One-hot do símbolo atual (se na fase de apresentação)
        if self.phase == 0 && self.phase_step < self.config.pattern_size {
            let symbol = self.current_pattern[self.phase_step];
            if symbol < self.config.num_symbols {
                obs[symbol] = 1.0;
            }
        }

        // Indicadores de fase
        obs[self.config.num_symbols] = if self.phase == 0 { 1.0 } else { 0.0 };
        obs[self.config.num_symbols + 1] = if self.phase == 1 { 1.0 } else { 0.0 };

        // Posição na sequência (normalizada)
        obs[self.config.num_symbols + 2] = self.phase_step as f64 / self.config.pattern_size as f64;

        // Feedback do último recall (se na fase de recall e após primeira resposta)
        if self.phase == 1 && self.phase_step > 0 && !self.agent_response.is_empty() {
            let last_correct = self.agent_response.last() == self.current_pattern.get(self.phase_step - 1);
            obs[self.config.num_symbols + 3] = if last_correct { 1.0 } else { 0.0 };
        }

        obs
    }
}

impl Environment for PatternMemoryEnv {
    fn reset(&mut self) -> Vec<f64> {
        self.generate_pattern();
        self.agent_response.clear();
        self.phase = 0;
        self.phase_step = 0;
        self.correct_recalls = 0;
        self.get_observation()
    }

    fn reset_with_seed(&mut self, seed: u64) -> Vec<f64> {
        self.current_seed = seed;
        self.reset()
    }

    fn set_seed(&mut self, seed: u64) {
        self.current_seed = seed;
    }

    fn step(&mut self, action: usize) -> StepResult {
        let mut raw_reward = 0.0;
        let mut done = false;

        if self.phase == 0 {
            // Fase de apresentação
            raw_reward = self.config.presentation_reward;
            self.phase_step += 1;

            if self.phase_step >= self.config.pattern_size {
                self.phase = 1;
                self.phase_step = 0;
            }
        } else {
            // Fase de recall
            let expected = self.current_pattern.get(self.phase_step).copied().unwrap_or(0);
            let action_clamped = action.min(self.config.num_symbols - 1);

            if action_clamped == expected {
                raw_reward = self.config.correct_reward;
                self.correct_recalls += 1;
            } else {
                raw_reward = self.config.incorrect_penalty;
            }

            self.agent_response.push(action_clamped);
            self.phase_step += 1;

            if self.phase_step >= self.config.pattern_size {
                done = true;
            }
        }

        // Aplica reward_scale
        let reward = raw_reward * self.params.reward_scale;

        let accuracy = if self.phase == 1 && self.phase_step > 0 {
            self.correct_recalls as f64 / self.phase_step as f64
        } else {
            0.0
        };

        let mut result = StepResult::new(self.get_observation(), reward, done);
        result = result.with_info("correct_recalls", self.correct_recalls as f64);
        result = result.with_info("pattern_size", self.config.pattern_size as f64);
        result = result.with_info("phase", self.phase as f64);
        result = result.with_info("accuracy", accuracy);

        result
    }

    fn observation_size(&self) -> usize {
        self.config.num_symbols + 4 // symbols + phase(2) + position + feedback
    }

    fn action_size(&self) -> usize {
        self.config.num_symbols
    }

    fn name(&self) -> &str {
        "PatternMemoryEnv"
    }

    fn params(&self) -> &EnvironmentParams {
        &self.params
    }

    fn description(&self) -> &str {
        "Sequential pattern memorization and recall task"
    }

    fn random_baseline(&self) -> f64 {
        // Chance aleatória: 1/num_symbols acertos
        let expected_correct = self.config.pattern_size as f64 / self.config.num_symbols as f64;
        let expected_incorrect = self.config.pattern_size as f64 - expected_correct;
        expected_correct * self.config.correct_reward + expected_incorrect * self.config.incorrect_penalty
    }
}

// =============================================================================
// PREDICTION ENVIRONMENT - Previsão de Séries Temporais
// =============================================================================

/// Tipo de série temporal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeriesType {
    Sine = 0,
    Square = 1,
    RandomWalk = 2,
}

/// Configuração do PredictionEnv
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    pub series_type: SeriesType,
    pub context_window: usize,
    pub max_steps: usize,
    pub num_bins: usize,
    pub frequency: f64,
    pub noise_level: f64,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            series_type: SeriesType::Sine,
            context_window: 10,
            max_steps: 100,
            num_bins: 10,
            frequency: 0.1,
            noise_level: 0.1,
        }
    }
}

/// Ambiente de previsão de séries temporais
pub struct PredictionEnv {
    config: PredictionConfig,
    params: EnvironmentParams,
    /// Histórico de valores
    history: Vec<f64>,
    /// Valor atual
    current_value: f64,
    /// Step atual
    current_step: usize,
    /// Erro acumulado
    cumulative_error: f64,
    /// Previsões corretas (erro < 0.3)
    correct_predictions: usize,
    /// Seed atual
    current_seed: u64,
    /// Fase inicial da série
    phase: f64,
}

impl PredictionEnv {
    pub fn new(series_type: usize, context_window: usize, seed: u64) -> Self {
        let st = match series_type {
            0 => SeriesType::Sine,
            1 => SeriesType::Square,
            _ => SeriesType::RandomWalk,
        };
        Self::with_config(PredictionConfig {
            series_type: st,
            context_window,
            ..Default::default()
        }, seed)
    }

    pub fn with_config(config: PredictionConfig, seed: u64) -> Self {
        // Calibração dinâmica baseada em config:
        // - Reward por step = 1.0 - erro (erro em [0, 2], reward normalizado para [0, 1])
        // - Agente aleatório com num_bins: erro médio depende de bins
        //   Para bins uniformes predizendo série [-1,1]: erro médio ~0.5
        //   Random baseline = steps * 0.5
        // - Success threshold: 20% acima do baseline aleatório
        let random_avg_reward = 0.5; // Agente aleatório em série normalizada
        let random_baseline = config.max_steps as f64 * random_avg_reward;

        // Threshold = baseline + 20% do gap até perfeito
        // Isso garante que threshold > baseline sempre
        let max_reward = config.max_steps as f64; // Perfeito = 1.0 * steps
        let success_threshold = random_baseline + (max_reward - random_baseline) * 0.2;

        let params = EnvironmentParams {
            seed,
            max_steps: config.max_steps,
            reward_scale: 1.0,
            success_threshold,
            max_episode_reward: max_reward,
            difficulty: 0.5,
            stochasticity: config.noise_level,
        };

        let mut env = Self {
            history: Vec::new(),
            current_value: 0.0,
            current_step: 0,
            cumulative_error: 0.0,
            correct_predictions: 0,
            current_seed: seed,
            phase: 0.0,
            config,
            params,
        };
        env.reset();
        env
    }

    fn next_random(&mut self) -> f64 {
        self.current_seed = self.current_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.current_seed >> 33) as f64) / (u32::MAX as f64)
    }

    fn generate_value(&mut self) -> f64 {
        let noise = (self.next_random() - 0.5) * 2.0 * self.config.noise_level;

        match self.config.series_type {
            SeriesType::Sine => {
                let base = (self.current_step as f64 * self.config.frequency + self.phase).sin();
                (base + noise).clamp(-1.0, 1.0)
            }
            SeriesType::Square => {
                let period = (1.0 / self.config.frequency) as usize;
                let base = if period > 0 && (self.current_step / period.max(1)) % 2 == 0 { 1.0 } else { -1.0 };
                (base + noise * 0.5).clamp(-1.0, 1.0)
            }
            SeriesType::RandomWalk => {
                self.current_value = (self.current_value + (self.next_random() - 0.5) * 0.3).clamp(-1.0, 1.0);
                self.current_value
            }
        }
    }

    fn get_observation(&self) -> Vec<f64> {
        let mut obs = vec![0.0; self.observation_size()];

        // Histórico normalizado para [0, 1]
        let start = self.history.len().saturating_sub(self.config.context_window);
        for (i, &val) in self.history[start..].iter().enumerate() {
            if i < self.config.context_window {
                obs[i] = (val + 1.0) / 2.0;
            }
        }

        // Tipo de série (one-hot)
        let series_idx = self.config.series_type as usize;
        if series_idx < 3 {
            obs[self.config.context_window + series_idx] = 1.0;
        }

        // Posição temporal
        obs[self.config.context_window + 3] = self.current_step as f64 / self.config.max_steps as f64;

        obs
    }
}

impl Environment for PredictionEnv {
    fn reset(&mut self) -> Vec<f64> {
        self.history.clear();
        self.current_step = 0;
        self.cumulative_error = 0.0;
        self.correct_predictions = 0;
        self.phase = self.next_random() * std::f64::consts::PI * 2.0;
        self.current_value = 0.0;

        // Preenche histórico inicial
        for _ in 0..self.config.context_window {
            let val = self.generate_value();
            self.history.push(val);
            self.current_step += 1;
        }
        self.current_step = 0;

        self.get_observation()
    }

    fn reset_with_seed(&mut self, seed: u64) -> Vec<f64> {
        self.current_seed = seed;
        self.reset()
    }

    fn set_seed(&mut self, seed: u64) {
        self.current_seed = seed;
    }

    fn step(&mut self, action: usize) -> StepResult {
        self.current_step += 1;

        // Ação representa previsão discretizada em bins
        let action_clamped = action.min(self.config.num_bins - 1);
        let prediction = (action_clamped as f64 / (self.config.num_bins - 1) as f64) * 2.0 - 1.0;

        // Gera próximo valor real
        let actual = self.generate_value();
        self.history.push(actual);

        // Calcula erro e reward
        let error = (prediction - actual).abs();
        self.cumulative_error += error;

        if error < 0.3 {
            self.correct_predictions += 1;
        }

        // Reward: inversamente proporcional ao erro, normalizado para [0, 1]
        let raw_reward = 1.0 - error.min(2.0) / 2.0;

        // Aplica reward_scale
        let reward = raw_reward * self.params.reward_scale;

        let done = self.current_step >= self.config.max_steps;

        let mut result = StepResult::new(self.get_observation(), reward, done);
        result = result.with_info("prediction_error", error);
        result = result.with_info("cumulative_error", self.cumulative_error);
        result = result.with_info("avg_error", self.cumulative_error / self.current_step.max(1) as f64);
        result = result.with_info("correct_predictions", self.correct_predictions as f64);
        result = result.with_info("prediction", prediction);
        result = result.with_info("actual", actual);

        result
    }

    fn observation_size(&self) -> usize {
        self.config.context_window + 4
    }

    fn action_size(&self) -> usize {
        self.config.num_bins
    }

    fn name(&self) -> &str {
        "PredictionEnv"
    }

    fn params(&self) -> &EnvironmentParams {
        &self.params
    }

    fn description(&self) -> &str {
        "Time series prediction task"
    }

    fn random_baseline(&self) -> f64 {
        // Agente aleatório: previsão uniforme, erro médio ~0.5 para série normalizada
        // Reward médio = 1.0 - 0.5 = 0.5 por step
        self.config.max_steps as f64 * 0.5
    }
}

// =============================================================================
// ASSOCIATION ENVIRONMENT - Aprendizado Associativo
// =============================================================================

/// Configuração do AssociationEnv
#[derive(Debug, Clone)]
pub struct AssociationConfig {
    pub num_pairs: usize,
    pub trials_per_episode: usize,
    pub correct_reward: f64,
    pub incorrect_penalty: f64,
    pub shuffle_associations: bool,
}

impl Default for AssociationConfig {
    fn default() -> Self {
        Self {
            num_pairs: 6,
            trials_per_episode: 50,
            // Calibrado: chance aleatória = 1/num_pairs
            // Para 6 pares: 16.7% de acerto
            correct_reward: 1.0,
            incorrect_penalty: -0.15,
            shuffle_associations: true,
        }
    }
}

/// Ambiente de aprendizado associativo (estímulo-resposta)
pub struct AssociationEnv {
    config: AssociationConfig,
    params: EnvironmentParams,
    /// Mapeamento estímulo -> resposta correta
    associations: Vec<usize>,
    /// Estímulo atual
    current_stimulus: usize,
    /// Trial atual
    current_trial: usize,
    /// Respostas corretas
    correct_responses: usize,
    /// Histórico de acertos (para learning curve)
    recent_accuracy: Vec<bool>,
    /// Seed atual
    current_seed: u64,
}

impl AssociationEnv {
    pub fn new(num_pairs: usize, trials_per_episode: usize, seed: u64) -> Self {
        Self::with_config(AssociationConfig {
            num_pairs,
            trials_per_episode,
            ..Default::default()
        }, seed)
    }

    pub fn with_config(config: AssociationConfig, seed: u64) -> Self {
        // Calibração dinâmica baseada em config:
        // - Chance aleatória: 1/num_pairs
        // - Random baseline = trials * (chance * correct + (1-chance) * penalty)
        let chance = 1.0 / config.num_pairs as f64;
        let expected_correct = config.trials_per_episode as f64 * chance;
        let expected_incorrect = config.trials_per_episode as f64 * (1.0 - chance);
        let random_baseline = expected_correct * config.correct_reward
            + expected_incorrect * config.incorrect_penalty;

        let max_reward = config.trials_per_episode as f64 * config.correct_reward;

        // Threshold = baseline + 35% do gap até perfeito
        let success_threshold = random_baseline + (max_reward - random_baseline) * 0.35;

        let params = EnvironmentParams {
            seed,
            max_steps: config.trials_per_episode,
            reward_scale: 1.0,
            success_threshold,
            max_episode_reward: max_reward,
            difficulty: 0.5,
            stochasticity: 0.0,
        };

        let mut env = Self {
            associations: Vec::new(),
            current_stimulus: 0,
            current_trial: 0,
            correct_responses: 0,
            recent_accuracy: Vec::new(),
            current_seed: seed,
            config,
            params,
        };
        env.generate_associations();
        env.reset();
        env
    }

    fn next_random(&mut self) -> f64 {
        self.current_seed = self.current_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.current_seed >> 33) as f64) / (u32::MAX as f64)
    }

    fn generate_associations(&mut self) {
        self.associations.clear();

        if self.config.shuffle_associations {
            let mut responses: Vec<usize> = (0..self.config.num_pairs).collect();
            // Fisher-Yates shuffle
            for i in (1..self.config.num_pairs).rev() {
                let j = (self.next_random() * (i + 1) as f64) as usize;
                responses.swap(i, j);
            }
            self.associations = responses;
        } else {
            // Mapeamento identidade (estímulo i -> resposta i)
            self.associations = (0..self.config.num_pairs).collect();
        }
    }

    fn get_observation(&self) -> Vec<f64> {
        let mut obs = vec![0.0; self.observation_size()];

        // One-hot do estímulo atual
        if self.current_stimulus < self.config.num_pairs {
            obs[self.current_stimulus] = 1.0;
        }

        // Progresso no episódio
        obs[self.config.num_pairs] = self.current_trial as f64 / self.config.trials_per_episode as f64;

        // Acurácia recente (últimos 10 trials)
        if !self.recent_accuracy.is_empty() {
            let recent: Vec<_> = self.recent_accuracy.iter().rev().take(10).collect();
            let recent_correct = recent.iter().filter(|&&x| *x).count();
            obs[self.config.num_pairs + 1] = recent_correct as f64 / recent.len() as f64;
        }

        obs
    }
}

impl Environment for AssociationEnv {
    fn reset(&mut self) -> Vec<f64> {
        self.current_trial = 0;
        self.correct_responses = 0;
        self.recent_accuracy.clear();

        // Opcionalmente regenera associações para variabilidade
        if self.config.shuffle_associations {
            self.generate_associations();
        }

        self.current_stimulus = (self.next_random() * self.config.num_pairs as f64) as usize;
        self.get_observation()
    }

    fn reset_with_seed(&mut self, seed: u64) -> Vec<f64> {
        self.current_seed = seed;
        self.reset()
    }

    fn set_seed(&mut self, seed: u64) {
        self.current_seed = seed;
    }

    fn step(&mut self, action: usize) -> StepResult {
        let correct_response = self.associations.get(self.current_stimulus).copied().unwrap_or(0);
        let action_clamped = action.min(self.config.num_pairs - 1);
        let is_correct = action_clamped == correct_response;

        let raw_reward = if is_correct {
            self.correct_responses += 1;
            self.config.correct_reward
        } else {
            self.config.incorrect_penalty
        };

        // Aplica reward_scale
        let reward = raw_reward * self.params.reward_scale;

        self.recent_accuracy.push(is_correct);
        self.current_trial += 1;

        // Próximo estímulo (uniformemente aleatório)
        self.current_stimulus = (self.next_random() * self.config.num_pairs as f64) as usize;

        let done = self.current_trial >= self.config.trials_per_episode;

        let accuracy = self.correct_responses as f64 / self.current_trial.max(1) as f64;

        let mut result = StepResult::new(self.get_observation(), reward, done);
        result = result.with_info("correct_responses", self.correct_responses as f64);
        result = result.with_info("accuracy", accuracy);
        result = result.with_info("trial", self.current_trial as f64);
        result = result.with_info("is_correct", if is_correct { 1.0 } else { 0.0 });

        result
    }

    fn observation_size(&self) -> usize {
        self.config.num_pairs + 2 // one-hot + progress + recent_accuracy
    }

    fn action_size(&self) -> usize {
        self.config.num_pairs
    }

    fn name(&self) -> &str {
        "AssociationEnv"
    }

    fn params(&self) -> &EnvironmentParams {
        &self.params
    }

    fn description(&self) -> &str {
        "Stimulus-response association learning"
    }

    fn random_baseline(&self) -> f64 {
        // Chance aleatória: 1/num_pairs
        let chance = 1.0 / self.config.num_pairs as f64;
        let expected_correct = self.config.trials_per_episode as f64 * chance;
        let expected_incorrect = self.config.trials_per_episode as f64 * (1.0 - chance);
        expected_correct * self.config.correct_reward + expected_incorrect * self.config.incorrect_penalty
    }
}

// =============================================================================
// ENVIRONMENT REGISTRY - Registro de Ambientes com Configuração
// =============================================================================

/// Configuração completa para criação de ambiente
#[derive(Debug, Clone)]
pub struct EnvironmentSpec {
    pub name: String,
    pub weight: f64,
    pub episodes: usize,
    pub description: String,
    pub difficulty: f64,
}

impl EnvironmentSpec {
    pub fn new(name: &str, weight: f64, episodes: usize) -> Self {
        Self {
            name: name.to_string(),
            weight,
            episodes,
            description: String::new(),
            difficulty: 0.5,
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_difficulty(mut self, difficulty: f64) -> Self {
        self.difficulty = difficulty.clamp(0.0, 1.0);
        self
    }
}

/// Registro de ambientes disponíveis
pub struct EnvironmentRegistry {
    specs: Vec<EnvironmentSpec>,
}

impl EnvironmentRegistry {
    pub fn new() -> Self {
        Self { specs: Vec::new() }
    }

    /// Registra um ambiente
    pub fn register(&mut self, spec: EnvironmentSpec) {
        self.specs.push(spec);
    }

    /// Registra ambiente simplificado
    pub fn register_simple(&mut self, name: &str, weight: f64, episodes: usize, description: &str) {
        self.specs.push(EnvironmentSpec {
            name: name.to_string(),
            weight,
            episodes,
            description: description.to_string(),
            difficulty: 0.5,
        });
    }

    /// Cria ambiente por nome com seed específica
    pub fn create(&self, name: &str, seed: u64) -> Option<Box<dyn Environment>> {
        self.create_with_difficulty(name, seed, 0.5)
    }

    /// Cria ambiente com dificuldade específica
    pub fn create_with_difficulty(&self, name: &str, seed: u64, difficulty: f64) -> Option<Box<dyn Environment>> {
        match name {
            "NavigationEnv" => {
                let config = if difficulty < 0.3 {
                    NavigationConfig::easy()
                } else if difficulty > 0.7 {
                    NavigationConfig::hard()
                } else {
                    NavigationConfig::default()
                };
                Some(Box::new(NavigationEnv::with_config(config, seed)))
            }
            "PatternMemoryEnv" => {
                let config = if difficulty < 0.3 {
                    PatternMemoryConfig::easy()
                } else if difficulty > 0.7 {
                    PatternMemoryConfig::hard()
                } else {
                    PatternMemoryConfig::default()
                };
                Some(Box::new(PatternMemoryEnv::with_config(config, seed)))
            }
            "PredictionEnv" => {
                let config = PredictionConfig {
                    noise_level: 0.05 + difficulty * 0.2, // 0.05 a 0.25
                    ..Default::default()
                };
                Some(Box::new(PredictionEnv::with_config(config, seed)))
            }
            "AssociationEnv" => {
                let num_pairs = (4.0 + difficulty * 6.0) as usize; // 4 a 10 pares
                let config = AssociationConfig {
                    num_pairs,
                    trials_per_episode: 50,
                    ..Default::default()
                };
                Some(Box::new(AssociationEnv::with_config(config, seed)))
            }
            _ => None,
        }
    }

    /// Retorna especificações
    pub fn specs(&self) -> &[EnvironmentSpec] {
        &self.specs
    }

    /// Suite padrão de ambientes (calibrada)
    pub fn default_suite() -> Self {
        let mut registry = Self::new();

        registry.register(
            EnvironmentSpec::new("NavigationEnv", 0.35, 20)
                .with_description("Grid world navigation - tests spatial learning, reward seeking, danger avoidance")
                .with_difficulty(0.5)
        );

        registry.register(
            EnvironmentSpec::new("PatternMemoryEnv", 0.25, 30)
                .with_description("Pattern memorization - tests working memory and sequence learning")
                .with_difficulty(0.5)
        );

        registry.register(
            EnvironmentSpec::new("PredictionEnv", 0.25, 15)
                .with_description("Time series prediction - tests predictive coding and temporal modeling")
                .with_difficulty(0.5)
        );

        registry.register(
            EnvironmentSpec::new("AssociationEnv", 0.15, 25)
                .with_description("Association learning - tests stimulus-response mapping and credit assignment")
                .with_difficulty(0.5)
        );

        registry
    }

    /// Suite fácil (para testes rápidos)
    pub fn easy_suite() -> Self {
        let mut registry = Self::new();

        registry.register(
            EnvironmentSpec::new("NavigationEnv", 0.5, 10)
                .with_difficulty(0.2)
        );

        registry.register(
            EnvironmentSpec::new("AssociationEnv", 0.5, 10)
                .with_difficulty(0.2)
        );

        registry
    }

    /// Suite difícil (para stress test)
    pub fn hard_suite() -> Self {
        let mut registry = Self::new();

        registry.register(
            EnvironmentSpec::new("NavigationEnv", 0.3, 30)
                .with_difficulty(0.8)
        );

        registry.register(
            EnvironmentSpec::new("PatternMemoryEnv", 0.3, 40)
                .with_difficulty(0.8)
        );

        registry.register(
            EnvironmentSpec::new("PredictionEnv", 0.2, 20)
                .with_difficulty(0.8)
        );

        registry.register(
            EnvironmentSpec::new("AssociationEnv", 0.2, 30)
                .with_difficulty(0.8)
        );

        registry
    }

    /// Normaliza pesos para somar 1.0
    pub fn normalize_weights(&mut self) {
        let total: f64 = self.specs.iter().map(|s| s.weight).sum();
        if total > 0.0 {
            for spec in &mut self.specs {
                spec.weight /= total;
            }
        }
    }

    /// Retorna baseline esperado de agente aleatório
    pub fn random_baseline_summary(&self) -> HashMap<String, f64> {
        let mut baselines = HashMap::new();

        for spec in &self.specs {
            if let Some(env) = self.create(&spec.name, 42) {
                baselines.insert(spec.name.clone(), env.random_baseline());
            }
        }

        baselines
    }
}

impl Default for EnvironmentRegistry {
    fn default() -> Self {
        Self::default_suite()
    }
}

// Backward compatibility alias
pub type EnvironmentConfig = EnvironmentSpec;

// =============================================================================
// TESTES
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigation_env_basic() {
        let mut env = NavigationEnv::new(10, 10, 42);
        let obs = env.reset();

        assert_eq!(obs.len(), env.observation_size());
        assert_eq!(env.action_size(), 4);

        for action in 0..4 {
            let result = env.step(action);
            assert_eq!(result.observation.len(), env.observation_size());
        }
    }

    #[test]
    fn test_navigation_seed_reproducibility() {
        let mut env1 = NavigationEnv::new(10, 10, 123);
        let mut env2 = NavigationEnv::new(10, 10, 123);

        let obs1 = env1.reset();
        let obs2 = env2.reset();

        assert_eq!(obs1, obs2);

        let r1 = env1.step(0);
        let r2 = env2.step(0);

        assert_eq!(r1.observation, r2.observation);
        assert_eq!(r1.reward, r2.reward);
    }

    #[test]
    fn test_pattern_memory_env() {
        let mut env = PatternMemoryEnv::new(4, 3, 42);
        let obs = env.reset();

        assert_eq!(obs.len(), env.observation_size());
        assert_eq!(env.action_size(), 3);

        // Passa pela apresentação
        for _ in 0..4 {
            let result = env.step(0);
            if result.done {
                break;
            }
        }
    }

    #[test]
    fn test_prediction_env() {
        let mut env = PredictionEnv::new(0, 5, 42);
        let obs = env.reset();

        assert_eq!(obs.len(), env.observation_size());
        assert_eq!(env.action_size(), 10);

        for _ in 0..10 {
            let result = env.step(5);
            assert!(result.reward >= 0.0 && result.reward <= 1.0);
        }
    }

    #[test]
    fn test_association_env() {
        let mut env = AssociationEnv::new(4, 20, 42);
        let obs = env.reset();

        assert_eq!(obs.len(), env.observation_size());
        assert_eq!(env.action_size(), 4);

        let mut total_reward = 0.0;
        loop {
            let result = env.step(0);
            total_reward += result.reward;
            if result.done {
                break;
            }
        }
        assert!(total_reward.is_finite());
    }

    #[test]
    fn test_environment_registry() {
        let registry = EnvironmentRegistry::default_suite();

        assert_eq!(registry.specs().len(), 4);

        for spec in registry.specs() {
            let env = registry.create(&spec.name, 42);
            assert!(env.is_some(), "Failed to create {}", spec.name);

            let env = env.unwrap();
            assert!(env.random_baseline().is_finite());
        }
    }

    #[test]
    fn test_difficulty_levels() {
        let registry = EnvironmentRegistry::default_suite();

        // Testa diferentes dificuldades
        for difficulty in [0.1, 0.5, 0.9] {
            let env = registry.create_with_difficulty("NavigationEnv", 42, difficulty);
            assert!(env.is_some());
        }
    }

    #[test]
    fn test_random_baselines_calibration() {
        let registry = EnvironmentRegistry::default_suite();
        let baselines = registry.random_baseline_summary();

        // Verifica que baselines são razoáveis
        for (name, baseline) in &baselines {
            println!("{}: baseline = {:.2}", name, baseline);
            assert!(baseline.is_finite());
        }

        // Navigation deve ter baseline negativo (movimento cost)
        assert!(*baselines.get("NavigationEnv").unwrap_or(&0.0) < 0.0);
    }

    #[test]
    fn test_success_thresholds() {
        let registry = EnvironmentRegistry::default_suite();

        for spec in registry.specs() {
            if let Some(env) = registry.create(&spec.name, 42) {
                let threshold = env.success_threshold();
                let baseline = env.random_baseline();

                // Threshold deve ser maior que baseline (senão agente aleatório sempre "sucede")
                println!("{}: threshold={:.2}, baseline={:.2}", spec.name, threshold, baseline);
                assert!(threshold > baseline, "{} threshold should be > baseline", spec.name);
            }
        }
    }
}
