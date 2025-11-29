//! # Adaptadores para Ambientes Externos
//!
//! Este módulo permite integrar simulações existentes da pasta `simulations/`
//! ao sistema de avaliação do hyperparameter_search.
//!
//! ## Como Adicionar Novos Ambientes
//!
//! 1. Crie uma struct que implemente `Environment` para sua simulação
//! 2. Registre no `EnvironmentRegistry::with_external_environments()`
//! 3. O ambiente será automaticamente incluído na avaliação do hyperopt
//!
//! ## Exemplo
//!
//! ```rust
//! pub struct MinhaSimulacaoEnv {
//!     // campos internos
//! }
//!
//! impl Environment for MinhaSimulacaoEnv {
//!     fn reset(&mut self) -> Vec<f64> { ... }
//!     fn step(&mut self, action: usize) -> StepResult { ... }
//!     fn observation_size(&self) -> usize { ... }
//!     fn action_size(&self) -> usize { ... }
//!     fn name(&self) -> &str { "MinhaSimulacao" }
//!     fn params(&self) -> &EnvironmentParams { ... }
//! }
//! ```

use super::environments::{Environment, EnvironmentParams, StepResult};

// =============================================================================
// GRIDWORLD SENSORIMOTOR ADAPTER
// =============================================================================

/// Configuração do GridWorld Sensorimotor
#[derive(Clone, Debug)]
pub struct GridWorldConfig {
    pub grid_size: i32,
    pub max_steps: usize,
    pub success_threshold: f64,
}

impl Default for GridWorldConfig {
    fn default() -> Self {
        Self {
            grid_size: 5,
            max_steps: 100,
            success_threshold: 3.0, // 3 comidas para "sucesso"
        }
    }
}

/// Adaptador do GridWorld Sensorimotor para o sistema de hyperopt
///
/// O ambiente original usa macroquad para visualização.
/// Este adaptador extrai apenas a lógica do ambiente para avaliação headless.
pub struct GridWorldEnv {
    // Estado do ambiente
    grid_size: i32,
    agent_x: i32,
    agent_y: i32,
    food_x: i32,
    food_y: i32,

    // Contadores
    steps: usize,
    food_collected: usize,

    // Configuração
    config: GridWorldConfig,
    params: EnvironmentParams,

    // RNG
    seed: u64,
    rng_state: u64,
}

impl GridWorldEnv {
    pub fn new(seed: u64) -> Self {
        Self::with_config(GridWorldConfig::default(), seed)
    }

    pub fn with_config(config: GridWorldConfig, seed: u64) -> Self {
        let params = EnvironmentParams {
            seed,
            max_steps: config.max_steps,
            reward_scale: 1.0,
            max_episode_reward: 10.0,
            success_threshold: config.success_threshold,
            difficulty: 0.5,
            stochasticity: 0.0,
        };

        let mut env = Self {
            grid_size: config.grid_size,
            agent_x: config.grid_size / 2,
            agent_y: config.grid_size / 2,
            food_x: 0,
            food_y: 0,
            steps: 0,
            food_collected: 0,
            config,
            params,
            seed,
            rng_state: seed,
        };

        env.spawn_food();
        env
    }

    fn next_random(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    fn spawn_food(&mut self) {
        loop {
            self.food_x = (self.next_random() * self.grid_size as f64) as i32;
            self.food_y = (self.next_random() * self.grid_size as f64) as i32;
            if self.food_x != self.agent_x || self.food_y != self.agent_y {
                break;
            }
        }
    }

    fn get_observation(&self) -> Vec<f64> {
        // 4 sensores direcionais: UP, DOWN, LEFT, RIGHT
        let dx = self.food_x - self.agent_x;
        let dy = self.food_y - self.agent_y;

        let manhattan_dist = (dx.abs() + dy.abs()).max(1) as f64;
        let euclidean_dist = ((dx * dx + dy * dy) as f64).sqrt();

        // Urgency: quanto mais perto, mais forte
        let urgency = if euclidean_dist > 0.1 {
            2.5 - (euclidean_dist / (self.grid_size as f64 * 0.7)).min(2.0)
        } else {
            2.5
        };

        let mut sensors = vec![0.0; 4];

        // UP
        if dy < 0 {
            sensors[0] = urgency * (dy.abs() as f64 / manhattan_dist) * 1.5;
        }
        // DOWN
        if dy > 0 {
            sensors[1] = urgency * (dy.abs() as f64 / manhattan_dist) * 1.5;
        }
        // LEFT
        if dx < 0 {
            sensors[2] = urgency * (dx.abs() as f64 / manhattan_dist) * 1.5;
        }
        // RIGHT
        if dx > 0 {
            sensors[3] = urgency * (dx.abs() as f64 / manhattan_dist) * 1.5;
        }

        sensors
    }
}

impl Environment for GridWorldEnv {
    fn reset(&mut self) -> Vec<f64> {
        self.rng_state = self.seed;
        self.agent_x = self.grid_size / 2;
        self.agent_y = self.grid_size / 2;
        self.steps = 0;
        self.food_collected = 0;
        self.spawn_food();
        self.get_observation()
    }

    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng_state = seed;
        self.params.seed = seed;
    }

    fn step(&mut self, action: usize) -> StepResult {
        self.steps += 1;
        let (old_x, old_y) = (self.agent_x, self.agent_y);

        // Executa ação: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        match action {
            0 => self.agent_y -= 1, // UP
            1 => self.agent_y += 1, // DOWN
            2 => self.agent_x -= 1, // LEFT
            3 => self.agent_x += 1, // RIGHT
            _ => {}
        }

        let mut reward = -0.01; // Pequeno custo por movimento

        // Colisão com parede
        if self.agent_x < 0 || self.agent_x >= self.grid_size ||
           self.agent_y < 0 || self.agent_y >= self.grid_size {
            self.agent_x = old_x;
            self.agent_y = old_y;
            reward = -0.1; // Penalidade por bater na parede
        }

        // Comeu comida?
        if self.agent_x == self.food_x && self.agent_y == self.food_y {
            reward = 1.0;
            self.food_collected += 1;
            self.spawn_food();
        }

        let done = self.steps >= self.config.max_steps;

        StepResult {
            observation: self.get_observation(),
            reward,
            done,
            info: std::collections::HashMap::new(),
        }
    }

    fn observation_size(&self) -> usize { 4 }
    fn action_size(&self) -> usize { 4 }
    fn name(&self) -> &str { "GridWorldSensorimotor" }
    fn params(&self) -> &EnvironmentParams { &self.params }

    fn description(&self) -> &str {
        "GridWorld navigation - agent must follow directional sensors to find food"
    }

    fn random_baseline(&self) -> f64 {
        // Empiricamente, agente random coleta ~0.5 comidas em 100 steps
        0.5
    }
}

// =============================================================================
// REALTIME ENVIRONMENT ADAPTER
// =============================================================================

/// Configuração do ambiente realtime
#[derive(Clone, Debug)]
pub struct RealtimeEnvConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub num_food: usize,
    pub num_danger: usize,
    pub max_steps: usize,
    pub food_reward: f64,
    pub danger_penalty: f64,
    pub success_threshold: f64,
}

impl Default for RealtimeEnvConfig {
    fn default() -> Self {
        Self {
            grid_width: 10,
            grid_height: 10,
            num_food: 3,
            num_danger: 2,
            max_steps: 200,
            food_reward: 1.0,
            danger_penalty: -0.5,
            success_threshold: 3.0,
        }
    }
}

/// Tipos de célula
#[derive(Clone, Copy, PartialEq)]
enum CellType {
    Empty,
    Food,
    Danger,
}

/// Adaptador do ambiente realtime para hyperopt
pub struct RealtimeEnv {
    // Grid
    width: usize,
    height: usize,
    cells: Vec<CellType>,

    // Agente
    agent_x: usize,
    agent_y: usize,

    // Estado
    steps: usize,
    total_reward: f64,

    // Config
    config: RealtimeEnvConfig,
    params: EnvironmentParams,

    // RNG
    seed: u64,
    rng_state: u64,
}

impl RealtimeEnv {
    pub fn new(seed: u64) -> Self {
        Self::with_config(RealtimeEnvConfig::default(), seed)
    }

    pub fn with_config(config: RealtimeEnvConfig, seed: u64) -> Self {
        let params = EnvironmentParams {
            seed,
            max_steps: config.max_steps,
            reward_scale: 1.0,
            max_episode_reward: config.num_food as f64 * config.food_reward * 2.0,
            success_threshold: config.success_threshold,
            difficulty: 0.5,
            stochasticity: 0.1,
        };

        let mut env = Self {
            width: config.grid_width,
            height: config.grid_height,
            cells: vec![CellType::Empty; config.grid_width * config.grid_height],
            agent_x: config.grid_width / 2,
            agent_y: config.grid_height / 2,
            steps: 0,
            total_reward: 0.0,
            config,
            params,
            seed,
            rng_state: seed,
        };

        env.populate_grid();
        env
    }

    fn next_random(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    fn cell_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn populate_grid(&mut self) {
        // Limpa grid
        self.cells.fill(CellType::Empty);

        // Coloca comida
        for _ in 0..self.config.num_food {
            loop {
                let x = (self.next_random() * self.width as f64) as usize;
                let y = (self.next_random() * self.height as f64) as usize;
                let idx = self.cell_index(x, y);
                if self.cells[idx] == CellType::Empty && (x != self.agent_x || y != self.agent_y) {
                    self.cells[idx] = CellType::Food;
                    break;
                }
            }
        }

        // Coloca perigos
        for _ in 0..self.config.num_danger {
            loop {
                let x = (self.next_random() * self.width as f64) as usize;
                let y = (self.next_random() * self.height as f64) as usize;
                let idx = self.cell_index(x, y);
                if self.cells[idx] == CellType::Empty && (x != self.agent_x || y != self.agent_y) {
                    self.cells[idx] = CellType::Danger;
                    break;
                }
            }
        }
    }

    fn get_observation(&self) -> Vec<f64> {
        // Observação: 8 sensores (raycasts em 8 direções)
        // Cada sensor retorna: [distância_comida, distância_perigo, distância_parede]
        // Total: 8 direções * 3 tipos = 24 sensores

        let directions = [
            (0, -1),  // N
            (1, -1),  // NE
            (1, 0),   // E
            (1, 1),   // SE
            (0, 1),   // S
            (-1, 1),  // SW
            (-1, 0),  // W
            (-1, -1), // NW
        ];

        let mut obs = vec![0.0; 24];
        let max_dist = (self.width.max(self.height)) as f64;

        for (dir_idx, (dx, dy)) in directions.iter().enumerate() {
            let mut x = self.agent_x as i32;
            let mut y = self.agent_y as i32;
            let mut dist = 0.0;

            let mut found_food = false;
            let mut found_danger = false;
            let mut found_wall = false;

            loop {
                x += dx;
                y += dy;
                dist += 1.0;

                // Parede
                if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
                    if !found_wall {
                        obs[dir_idx * 3 + 2] = 1.0 - (dist / max_dist); // Wall sensor
                        found_wall = true;
                    }
                    break;
                }

                let idx = self.cell_index(x as usize, y as usize);
                match self.cells[idx] {
                    CellType::Food if !found_food => {
                        obs[dir_idx * 3] = 1.0 - (dist / max_dist); // Food sensor
                        found_food = true;
                    }
                    CellType::Danger if !found_danger => {
                        obs[dir_idx * 3 + 1] = 1.0 - (dist / max_dist); // Danger sensor
                        found_danger = true;
                    }
                    _ => {}
                }

                if found_food && found_danger && found_wall {
                    break;
                }
            }
        }

        obs
    }
}

impl Environment for RealtimeEnv {
    fn reset(&mut self) -> Vec<f64> {
        self.rng_state = self.seed;
        self.agent_x = self.width / 2;
        self.agent_y = self.height / 2;
        self.steps = 0;
        self.total_reward = 0.0;
        self.populate_grid();
        self.get_observation()
    }

    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng_state = seed;
        self.params.seed = seed;
    }

    fn step(&mut self, action: usize) -> StepResult {
        self.steps += 1;

        // Ações: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        let directions = [
            (0, -1),  // N
            (1, -1),  // NE
            (1, 0),   // E
            (1, 1),   // SE
            (0, 1),   // S
            (-1, 1),  // SW
            (-1, 0),  // W
            (-1, -1), // NW
        ];

        let (dx, dy) = directions.get(action).copied().unwrap_or((0, 0));

        let new_x = (self.agent_x as i32 + dx).clamp(0, self.width as i32 - 1) as usize;
        let new_y = (self.agent_y as i32 + dy).clamp(0, self.height as i32 - 1) as usize;

        self.agent_x = new_x;
        self.agent_y = new_y;

        let idx = self.cell_index(new_x, new_y);
        let reward = match self.cells[idx] {
            CellType::Food => {
                self.cells[idx] = CellType::Empty;
                // Respawn comida em outro lugar
                loop {
                    let x = (self.next_random() * self.width as f64) as usize;
                    let y = (self.next_random() * self.height as f64) as usize;
                    let new_idx = self.cell_index(x, y);
                    if self.cells[new_idx] == CellType::Empty && (x != self.agent_x || y != self.agent_y) {
                        self.cells[new_idx] = CellType::Food;
                        break;
                    }
                }
                self.config.food_reward
            }
            CellType::Danger => self.config.danger_penalty,
            CellType::Empty => -0.01, // Pequeno custo por movimento
        };

        self.total_reward += reward;
        let done = self.steps >= self.config.max_steps;

        StepResult {
            observation: self.get_observation(),
            reward,
            done,
            info: std::collections::HashMap::new(),
        }
    }

    fn observation_size(&self) -> usize { 24 } // 8 direções * 3 tipos
    fn action_size(&self) -> usize { 8 }       // 8 direções de movimento
    fn name(&self) -> &str { "RealtimeNavigation" }
    fn params(&self) -> &EnvironmentParams { &self.params }

    fn description(&self) -> &str {
        "Complex navigation with food, danger, and 8-directional movement"
    }

    fn random_baseline(&self) -> f64 {
        // Empiricamente, agente random score ~0
        0.0
    }
}

// =============================================================================
// REGISTRO DE AMBIENTES EXTERNOS
// =============================================================================

use super::environments::{EnvironmentRegistry, EnvironmentSpec};

/// Extensão do EnvironmentRegistry para incluir ambientes externos
pub trait ExternalEnvironments {
    /// Adiciona ambientes da pasta simulations/ ao registry
    fn with_external_environments(self) -> Self;

    /// Cria ambiente externo por nome
    fn create_external(&self, name: &str, seed: u64, difficulty: f64) -> Option<Box<dyn Environment>>;
}

impl ExternalEnvironments for EnvironmentRegistry {
    fn with_external_environments(mut self) -> Self {
        // GridWorld Sensorimotor
        self.register(
            EnvironmentSpec::new("GridWorldSensorimotor", 0.20, 30)
                .with_description("GridWorld navigation from simulations/gridworld_sensorimotor")
                .with_difficulty(0.4)
        );

        // Realtime Navigation
        self.register(
            EnvironmentSpec::new("RealtimeNavigation", 0.20, 25)
                .with_description("Complex navigation from simulations/realtime_environment")
                .with_difficulty(0.6)
        );

        self
    }

    fn create_external(&self, name: &str, seed: u64, difficulty: f64) -> Option<Box<dyn Environment>> {
        match name {
            "GridWorldSensorimotor" => {
                let config = GridWorldConfig {
                    grid_size: (3.0 + difficulty * 7.0) as i32, // 3-10
                    max_steps: (50.0 + difficulty * 150.0) as usize, // 50-200
                    success_threshold: 2.0 + difficulty * 3.0, // 2-5
                };
                Some(Box::new(GridWorldEnv::with_config(config, seed)))
            }
            "RealtimeNavigation" => {
                let config = RealtimeEnvConfig {
                    grid_width: (8.0 + difficulty * 7.0) as usize,  // 8-15
                    grid_height: (8.0 + difficulty * 7.0) as usize, // 8-15
                    num_food: (2.0 + difficulty * 3.0) as usize,    // 2-5
                    num_danger: (1.0 + difficulty * 4.0) as usize,  // 1-5
                    max_steps: (100.0 + difficulty * 200.0) as usize, // 100-300
                    success_threshold: 2.0 + difficulty * 4.0,      // 2-6
                    ..Default::default()
                };
                Some(Box::new(RealtimeEnv::with_config(config, seed)))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gridworld_env() {
        let mut env = GridWorldEnv::new(42);
        let obs = env.reset();

        assert_eq!(obs.len(), 4);
        assert_eq!(env.action_size(), 4);

        // Testa alguns steps
        for action in 0..4 {
            let result = env.step(action);
            assert_eq!(result.observation.len(), 4);
        }
    }

    #[test]
    fn test_realtime_env() {
        let mut env = RealtimeEnv::new(42);
        let obs = env.reset();

        assert_eq!(obs.len(), 24);
        assert_eq!(env.action_size(), 8);

        // Testa alguns steps
        for action in 0..8 {
            let result = env.step(action);
            assert_eq!(result.observation.len(), 24);
        }
    }

    #[test]
    fn test_registry_with_external() {
        let registry = EnvironmentRegistry::default_suite().with_external_environments();

        // Deve ter mais ambientes agora
        assert!(registry.specs().len() >= 6); // 4 originais + 2 externos

        // Deve conseguir criar ambientes externos
        let gridworld = registry.create_external("GridWorldSensorimotor", 42, 0.5);
        assert!(gridworld.is_some());

        let realtime = registry.create_external("RealtimeNavigation", 42, 0.5);
        assert!(realtime.is_some());
    }
}
