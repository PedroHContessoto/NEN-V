use ::rand::Rng;

// === ESTRUTURAS DO MUNDO ===

#[derive(Clone, Copy)]
pub struct Agent {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Copy)]
pub struct Food {
    pub x: i32,
    pub y: i32,
}

pub struct Environment {
    pub grid_size: i32,
    pub agent: Agent,
    pub food: Food,
    pub steps_current_episode: u32,
}

pub enum ActionResult {
    Moved,
    HitWall,
    AteFood,
    None,
}

impl Environment {
    pub fn new(grid_size: i32) -> Self {
        let mut rng = ::rand::thread_rng();
        let agent = Agent {
            x: grid_size / 2,
            y: grid_size / 2,
        };
        
        // Garante que comida não nasce no agente
        let mut food = Food { x: 0, y: 0 }; 
        loop {
            food.x = rng.gen_range(0..grid_size);
            food.y = rng.gen_range(0..grid_size);
            if food.x != agent.x || food.y != agent.y { break; }
        }

        Self {
            grid_size,
            agent,
            food,
            steps_current_episode: 0,
        }
    }

    pub fn get_sensor_inputs(&self) -> [f64; 4] {
        let dx = self.food.x - self.agent.x;
        let dy = self.food.y - self.agent.y;

        // Distância Manhattan (steps necessários)
        let manhattan_dist = dx.abs() + dy.abs();

        // Distância Euclidiana (distância real)
        let euclidean_dist = ((dx * dx + dy * dy) as f64).sqrt();

        // Fator de urgência: quanto mais perto, mais forte o sinal
        // Normalizado para ficar entre 0.5 (longe) e 2.5 (muito perto)
        let urgency = if euclidean_dist > 0.1 {
            2.5 - (euclidean_dist / (self.grid_size as f64 * 0.7)).min(2.0)
        } else {
            2.5 // Muito perto!
        };

        let mut sensors = [0.0; 4];

        // Sensor UP (índice 0)
        if dy < 0 {
            // Intensidade proporcional à distância naquela direção
            let dist_ratio = dy.abs() as f64 / manhattan_dist.max(1) as f64;
            sensors[0] = urgency * dist_ratio * 1.5;
        }

        // Sensor DOWN (índice 1)
        if dy > 0 {
            let dist_ratio = dy.abs() as f64 / manhattan_dist.max(1) as f64;
            sensors[1] = urgency * dist_ratio * 1.5;
        }

        // Sensor LEFT (índice 2)
        if dx < 0 {
            let dist_ratio = dx.abs() as f64 / manhattan_dist.max(1) as f64;
            sensors[2] = urgency * dist_ratio * 1.5;
        }

        // Sensor RIGHT (índice 3)
        if dx > 0 {
            let dist_ratio = dx.abs() as f64 / manhattan_dist.max(1) as f64;
            sensors[3] = urgency * dist_ratio * 1.5;
        }

        sensors
    }

    pub fn execute_motor(&mut self, motor_id: usize, motor_map: &[usize; 4]) -> ActionResult {
        let (old_x, old_y) = (self.agent.x, self.agent.y);

        // Mapeamento: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        // motor_id deve ser comparado com os IDs reais passados pelo main
        if motor_id == motor_map[0] { self.agent.y -= 1; }      // UP
        else if motor_id == motor_map[1] { self.agent.y += 1; } // DOWN
        else if motor_id == motor_map[2] { self.agent.x -= 1; } // LEFT
        else if motor_id == motor_map[3] { self.agent.x += 1; } // RIGHT
        else { return ActionResult::None; }

        // Colisão com paredes (HitWall)
        if self.agent.x < 0 || self.agent.x >= self.grid_size || 
           self.agent.y < 0 || self.agent.y >= self.grid_size {
            self.agent.x = old_x;
            self.agent.y = old_y;
            return ActionResult::HitWall;
        }

        // Comeu comida?
        if self.agent.x == self.food.x && self.agent.y == self.food.y {
            return ActionResult::AteFood;
        }

        ActionResult::Moved
    }

    pub fn respawn_food(&mut self) {
        let mut rng = ::rand::thread_rng();
        loop {
            self.food.x = rng.gen_range(0..self.grid_size);
            self.food.y = rng.gen_range(0..self.grid_size);
            if self.food.x != self.agent.x || self.food.y != self.agent.y { break; }
        }
    }

    pub fn expand_grid(&mut self) {
        self.grid_size += 1;
        self.agent.x = self.grid_size / 2;
        self.agent.y = self.grid_size / 2;
        self.respawn_food();
    }
}

// === MÉTRICAS EXPANDIDAS ===

pub struct Metrics {
    pub total_steps: u64,
    pub total_movements: u64,           // Total de movimentos (quando motor dispara)
    pub score: u32,
    pub successes: u32,
    pub steps_history: Vec<u32>,
    pub motor_fires: [u32; 4],

    // Novas métricas
    pub wall_collisions: u32,           // Total de colisões
    pub score_history: Vec<u32>,        // Histórico de score (para gráfico)
    pub recent_success_window: Vec<bool>, // Últimos 100 episódios (sucesso/falha)
    pub best_episode_steps: u32,        // Melhor performance (menos steps)
    pub worst_episode_steps: u32,       // Pior performance (mais steps)
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            total_movements: 0,
            score: 0,
            successes: 0,
            steps_history: Vec::new(),
            motor_fires: [0; 4],
            wall_collisions: 0,
            score_history: Vec::new(),
            recent_success_window: Vec::new(),
            best_episode_steps: u32::MAX,
            worst_episode_steps: 0,
        }
    }

    pub fn record_success(&mut self, steps_taken: u32) {
        self.score += 1;
        self.successes += 1;
        self.steps_history.push(steps_taken);

        // Mantém histórico limpo (últimos 50 episódios)
        if self.steps_history.len() > 50 {
            self.steps_history.remove(0);
        }

        // Atualiza melhor/pior
        if steps_taken < self.best_episode_steps {
            self.best_episode_steps = steps_taken;
        }
        if steps_taken > self.worst_episode_steps {
            self.worst_episode_steps = steps_taken;
        }

        // Score history (para gráfico)
        self.score_history.push(self.score);
        if self.score_history.len() > 100 {
            self.score_history.remove(0);
        }

        // Success window
        self.recent_success_window.push(true);
        if self.recent_success_window.len() > 100 {
            self.recent_success_window.remove(0);
        }
    }

    pub fn record_wall_hit(&mut self) {
        self.wall_collisions += 1;
    }

    pub fn record_failure(&mut self) {
        // Registra falha (timeout sem comer)
        self.recent_success_window.push(false);
        if self.recent_success_window.len() > 100 {
            self.recent_success_window.remove(0);
        }
    }

    pub fn average_steps(&self) -> f64 {
        if self.steps_history.is_empty() {
            0.0
        } else {
            self.steps_history.iter().sum::<u32>() as f64 / self.steps_history.len() as f64
        }
    }

    pub fn success_rate_recent(&self) -> f64 {
        if self.recent_success_window.is_empty() {
            0.0
        } else {
            let successes = self.recent_success_window.iter().filter(|&&x| x).count();
            (successes as f64 / self.recent_success_window.len() as f64) * 100.0
        }
    }
}