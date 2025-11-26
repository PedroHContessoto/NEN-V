//! Sistema de Neuromodulação
//!
//! Implementa neuromoduladores que afetam globalmente o comportamento da rede:
//! - **Dopamina**: Sinal de reward, modula STDP e eligibility traces
//! - **Norepinefrina**: Alerta/arousal, afeta gain e threshold
//! - **Acetilcolina**: Atenção, modula plasticidade
//! - **Serotonina**: Humor/baseline, afeta exploração vs exploitation
//!
//! ## Uso
//!
//! ```rust,ignore
//! let mut nm_system = NeuromodulationSystem::new();
//!
//! // Quando há reward
//! nm_system.release(NeuromodulatorType::Dopamine, 1.0);
//!
//! // Quando há novidade/perigo
//! nm_system.release(NeuromodulatorType::Norepinephrine, 0.5);
//!
//! // A cada timestep
//! nm_system.update();
//!
//! // Obtém níveis para modular rede
//! let da_level = nm_system.get_level(NeuromodulatorType::Dopamine);
//! ```

use std::collections::HashMap;

/// Tipos de neuromoduladores disponíveis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeuromodulatorType {
    /// Dopamina: reward prediction error, reforço
    Dopamine,

    /// Norepinefrina: arousal, alerta, fight-or-flight
    Norepinephrine,

    /// Acetilcolina: atenção, foco, plasticidade cortical
    Acetylcholine,

    /// Serotonina: humor, saciedade, paciÃªncia temporal
    Serotonin,
}

/// Um neuromodulador individual
#[derive(Debug, Clone)]
pub struct Neuromodulator {
    /// Tipo do neuromodulador
    pub nm_type: NeuromodulatorType,

    /// Nível atual [0.0, 2.0] onde 1.0 é baseline
    pub level: f64,

    /// Nível baseline (equilíbrio)
    pub baseline: f64,

    /// Taxa de decaimento para baseline
    pub decay_rate: f64,

    /// Taxa de liberação máxima por evento
    pub max_release: f64,

    /// Acumulador de liberação recente (para saturação)
    recent_release: f64,
}

impl Neuromodulator {
    /// Cria um novo neuromodulador
    pub fn new(nm_type: NeuromodulatorType) -> Self {
        let (baseline, decay_rate, max_release) = match nm_type {
            NeuromodulatorType::Dopamine => (0.5, 0.05, 1.5),
            NeuromodulatorType::Norepinephrine => (0.3, 0.08, 1.2),
            NeuromodulatorType::Acetylcholine => (0.5, 0.03, 1.0),
            NeuromodulatorType::Serotonin => (0.7, 0.02, 0.8),
        };

        Self {
            nm_type,
            level: baseline,
            baseline,
            decay_rate,
            max_release,
            recent_release: 0.0,
        }
    }

    /// Libera o neuromodulador
    ///
    /// # Argumentos
    /// * `amount` - Quantidade a liberar [0.0, 1.0]
    pub fn release(&mut self, amount: f64) {
        let clamped_amount = amount.clamp(0.0, 1.0);

        // Saturação: liberações recentes reduzem eficácia
        let saturation_factor = 1.0 / (1.0 + self.recent_release);
        let effective_amount = clamped_amount * self.max_release * saturation_factor;

        self.level += effective_amount;
        self.level = self.level.min(2.0);

        self.recent_release += clamped_amount;
    }

    /// Atualiza o estado (decaimento para baseline)
    pub fn update(&mut self) {
        // Decaimento exponencial para baseline
        let diff = self.level - self.baseline;
        self.level -= diff * self.decay_rate;

        // Decaimento da saturação
        self.recent_release *= 0.9;
    }

    /// Retorna nível normalizado [-1.0, 1.0] relativo ao baseline
    pub fn normalized_level(&self) -> f64 {
        (self.level - self.baseline) / self.baseline
    }
}

/// Sistema completo de neuromodulação
#[derive(Debug, Clone)]
pub struct NeuromodulationSystem {
    /// Mapa de neuromoduladores
    modulators: HashMap<NeuromodulatorType, Neuromodulator>,

    /// Histórico de dopamina para TD-learning
    dopamine_history: Vec<f64>,

    /// Tamanho máximo do histórico
    history_size: usize,

    /// Reward prediction (para calcular RPE)
    reward_prediction: f64,

    /// Taxa de aprendizado do preditor de reward
    prediction_lr: f64,
}

impl NeuromodulationSystem {
    /// Cria um novo sistema de neuromodulação
    pub fn new() -> Self {
        let mut modulators = HashMap::new();

        modulators.insert(
            NeuromodulatorType::Dopamine,
            Neuromodulator::new(NeuromodulatorType::Dopamine),
        );
        modulators.insert(
            NeuromodulatorType::Norepinephrine,
            Neuromodulator::new(NeuromodulatorType::Norepinephrine),
        );
        modulators.insert(
            NeuromodulatorType::Acetylcholine,
            Neuromodulator::new(NeuromodulatorType::Acetylcholine),
        );
        modulators.insert(
            NeuromodulatorType::Serotonin,
            Neuromodulator::new(NeuromodulatorType::Serotonin),
        );

        Self {
            modulators,
            dopamine_history: Vec::new(),
            history_size: 100,
            reward_prediction: 0.0,
            prediction_lr: 0.1,
        }
    }

    /// Libera um neuromodulador específico
    pub fn release(&mut self, nm_type: NeuromodulatorType, amount: f64) {
        if let Some(nm) = self.modulators.get_mut(&nm_type) {
            nm.release(amount);
        }
    }

    /// Processa sinal de reward e calcula RPE (Reward Prediction Error)
    ///
    /// RPE = reward_atual - reward_esperado
    ///
    /// Libera dopamina proporcional ao RPE:
    /// - RPE > 0: reward maior que esperado â†’ dopamina alta
    /// - RPE = 0: reward como esperado â†’ dopamina baseline
    /// - RPE < 0: reward menor que esperado â†’ dopamina baixa
    pub fn process_reward(&mut self, actual_reward: f64) -> f64 {
        // Calcula RPE
        let rpe = actual_reward - self.reward_prediction;

        // Atualiza predição (TD-learning simples)
        self.reward_prediction += self.prediction_lr * rpe;

        // Libera dopamina baseado no RPE
        // RPE positivo â†’ liberação, RPE negativo â†’ supressão
        if rpe > 0.0 {
            self.release(NeuromodulatorType::Dopamine, rpe.min(1.0));
        } else {
            // Suprime dopamina abaixo do baseline
            if let Some(da) = self.modulators.get_mut(&NeuromodulatorType::Dopamine) {
                da.level = (da.level + rpe * 0.5).max(0.0);
            }
        }

        // Armazena no histórico
        self.dopamine_history.push(actual_reward);
        if self.dopamine_history.len() > self.history_size {
            self.dopamine_history.remove(0);
        }

        rpe
    }

    /// Processa evento de novidade/alerta
    pub fn process_novelty(&mut self, novelty_level: f64) {
        // Novidade libera norepinefrina e acetilcolina
        self.release(NeuromodulatorType::Norepinephrine, novelty_level * 0.8);
        self.release(NeuromodulatorType::Acetylcholine, novelty_level * 0.5);
    }

    /// Atualiza todos os neuromoduladores
    pub fn update(&mut self) {
        for nm in self.modulators.values_mut() {
            nm.update();
        }
    }

    /// Obtém o nível de um neuromodulador
    pub fn get_level(&self, nm_type: NeuromodulatorType) -> f64 {
        self.modulators
            .get(&nm_type)
            .map(|nm| nm.level)
            .unwrap_or(0.5)
    }

    /// Obtém nível normalizado [-1, 1]
    pub fn get_normalized_level(&self, nm_type: NeuromodulatorType) -> f64 {
        self.modulators
            .get(&nm_type)
            .map(|nm| nm.normalized_level())
            .unwrap_or(0.0)
    }

    /// Calcula fator de modulação de plasticidade
    ///
    /// Combina dopamina e acetilcolina para modular aprendizado
    pub fn plasticity_modulation(&self) -> f64 {
        let da = self.get_level(NeuromodulatorType::Dopamine);
        let ach = self.get_level(NeuromodulatorType::Acetylcholine);

        // Plasticidade aumenta com dopamina E acetilcolina
        // Baseline (ambos em 0.5) â†’ modulação = 1.0
        let da_factor = 0.5 + da;       // [0.5, 2.5]
        let ach_factor = 0.5 + ach;     // [0.5, 2.5]

        // Combinação multiplicativa normalizada
        (da_factor * ach_factor) / 1.0 // Normaliza para ~1.0 no baseline
    }

    /// Calcula fator de exploração vs exploitation
    ///
    /// Baixa serotonina â†’ mais exploração
    /// Alta serotonina â†’ mais exploitation
    pub fn exploration_factor(&self) -> f64 {
        let sero = self.get_level(NeuromodulatorType::Serotonin);

        // Exploração diminui com serotonina
        // sero = 0.0 â†’ exploration = 1.0
        // sero = 1.0 â†’ exploration = 0.3
        1.0 - sero * 0.7
    }

    /// Calcula fator de arousal/alerta
    pub fn arousal_factor(&self) -> f64 {
        let ne = self.get_level(NeuromodulatorType::Norepinephrine);

        // Arousal aumenta linearmente com norepinefrina
        0.5 + ne
    }

    /// Retorna estatísticas do sistema
    pub fn get_stats(&self) -> NeuromodulationStats {
        NeuromodulationStats {
            dopamine: self.get_level(NeuromodulatorType::Dopamine),
            norepinephrine: self.get_level(NeuromodulatorType::Norepinephrine),
            acetylcholine: self.get_level(NeuromodulatorType::Acetylcholine),
            serotonin: self.get_level(NeuromodulatorType::Serotonin),
            reward_prediction: self.reward_prediction,
            plasticity_mod: self.plasticity_modulation(),
            exploration: self.exploration_factor(),
            arousal: self.arousal_factor(),
        }
    }

    /// Reseta todos os neuromoduladores para baseline
    pub fn reset(&mut self) {
        for nm in self.modulators.values_mut() {
            nm.level = nm.baseline;
            nm.recent_release = 0.0;
        }
        self.reward_prediction = 0.0;
        self.dopamine_history.clear();
    }
}

impl Default for NeuromodulationSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Estatísticas do sistema de neuromodulação
#[derive(Debug, Clone)]
pub struct NeuromodulationStats {
    pub dopamine: f64,
    pub norepinephrine: f64,
    pub acetylcholine: f64,
    pub serotonin: f64,
    pub reward_prediction: f64,
    pub plasticity_mod: f64,
    pub exploration: f64,
    pub arousal: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromodulator_baseline() {
        let da = Neuromodulator::new(NeuromodulatorType::Dopamine);
        assert_eq!(da.level, da.baseline);
    }

    #[test]
    fn test_neuromodulator_release() {
        let mut da = Neuromodulator::new(NeuromodulatorType::Dopamine);
        let initial = da.level;

        da.release(0.5);

        assert!(da.level > initial);
    }

    #[test]
    fn test_neuromodulator_decay() {
        let mut da = Neuromodulator::new(NeuromodulatorType::Dopamine);
        da.release(1.0);

        let after_release = da.level;

        // Vários updates
        for _ in 0..100 {
            da.update();
        }

        // Deve ter decaído em direção ao baseline
        assert!(da.level < after_release);
        assert!((da.level - da.baseline).abs() < 0.1);
    }

    #[test]
    fn test_neuromodulator_saturation() {
        let mut da = Neuromodulator::new(NeuromodulatorType::Dopamine);

        let first_release_level = {
            da.release(0.5);
            da.level
        };

        // Segunda liberação em sequÃªncia deve ser menos efetiva
        let before_second = da.level;
        da.release(0.5);
        let second_increase = da.level - before_second;

        // Primeiro aumento
        let first_increase = first_release_level - da.baseline;

        // Segundo deve ser menor devido Ã  saturação
        assert!(second_increase < first_increase);
    }

    #[test]
    fn test_system_reward_processing() {
        let mut system = NeuromodulationSystem::new();

        // Reward inesperado (maior que predição inicial de 0)
        let rpe = system.process_reward(1.0);

        assert!(rpe > 0.0);
        assert!(system.get_level(NeuromodulatorType::Dopamine) > 0.5);
    }

    #[test]
    fn test_system_plasticity_modulation() {
        let mut system = NeuromodulationSystem::new();

        // Baseline
        let baseline_mod = system.plasticity_modulation();

        // Libera dopamina e acetilcolina
        system.release(NeuromodulatorType::Dopamine, 1.0);
        system.release(NeuromodulatorType::Acetylcholine, 1.0);

        let high_mod = system.plasticity_modulation();

        // Modulação deve ser maior
        assert!(high_mod > baseline_mod);
    }

    #[test]
    fn test_system_exploration() {
        let mut system = NeuromodulationSystem::new();

        let normal_exploration = system.exploration_factor();

        // Aumenta serotonina
        system.release(NeuromodulatorType::Serotonin, 1.0);

        let low_exploration = system.exploration_factor();

        // Exploração deve diminuir com serotonina alta
        assert!(low_exploration < normal_exploration);
    }

    #[test]
    fn test_system_novelty() {
        let mut system = NeuromodulationSystem::new();

        let initial_ne = system.get_level(NeuromodulatorType::Norepinephrine);
        let initial_ach = system.get_level(NeuromodulatorType::Acetylcholine);

        system.process_novelty(0.8);

        // Ambos devem aumentar
        assert!(system.get_level(NeuromodulatorType::Norepinephrine) > initial_ne);
        assert!(system.get_level(NeuromodulatorType::Acetylcholine) > initial_ach);
    }
}