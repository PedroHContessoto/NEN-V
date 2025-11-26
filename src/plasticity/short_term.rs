//! # Short-Term Plasticity (STP)
//!
//! Implementa dinâmicas de plasticidade de curto prazo baseadas no modelo
//! Tsodyks-Markram.
//!
//! ## Fundamentação
//!
//! STP modula a eficácia sináptica baseado em uso recente:
//! - Depressão: Inputs repetitivos perdem eficácia (recursos esgotados)
//! - Facilitação: Inputs em sequência rápida podem ter boost temporário
//!
//! Isso naturalmente favorece padrões estruturados sobre ruído aleatório.
//!
//! Referência: Tsodyks & Markram (1997)

use crate::constants::timing;
use crate::constants::learning;

/// Trait para sistemas de Short-Term Plasticity
pub trait ShortTermPlasticity {
    /// Aplica modulação STP aos pesos efetivos
    fn modulate_weights(&self, base_weights: &[f64]) -> Vec<f64>;

    /// Consome recursos de uma sinapse (chamado quando input ativo)
    fn consume_resources(&mut self, synapse_idx: usize, input_strength: f64);

    /// Atualiza estado do STP (recuperação de recursos)
    fn update(&mut self);

    /// Reseta estado do STP
    fn reset(&mut self);

    /// Retorna média de recursos disponíveis
    fn average_resources(&self) -> f64;
}

/// Estado completo do STP para um conjunto de sinapses
#[derive(Debug, Clone)]
pub struct STPState {
    /// Recursos sinápticos disponíveis [0.0, 1.0]
    /// Simula vesículas de neurotransmissor disponíveis
    pub resources: Vec<f64>,

    /// Fator de facilitação (aumenta com uso repetido em curto prazo)
    pub facilitation: Vec<f64>,

    /// Taxa de recuperação de recursos (timesteps para 63% de recuperação)
    pub recovery_tau: f64,

    /// Fração de recursos usados por spike pré-sináptico
    pub use_fraction: f64,

    /// Taxa de decaimento da facilitação
    pub facilitation_decay: f64,

    /// Máximo de facilitação permitido
    pub max_facilitation: f64,

    /// Contador de updates
    update_count: u64,
}

impl STPState {
    /// Cria novo estado STP
    pub fn new(num_synapses: usize) -> Self {
        Self {
            resources: vec![1.0; num_synapses],
            facilitation: vec![1.0; num_synapses],
            recovery_tau: timing::STP_RECOVERY_TAU,
            use_fraction: learning::STP_USE_FRACTION,
            facilitation_decay: learning::STP_FACILITATION_DECAY,
            max_facilitation: 2.0,
            update_count: 0,
        }
    }

    /// Cria com parâmetros personalizados
    pub fn with_params(
        num_synapses: usize,
        recovery_tau: f64,
        use_fraction: f64,
    ) -> Self {
        Self {
            resources: vec![1.0; num_synapses],
            facilitation: vec![1.0; num_synapses],
            recovery_tau: recovery_tau.max(10.0),
            use_fraction: use_fraction.clamp(0.05, 0.5),
            facilitation_decay: learning::STP_FACILITATION_DECAY,
            max_facilitation: 2.0,
            update_count: 0,
        }
    }

    /// Retorna número de sinapses
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Verifica se está vazio
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Retorna recursos de uma sinapse específica
    pub fn resource(&self, idx: usize) -> f64 {
        self.resources.get(idx).copied().unwrap_or(1.0)
    }

    /// Retorna facilitação de uma sinapse específica
    pub fn facilitation(&self, idx: usize) -> f64 {
        self.facilitation.get(idx).copied().unwrap_or(1.0)
    }

    /// Calcula modulação efetiva para uma sinapse
    pub fn effective_modulation(&self, idx: usize) -> f64 {
        self.resource(idx) * self.facilitation(idx)
    }
}

impl ShortTermPlasticity for STPState {
    fn modulate_weights(&self, base_weights: &[f64]) -> Vec<f64> {
        base_weights.iter()
            .enumerate()
            .map(|(i, &w)| w * self.effective_modulation(i))
            .collect()
    }

    fn consume_resources(&mut self, synapse_idx: usize, input_strength: f64) {
        if synapse_idx >= self.resources.len() {
            return;
        }

        // Só consome se input significativo
        if input_strength.abs() > 0.1 {
            // Consome recursos (depressão)
            self.resources[synapse_idx] *= 1.0 - self.use_fraction;

            // Aumenta facilitação temporariamente
            self.facilitation[synapse_idx] += 0.1;
            self.facilitation[synapse_idx] =
                self.facilitation[synapse_idx].min(self.max_facilitation);
        }
    }

    fn update(&mut self) {
        self.update_count += 1;

        for i in 0..self.resources.len() {
            // Recuperação de recursos (exponencial)
            let recovery = (1.0 - self.resources[i]) / self.recovery_tau;
            self.resources[i] += recovery;
            self.resources[i] = self.resources[i].min(1.0);

            // Decaimento da facilitação
            self.facilitation[i] =
                1.0 + (self.facilitation[i] - 1.0) * self.facilitation_decay;
        }
    }

    fn reset(&mut self) {
        for r in &mut self.resources {
            *r = 1.0;
        }
        for f in &mut self.facilitation {
            *f = 1.0;
        }
        self.update_count = 0;
    }

    fn average_resources(&self) -> f64 {
        if self.resources.is_empty() {
            1.0
        } else {
            self.resources.iter().sum::<f64>() / self.resources.len() as f64
        }
    }
}

/// Integra inputs com modulação STP
pub fn integrate_with_stp(
    inputs: &[f64],
    weights: &[f64],
    stp: &mut STPState,
) -> f64 {
    assert_eq!(inputs.len(), weights.len());
    assert_eq!(inputs.len(), stp.len());

    let mut potential = 0.0;

    for i in 0..inputs.len() {
        // Peso efetivo = base * recursos * facilitação
        let effective_weight = weights[i] * stp.effective_modulation(i);
        potential += inputs[i] * effective_weight;

        // Consome recursos se input ativo
        stp.consume_resources(i, inputs[i]);
    }

    potential
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stp_initialization() {
        let stp = STPState::new(5);

        assert_eq!(stp.len(), 5);

        // Recursos começam cheios
        for r in &stp.resources {
            assert_eq!(*r, 1.0);
        }

        // Facilitação começa em 1.0
        for f in &stp.facilitation {
            assert_eq!(*f, 1.0);
        }
    }

    #[test]
    fn test_resource_depletion() {
        let mut stp = STPState::new(3);

        // Input repetido várias vezes
        for _ in 0..10 {
            stp.consume_resources(0, 1.0);
        }

        // Recursos da sinapse 0 devem estar baixos
        assert!(stp.resources[0] < 0.5);

        // Recursos das outras devem estar cheios
        assert!(stp.resources[1] > 0.9);
        assert!(stp.resources[2] > 0.9);
    }

    #[test]
    fn test_resource_recovery() {
        let mut stp = STPState::new(3);

        // Depleta recursos
        for _ in 0..10 {
            stp.consume_resources(0, 1.0);
        }

        let depleted = stp.resources[0];

        // Recupera por vários timesteps
        for _ in 0..500 {
            stp.update();
        }

        // Deve ter recuperado
        assert!(stp.resources[0] > depleted);
        assert!(stp.resources[0] > 0.8);
    }

    #[test]
    fn test_facilitation() {
        let mut stp = STPState::new(3);

        // Uso repetido aumenta facilitação
        stp.consume_resources(0, 1.0);

        assert!(stp.facilitation[0] > 1.0);
    }

    #[test]
    fn test_integration_with_stp() {
        let mut stp = STPState::new(3);
        let inputs = vec![1.0, 0.5, 0.0];
        let weights = vec![0.5, 0.5, 0.5];

        let potential = integrate_with_stp(&inputs, &weights, &mut stp);

        // Potencial deve ser positivo
        assert!(potential > 0.0);

        // Recursos da sinapse 0 devem ter sido consumidos
        assert!(stp.resources[0] < 1.0);
    }
}
