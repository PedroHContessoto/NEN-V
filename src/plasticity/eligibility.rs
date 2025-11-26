//! # Sistema de Eligibility Traces
//!
//! Implementa traces de elegibilidade para 3-factor learning.
//!
//! ## Fundamentação
//!
//! Eligibility traces "lembram" quais sinapses estavam ativas recentemente,
//! permitindo que reward tardio ainda afete as sinapses corretas.
//!
//! Referência: Izhikevich (2007) - "Solving the Distal Reward Problem"

use crate::constants::timing;
use crate::constants::learning;

/// Trait para sistemas de eligibility traces
pub trait EligibilitySystem {
    /// Atualiza traces baseado em atividade pré/pós sináptica
    fn update_traces(&mut self, pre_active: &[f64], post_fired: bool);

    /// Aplica aprendizado modulado por reward usando traces
    fn apply_reward_modulated(&mut self, weights: &mut [f64], reward: f64, modulation: f64);

    /// Retorna soma total dos traces (para diagnóstico)
    fn total_eligibility(&self) -> f64;

    /// Decai todos os traces
    fn decay(&mut self);

    /// Reseta todos os traces
    fn reset(&mut self);
}

/// Implementação concreta de eligibility traces
#[derive(Debug, Clone)]
pub struct EligibilityTrace {
    /// Traces para cada sinapse [0.0, 1.0]
    traces: Vec<f64>,

    /// Constante de tempo do trace (em timesteps)
    pub tau: f64,

    /// Incremento por correlação pré-pós
    pub increment: f64,

    /// Contador de atualizações
    update_count: u64,
}

impl EligibilityTrace {
    /// Cria novo sistema de eligibility traces
    pub fn new(num_synapses: usize) -> Self {
        Self {
            traces: vec![0.0; num_synapses],
            tau: timing::ELIGIBILITY_TRACE_TAU,
            increment: learning::TRACE_INCREMENT,
            update_count: 0,
        }
    }

    /// Cria com parâmetros personalizados
    pub fn with_params(num_synapses: usize, tau: f64, increment: f64) -> Self {
        Self {
            traces: vec![0.0; num_synapses],
            tau: tau.max(10.0),
            increment: increment.clamp(0.01, 0.5),
            update_count: 0,
        }
    }

    /// Retorna referência aos traces
    pub fn traces(&self) -> &[f64] {
        &self.traces
    }

    /// Retorna referência mutável aos traces
    pub fn traces_mut(&mut self) -> &mut [f64] {
        &mut self.traces
    }

    /// Retorna número de sinapses
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Verifica se está vazio
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Incrementa trace de uma sinapse específica
    pub fn increment_trace(&mut self, synapse_idx: usize, amount: f64) {
        if synapse_idx < self.traces.len() {
            self.traces[synapse_idx] += amount;
            self.traces[synapse_idx] = self.traces[synapse_idx].min(1.0);
        }
    }

    /// Consome parcialmente um trace (após uso)
    pub fn consume_trace(&mut self, synapse_idx: usize, factor: f64) {
        if synapse_idx < self.traces.len() {
            self.traces[synapse_idx] *= factor;
        }
    }
}

impl EligibilitySystem for EligibilityTrace {
    fn update_traces(&mut self, pre_active: &[f64], post_fired: bool) {
        assert_eq!(pre_active.len(), self.traces.len());
        self.update_count += 1;

        for i in 0..self.traces.len() {
            // Decaimento exponencial do trace
            self.traces[i] *= (-1.0 / self.tau).exp();

            // Incrementa trace se pré estava ativo E pós disparou (correlação Hebbiana)
            if pre_active[i].abs() > 0.1 && post_fired {
                self.traces[i] += self.increment;
                self.traces[i] = self.traces[i].min(1.0);
            }

            // Incremento menor se só pré ativo (permite algum crédito sem pós)
            if pre_active[i].abs() > 0.5 && !post_fired {
                self.traces[i] += self.increment * 0.1;
                self.traces[i] = self.traces[i].min(1.0);
            }
        }
    }

    fn apply_reward_modulated(&mut self, weights: &mut [f64], reward: f64, modulation: f64) {
        let effective_lr = learning::BASE_LEARNING_RATE * modulation;

        for i in 0..weights.len().min(self.traces.len()) {
            if self.traces[i] > 0.01 {
                // Δw = η * reward * trace
                let delta_w = effective_lr * reward * self.traces[i];
                weights[i] += delta_w;
                weights[i] = weights[i].clamp(0.0, learning::WEIGHT_CLAMP);

                // Consome parcialmente o trace (evita re-uso infinito)
                self.traces[i] *= 0.5;
            }
        }
    }

    fn total_eligibility(&self) -> f64 {
        self.traces.iter().sum()
    }

    fn decay(&mut self) {
        let decay_factor = (-1.0 / self.tau).exp();
        for trace in &mut self.traces {
            *trace *= decay_factor;
            if *trace < 1e-6 {
                *trace = 0.0;
            }
        }
    }

    fn reset(&mut self) {
        for trace in &mut self.traces {
            *trace = 0.0;
        }
        self.update_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eligibility_trace_creation() {
        let et = EligibilityTrace::new(10);
        assert_eq!(et.len(), 10);
        assert_eq!(et.total_eligibility(), 0.0);
    }

    #[test]
    fn test_trace_update_with_correlation() {
        let mut et = EligibilityTrace::new(5);

        // Inputs ativos
        let inputs = vec![1.0, 0.0, 1.0, 0.0, 0.5];

        // Atualiza com post_fired = true
        et.update_traces(&inputs, true);

        // Traces devem aumentar para inputs ativos
        assert!(et.traces[0] > 0.0);
        assert!(et.traces[2] > 0.0);

        // Trace para input inativo deve ser ~0
        assert!(et.traces[1] < 0.01);
    }

    #[test]
    fn test_reward_modulated_learning() {
        let mut et = EligibilityTrace::new(3);
        et.traces = vec![0.5, 0.0, 0.3];

        let mut weights = vec![0.1, 0.1, 0.1];
        let initial_w0 = weights[0];
        let initial_w1 = weights[1];

        // Aplica reward positivo
        et.apply_reward_modulated(&mut weights, 1.0, 1.0);

        // Peso 0 deve aumentar (trace alto, reward positivo)
        assert!(weights[0] > initial_w0);

        // Peso 1 não deve mudar (trace zero)
        assert!((weights[1] - initial_w1).abs() < 1e-6);
    }

    #[test]
    fn test_trace_decay() {
        // Usa tau pequeno para teste rápido de decaimento
        let mut et = EligibilityTrace::with_params(3, 20.0, 0.1);
        et.traces_mut()[0] = 1.0;
        et.traces_mut()[1] = 0.5;
        et.traces_mut()[2] = 0.2;

        let initial_sum = et.total_eligibility();

        // Decai várias vezes (com tau=20, após 100 passos: exp(-100/20) ≈ 0.007)
        for _ in 0..100 {
            et.decay();
        }

        // Soma deve ter diminuído significativamente
        assert!(et.total_eligibility() < initial_sum * 0.1);
    }
}
