//! # Módulo de Plasticidade Sináptica
//!
//! Decomposição do sistema de plasticidade em traits e structs especializados
//! para melhor modularidade e testabilidade.
//!
//! ## Componentes
//!
//! - `SynapticLearning`: Trait para algoritmos de aprendizado (STDP, Hebbiano)
//! - `ShortTermPlasticity`: Trait para STP (facilitação/depressão)
//! - `EligibilitySystem`: Trait para eligibility traces
//! - `CompetitiveNormalization`: Trait para normalização de pesos

mod eligibility;
mod short_term;
mod stdp;
mod normalization;

pub use eligibility::{EligibilitySystem, EligibilityTrace};
pub use short_term::{ShortTermPlasticity, STPState, integrate_with_stp};
pub use stdp::{STDPLearning, STDPConfig, AsymmetricSTDP, STDPStats};
pub use normalization::{
    CompetitiveNormalization, NormalizationConfig, NormalizationState,
    NormalizationStats, normalize_weights, normalize_weights_l2,
};

/// Trait base para qualquer sistema de plasticidade sináptica
pub trait SynapticPlasticity {
    /// Aplica uma atualização de plasticidade
    fn update(&mut self, timestep: i64);

    /// Reseta o estado da plasticidade
    fn reset(&mut self);

    /// Retorna estatísticas do sistema
    fn get_stats(&self) -> PlasticityStats;
}

/// Estatísticas genéricas de plasticidade
#[derive(Debug, Clone, Default)]
pub struct PlasticityStats {
    pub total_updates: u64,
    pub avg_magnitude: f64,
    pub max_magnitude: f64,
}
