//! # NEN-V: Neuromorphic Energy-based Neural Virtual Model
//!
//! Uma implementação biologicamente plausível de rede neural com mecanismos de
//! aprendizado inspirados em neurociência, incluindo STDP, homeostase e consolidação
//! de memória durante o sono.
//!
//! ## Módulos Core
//!
//! - **nenv**: Neurônio individual com dendritoma, glia e axônio
//! - **dendritoma**: Sistema dendrítico com aprendizado sináptico (STDP/Hebbiano)
//! - **glia**: Modulação metabólica e energética
//! - **network**: Orquestração de múltiplos neurônios e dinâmicas de rede
//!
//! ## Exemplo de Uso
//!
//! ```rust,no_run
//! use nenv_visual_sim::network::{Network, ConnectivityType, LearningMode};
//!
//! let mut net = Network::new(
//!     20,                            // 20 neurônios
//!     ConnectivityType::FullyConnected,
//!     0.2,                           // 20% inibitórios
//!     0.15,                          // Threshold de disparo
//! );
//!
//! net.set_learning_mode(LearningMode::STDP);
//! net.set_weight_decay(0.002);
//!
//! // Loop de simulação
//! let inputs = vec![0.0; 20];
//! net.update(&inputs);
//! ```

pub mod dendritoma;
pub mod glia;
pub mod nenv;
pub mod network;

// Re-exporta tipos comuns para conveniência
pub use dendritoma::Dendritoma;
pub use glia::Glia;
pub use nenv::{NeuronType, SpikeOrigin, NENV};
pub use network::{ConnectivityType, LearningMode, Network, NetworkState};
