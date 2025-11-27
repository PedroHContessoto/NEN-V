//! # Definição do Espaço de Parâmetros
//!
//! Define todos os parâmetros otimizáveis da rede neural com seus ranges válidos.

use std::collections::HashMap;

/// Tipos de distribuição para amostragem
#[derive(Debug, Clone)]
pub enum SamplingDistribution {
    /// Uniforme linear
    Uniform,
    /// Log-uniforme (para parâmetros que variam em ordens de magnitude)
    LogUniform,
    /// Normal com média e desvio padrão
    Normal { mean: f64, std: f64 },
    /// Categórico (escolha entre opções discretas)
    Categorical,
}

/// Range de valores para um parâmetro
#[derive(Debug, Clone)]
pub enum ParameterRange {
    /// Valor contínuo com min/max
    Continuous {
        min: f64,
        max: f64,
        distribution: SamplingDistribution,
    },
    /// Valor inteiro com min/max
    Integer {
        min: i64,
        max: i64,
    },
    /// Valor booleano
    Boolean,
    /// Escolha categórica
    Categorical {
        options: Vec<String>,
    },
}

impl ParameterRange {
    pub fn continuous(min: f64, max: f64) -> Self {
        Self::Continuous {
            min,
            max,
            distribution: SamplingDistribution::Uniform,
        }
    }

    pub fn log_continuous(min: f64, max: f64) -> Self {
        Self::Continuous {
            min,
            max,
            distribution: SamplingDistribution::LogUniform,
        }
    }

    pub fn integer(min: i64, max: i64) -> Self {
        Self::Integer { min, max }
    }

    pub fn boolean() -> Self {
        Self::Boolean
    }

    pub fn categorical(options: Vec<&str>) -> Self {
        Self::Categorical {
            options: options.into_iter().map(String::from).collect(),
        }
    }

    /// Amostra um valor aleatório do range
    pub fn sample(&self, rng: &mut impl FnMut() -> f64) -> ParameterValue {
        match self {
            Self::Continuous { min, max, distribution } => {
                let value = match distribution {
                    SamplingDistribution::Uniform => {
                        min + rng() * (max - min)
                    }
                    SamplingDistribution::LogUniform => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        (log_min + rng() * (log_max - log_min)).exp()
                    }
                    SamplingDistribution::Normal { mean, std } => {
                        // Box-Muller transform
                        let u1 = rng();
                        let u2 = rng();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        (mean + std * z).clamp(*min, *max)
                    }
                    SamplingDistribution::Categorical => {
                        min + rng() * (max - min)
                    }
                };
                ParameterValue::Float(value)
            }
            Self::Integer { min, max } => {
                let value = *min + (rng() * (*max - *min + 1) as f64) as i64;
                ParameterValue::Int(value.min(*max))
            }
            Self::Boolean => {
                ParameterValue::Bool(rng() > 0.5)
            }
            Self::Categorical { options } => {
                let idx = (rng() * options.len() as f64) as usize;
                ParameterValue::String(options[idx.min(options.len() - 1)].clone())
            }
        }
    }
}

/// Valor de um parâmetro
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
}

impl ParameterValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
}

/// Definição de um parâmetro
#[derive(Debug, Clone)]
pub struct ParameterDef {
    /// Nome do parâmetro
    pub name: String,
    /// Categoria/grupo do parâmetro
    pub category: String,
    /// Descrição
    pub description: String,
    /// Range de valores válidos
    pub range: ParameterRange,
    /// Valor default
    pub default: ParameterValue,
    /// Importância estimada (0.0 - 1.0)
    pub importance: f64,
    /// Dependências de outros parâmetros
    pub dependencies: Vec<String>,
}

impl ParameterDef {
    pub fn new(
        name: &str,
        category: &str,
        description: &str,
        range: ParameterRange,
        default: ParameterValue,
    ) -> Self {
        Self {
            name: name.to_string(),
            category: category.to_string(),
            description: description.to_string(),
            range,
            default,
            importance: 0.5,
            dependencies: Vec::new(),
        }
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance;
        self
    }

    pub fn with_dependencies(mut self, deps: Vec<&str>) -> Self {
        self.dependencies = deps.into_iter().map(String::from).collect();
        self
    }
}

/// Espaço completo de parâmetros
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    /// Definições de parâmetros
    pub parameters: HashMap<String, ParameterDef>,
    /// Ordem de prioridade para otimização
    pub priority_order: Vec<String>,
}

impl ParameterSpace {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            priority_order: Vec::new(),
        }
    }

    pub fn add_parameter(&mut self, param: ParameterDef) {
        let name = param.name.clone();
        self.parameters.insert(name.clone(), param);
        self.priority_order.push(name);
    }

    /// Amostra uma configuração completa
    pub fn sample(&self, rng: &mut impl FnMut() -> f64) -> HashMap<String, ParameterValue> {
        self.parameters
            .iter()
            .map(|(name, def)| (name.clone(), def.range.sample(rng)))
            .collect()
    }

    /// Gera grid de valores para busca em grid
    pub fn generate_grid(&self, points_per_param: usize) -> Vec<HashMap<String, ParameterValue>> {
        let mut grid = vec![HashMap::new()];

        for (name, def) in &self.parameters {
            let values = self.generate_param_values(def, points_per_param);
            let mut new_grid = Vec::new();

            for config in &grid {
                for value in &values {
                    let mut new_config = config.clone();
                    new_config.insert(name.clone(), value.clone());
                    new_grid.push(new_config);
                }
            }

            grid = new_grid;
        }

        grid
    }

    fn generate_param_values(&self, def: &ParameterDef, n: usize) -> Vec<ParameterValue> {
        match &def.range {
            ParameterRange::Continuous { min, max, distribution } => {
                (0..n)
                    .map(|i| {
                        let t = i as f64 / (n - 1).max(1) as f64;
                        let value = match distribution {
                            SamplingDistribution::LogUniform => {
                                let log_min = min.ln();
                                let log_max = max.ln();
                                (log_min + t * (log_max - log_min)).exp()
                            }
                            _ => min + t * (max - min),
                        };
                        ParameterValue::Float(value)
                    })
                    .collect()
            }
            ParameterRange::Integer { min, max } => {
                let step = ((*max - *min) as usize / n.max(1)).max(1) as i64;
                (*min..=*max)
                    .step_by(step as usize)
                    .map(ParameterValue::Int)
                    .collect()
            }
            ParameterRange::Boolean => {
                vec![ParameterValue::Bool(false), ParameterValue::Bool(true)]
            }
            ParameterRange::Categorical { options } => {
                options.iter().map(|s| ParameterValue::String(s.clone())).collect()
            }
        }
    }

    /// Retorna número total de parâmetros
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Retorna parâmetros por categoria
    pub fn by_category(&self) -> HashMap<String, Vec<&ParameterDef>> {
        let mut by_cat: HashMap<String, Vec<&ParameterDef>> = HashMap::new();
        for def in self.parameters.values() {
            by_cat
                .entry(def.category.clone())
                .or_default()
                .push(def);
        }
        by_cat
    }

    /// Filtra parâmetros por importância mínima
    pub fn filter_by_importance(&self, min_importance: f64) -> Self {
        let filtered: HashMap<_, _> = self
            .parameters
            .iter()
            .filter(|(_, def)| def.importance >= min_importance)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let priority_order: Vec<_> = self
            .priority_order
            .iter()
            .filter(|name| filtered.contains_key(*name))
            .cloned()
            .collect();

        Self {
            parameters: filtered,
            priority_order,
        }
    }
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuração específica da rede
#[derive(Debug, Clone)]
pub struct NetworkParameterSpace {
    pub space: ParameterSpace,
}

impl NetworkParameterSpace {
    pub fn new() -> Self {
        Self {
            space: ParameterSpace::new(),
        }
    }
}

impl Default for NetworkParameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Cria o espaço completo de parâmetros para NEN-V
pub fn create_full_parameter_space() -> ParameterSpace {
    let mut space = ParameterSpace::new();

    // =========================================================================
    // PARÂMETROS DE TIMING
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "timing.refractory_period",
            "timing",
            "Período refratário após disparo (timesteps)",
            ParameterRange::integer(1, 20),
            ParameterValue::Int(2),
        )
        .with_importance(0.7)
    );

    space.add_parameter(
        ParameterDef::new(
            "timing.stdp_window",
            "timing",
            "Janela temporal para STDP (timesteps)",
            ParameterRange::integer(10, 100),
            ParameterValue::Int(12),
        )
        .with_importance(0.9)
    );

    space.add_parameter(
        ParameterDef::new(
            "timing.stdp_tau_plus",
            "timing",
            "Constante de tempo para LTP",
            ParameterRange::continuous(10.0, 100.0),
            ParameterValue::Float(44.64548719223717),
        )
        .with_importance(0.85)
    );

    space.add_parameter(
        ParameterDef::new(
            "timing.stdp_tau_minus",
            "timing",
            "Constante de tempo para LTD",
            ParameterRange::continuous(5.0, 50.0),
            ParameterValue::Float(18.10546591833827),
        )
        .with_importance(0.85)
    );

    space.add_parameter(
        ParameterDef::new(
            "timing.eligibility_trace_tau",
            "timing",
            "Constante de tempo do eligibility trace",
            ParameterRange::log_continuous(50.0, 500.0),
            ParameterValue::Float(244.2443063119995),
        )
        .with_importance(0.8)
    );

    space.add_parameter(
        ParameterDef::new(
            "timing.stp_recovery_tau",
            "timing",
            "Constante de tempo de recuperação STP",
            ParameterRange::continuous(50.0, 300.0),
            ParameterValue::Float(77.84214867927184),
        )
        .with_importance(0.6)
    );

    // =========================================================================
    // PARÂMETROS DE APRENDIZADO
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "learning.base_learning_rate",
            "learning",
            "Taxa de aprendizado base",
            ParameterRange::log_continuous(0.001, 0.1),
            ParameterValue::Float(0.025613599256522394),
        )
        .with_importance(0.95)
    );

    space.add_parameter(
        ParameterDef::new(
            "learning.stdp_a_plus",
            "learning",
            "Amplitude de LTP",
            ParameterRange::log_continuous(0.001, 0.1),
            ParameterValue::Float(0.04691687125920246),
        )
        .with_importance(0.9)
    );

    space.add_parameter(
        ParameterDef::new(
            "learning.stdp_a_minus",
            "learning",
            "Amplitude de LTD",
            ParameterRange::log_continuous(0.001, 0.1),
            ParameterValue::Float(0.048529884548748355),
        )
        .with_importance(0.9)
    );

    space.add_parameter(
        ParameterDef::new(
            "learning.ltp_ltd_ratio",
            "learning",
            "Razão LTP/LTD (a_plus/a_minus)",
            ParameterRange::continuous(1.0, 4.0),
            ParameterValue::Float(1.3500397988478745),
        )
        .with_importance(0.85)
    );

    space.add_parameter(
        ParameterDef::new(
            "learning.weight_decay",
            "learning",
            "Taxa de decaimento de peso",
            ParameterRange::log_continuous(0.00001, 0.01),
            ParameterValue::Float(0.004670334294955324),
        )
        .with_importance(0.7)
    );

    space.add_parameter(
        ParameterDef::new(
            "learning.trace_increment",
            "learning",
            "Incremento do eligibility trace",
            ParameterRange::continuous(0.05, 0.3),
            ParameterValue::Float(0.15911045220194164),
        )
        .with_importance(0.75)
    );

    space.add_parameter(
        ParameterDef::new(
            "learning.istdp_rate",
            "learning",
            "Taxa de iSTDP",
            ParameterRange::log_continuous(0.0001, 0.01),
            ParameterValue::Float(0.0033873888898145847),
        )
        .with_importance(0.6)
    );

    // =========================================================================
    // PARÂMETROS DE HOMEOSTASE
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "homeostasis.target_firing_rate",
            "homeostasis",
            "Taxa de disparo alvo",
            ParameterRange::continuous(0.03, 0.25),
            ParameterValue::Float(0.12540178825040388),
        )
        .with_importance(0.9)
    );

    space.add_parameter(
        ParameterDef::new(
            "homeostasis.homeo_eta",
            "homeostasis",
            "Taxa de ajuste homeostático",
            ParameterRange::log_continuous(0.01, 0.5),
            ParameterValue::Float(0.23140821013632423),
        )
        .with_importance(0.85)
    );

    space.add_parameter(
        ParameterDef::new(
            "homeostasis.homeo_interval",
            "homeostasis",
            "Intervalo entre ajustes homeostáticos",
            ParameterRange::integer(1, 20),
            ParameterValue::Int(9),
        )
        .with_importance(0.7)
    );

    space.add_parameter(
        ParameterDef::new(
            "homeostasis.memory_alpha",
            "homeostasis",
            "Fator de memória exponencial",
            ParameterRange::log_continuous(0.005, 0.1),
            ParameterValue::Float(0.045733904031974706),
        )
        .with_importance(0.65)
    );

    space.add_parameter(
        ParameterDef::new(
            "homeostasis.meta_threshold",
            "homeostasis",
            "Threshold para meta-plasticidade",
            ParameterRange::continuous(0.05, 0.3),
            ParameterValue::Float(0.07981004399010214),
        )
        .with_importance(0.6)
    );

    space.add_parameter(
        ParameterDef::new(
            "homeostasis.meta_alpha",
            "homeostasis",
            "Taxa de ajuste meta-plástico",
            ParameterRange::log_continuous(0.001, 0.05),
            ParameterValue::Float(0.006522558713453487),
        )
        .with_importance(0.55)
    );

    // =========================================================================
    // PARÂMETROS DE ENERGIA
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "energy.max_energy",
            "energy",
            "Energia máxima por neurônio",
            ParameterRange::continuous(50.0, 200.0),
            ParameterValue::Float(52.45337652798122),
        )
        .with_importance(0.5)
    );

    space.add_parameter(
        ParameterDef::new(
            "energy.cost_fire_ratio",
            "energy",
            "Razão de custo por disparo (% da energia máx)",
            ParameterRange::continuous(0.01, 0.15),
            ParameterValue::Float(0.03352459772106367),
        )
        .with_importance(0.7)
    );

    space.add_parameter(
        ParameterDef::new(
            "energy.recovery_rate",
            "energy",
            "Taxa de recuperação de energia",
            ParameterRange::continuous(1.0, 25.0),
            ParameterValue::Float(6.120117784738568),
        )
        .with_importance(0.65)
    );

    space.add_parameter(
        ParameterDef::new(
            "energy.plasticity_cost_factor",
            "energy",
            "Fator de custo energético da plasticidade",
            ParameterRange::continuous(0.01, 0.2),
            ParameterValue::Float(0.07404983029329446),
        )
        .with_importance(0.5)
    );

    // =========================================================================
    // PARÂMETROS DE MEMÓRIA
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "memory.weight_clamp",
            "memory",
            "Limite máximo de peso sináptico",
            ParameterRange::continuous(1.0, 5.0),
            ParameterValue::Float(2.4314622258375076),
        )
        .with_importance(0.7)
    );

    space.add_parameter(
        ParameterDef::new(
            "memory.tag_decay_rate",
            "memory",
            "Taxa de decaimento de tags sinápticos",
            ParameterRange::log_continuous(0.001, 0.05),
            ParameterValue::Float(0.019619692671257),
        )
        .with_importance(0.6)
    );

    space.add_parameter(
        ParameterDef::new(
            "memory.capture_threshold",
            "memory",
            "Threshold para captura de memória",
            ParameterRange::continuous(0.05, 0.3),
            ParameterValue::Float(0.09869092530540445),
        )
        .with_importance(0.55)
    );

    space.add_parameter(
        ParameterDef::new(
            "memory.dopamine_sensitivity",
            "memory",
            "Sensibilidade à dopamina",
            ParameterRange::continuous(1.0, 10.0),
            ParameterValue::Float(5.1077163797122695),
        )
        .with_importance(0.7)
    );

    space.add_parameter(
        ParameterDef::new(
            "memory.consolidation_rate",
            "memory",
            "Taxa base de consolidação",
            ParameterRange::log_continuous(0.0001, 0.01),
            ParameterValue::Float(0.0019728414964333275),
        )
        .with_importance(0.6)
    );

    // =========================================================================
    // PARÂMETROS DE CURIOSIDADE
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "curiosity.scale",
            "curiosity",
            "Escala do reward de curiosidade",
            ParameterRange::continuous(0.01, 0.5),
            ParameterValue::Float(0.09869625682912213),
        )
        .with_importance(0.75)
    );

    space.add_parameter(
        ParameterDef::new(
            "curiosity.surprise_threshold",
            "curiosity",
            "Threshold de surpresa",
            ParameterRange::log_continuous(0.001, 0.1),
            ParameterValue::Float(0.005122138252975917),
        )
        .with_importance(0.6)
    );

    space.add_parameter(
        ParameterDef::new(
            "curiosity.habituation_rate",
            "curiosity",
            "Taxa de habituação",
            ParameterRange::continuous(0.9, 0.999),
            ParameterValue::Float(0.9374820782326819),
        )
        .with_importance(0.65)
    );

    // =========================================================================
    // PARÂMETROS DE REDE
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "network.inhibitory_ratio",
            "network",
            "Razão de neurônios inibitórios",
            ParameterRange::continuous(0.1, 0.4),
            ParameterValue::Float(0.2074287303973522),
        )
        .with_importance(0.8)
    );

    space.add_parameter(
        ParameterDef::new(
            "network.initial_threshold",
            "network",
            "Threshold inicial de disparo",
            ParameterRange::continuous(0.1, 0.5),
            ParameterValue::Float(0.2428401663300675),
        )
        .with_importance(0.75)
    );

    space.add_parameter(
        ParameterDef::new(
            "network.initial_exc_weight",
            "network",
            "Peso excitatório inicial",
            ParameterRange::continuous(0.02, 0.15),
            ParameterValue::Float(0.04056526478858787),
        )
        .with_importance(0.7)
    );

    space.add_parameter(
        ParameterDef::new(
            "network.initial_inh_weight",
            "network",
            "Peso inibitório inicial",
            ParameterRange::continuous(0.1, 1.5),
            ParameterValue::Float(0.6594694106279567),
        )
        .with_importance(0.7)
    );

    space.add_parameter(
        ParameterDef::new(
            "network.adaptive_threshold_multiplier",
            "network",
            "Multiplicador do adaptive threshold (sparse coding strength)",
            ParameterRange::continuous(0.5, 5.0),
            ParameterValue::Float(1.0),  // Default reduzido de 3.0 para 1.0
        )
        .with_importance(0.85)  // Alta importância - afeta runaway LTP/LTD
    );

    // =========================================================================
    // PARÂMETROS DE COMPETIÇÃO
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "competition.strength",
            "competition",
            "Força da competição lateral",
            ParameterRange::continuous(0.1, 0.5),
            ParameterValue::Float(0.2213325429990265),
        )
        .with_importance(0.6)
    );

    space.add_parameter(
        ParameterDef::new(
            "competition.interval",
            "competition",
            "Intervalo entre competições",
            ParameterRange::integer(5, 20),
            ParameterValue::Int(7),
        )
        .with_importance(0.5)
    );

    // =========================================================================
    // PARÂMETROS DE WORKING MEMORY
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "working_memory.capacity",
            "working_memory",
            "Capacidade de slots WM",
            ParameterRange::integer(3, 12),
            ParameterValue::Int(5),
        )
        .with_importance(0.6)
    );

    space.add_parameter(
        ParameterDef::new(
            "working_memory.recurrent_strength",
            "working_memory",
            "Força da recorrência WM",
            ParameterRange::continuous(0.5, 0.95),
            ParameterValue::Float(0.5880624493206996),
        )
        .with_importance(0.55)
    );

    space.add_parameter(
        ParameterDef::new(
            "working_memory.decay_rate",
            "working_memory",
            "Taxa de decaimento WM",
            ParameterRange::log_continuous(0.005, 0.1),
            ParameterValue::Float(0.010828406036791487),
        )
        .with_importance(0.55)
    );

    // =========================================================================
    // PARÂMETROS DE SONO
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "sleep.interval",
            "sleep",
            "Intervalo entre ciclos de sono",
            ParameterRange::integer(500, 10000),
            ParameterValue::Int(3586),
        )
        .with_importance(0.5)
    );

    space.add_parameter(
        ParameterDef::new(
            "sleep.replay_noise",
            "sleep",
            "Ruído durante replay",
            ParameterRange::continuous(0.01, 0.2),
            ParameterValue::Float(0.0191880476286607),
        )
        .with_importance(0.4)
    );

    // =========================================================================
    // PARÂMETROS DE STP
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "stp.use_fraction",
            "stp",
            "Fração de recursos usados por spike",
            ParameterRange::continuous(0.05, 0.3),
            ParameterValue::Float(0.15309237989203361),
        )
        .with_importance(0.6)
    );

    // =========================================================================
    // PARÂMETROS PREDITIVOS
    // =========================================================================
    space.add_parameter(
        ParameterDef::new(
            "predictive.state_learning_rate",
            "predictive",
            "Taxa de aprendizado de estados",
            ParameterRange::log_continuous(0.01, 0.5),
            ParameterValue::Float(0.08930597877113752),
        )
        .with_importance(0.65)
    );

    space.add_parameter(
        ParameterDef::new(
            "predictive.inference_iterations",
            "predictive",
            "Iterações de inferência",
            ParameterRange::integer(1, 10),
            ParameterValue::Int(2),
        )
        .with_importance(0.5)
    );

    // Ordena por importância
    space.priority_order.sort_by(|a, b| {
        let imp_a = space.parameters.get(a).map(|p| p.importance).unwrap_or(0.0);
        let imp_b = space.parameters.get(b).map(|p| p.importance).unwrap_or(0.0);
        imp_b.partial_cmp(&imp_a).unwrap()
    });

    space
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_space_creation() {
        let space = create_full_parameter_space();
        assert!(space.len() > 30);
    }

    #[test]
    fn test_sampling() {
        let space = create_full_parameter_space();
        let mut seed = 12345u64;
        let mut rng = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) & 0x7fff) as f64 / 32768.0
        };

        let config = space.sample(&mut rng);
        assert_eq!(config.len(), space.len());
    }

    #[test]
    fn test_filter_by_importance() {
        let space = create_full_parameter_space();
        let filtered = space.filter_by_importance(0.8);
        assert!(filtered.len() < space.len());
        assert!(filtered.len() > 0);
    }

    #[test]
    fn test_by_category() {
        let space = create_full_parameter_space();
        let by_cat = space.by_category();
        assert!(by_cat.contains_key("timing"));
        assert!(by_cat.contains_key("learning"));
        assert!(by_cat.contains_key("homeostasis"));
    }
}
