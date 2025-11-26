//! # Constantes do Sistema NEN-V v2.0
//!
//! Centralização de todas as constantes mágicas do sistema com documentação
//! detalhada sobre fundamentação biológica e valores de referência.
//!
//! ## Organização
//!
//! - **Timing**: Constantes temporais (períodos refratários, janelas STDP, etc.)
//! - **Learning**: Taxas de aprendizado, amplitudes STDP
//! - **Energy**: Parâmetros metabólicos
//! - **Homeostasis**: Parâmetros homeostáticos
//! - **Memory**: Parâmetros de consolidação e tags sinápticas
//! - **Curiosity**: Parâmetros de motivação intrínseca
//! - **Network**: Parâmetros estruturais da rede

// ============================================================================
// METADADOS DO PROJETO
// ============================================================================

/// Versão atual da biblioteca
pub const VERSION: &str = "2.0.0";

/// Nome do projeto
pub const PROJECT_NAME: &str = "NEN-V";

/// Descrição completa
pub const DESCRIPTION: &str = "Neuromorphic Energy-based Neural Virtual Model";

// ============================================================================
// CONSTANTES TEMPORAIS (TIMING)
// ============================================================================

/// Módulo com constantes temporais baseadas em neurociência
pub mod timing {
    /// Período refratário padrão em timesteps (~5ms biológico)
    ///
    /// Referência: Período refratário absoluto de neurônios corticais
    /// é tipicamente 1-2ms absoluto + 5-10ms relativo
    pub const REFRACTORY_PERIOD: i64 = 5;

    /// Janela STDP total em timesteps (~50ms)
    ///
    /// Referência: Bi & Poo (1998), Markram et al. (1997)
    /// demonstraram janelas de 40-100ms para STDP
    pub const STDP_WINDOW: i64 = 50;

    /// Constante de tempo para LTP (potenciação) em ms
    ///
    /// ASSIMÉTRICO v2.0: tau_plus > tau_minus
    /// Favorece padrões causais - conexões pré→pós têm mais tempo
    /// para serem fortalecidas
    pub const STDP_TAU_PLUS: f64 = 40.0;

    /// Constante de tempo para LTD (depressão) em ms
    ///
    /// Mais curta que tau_plus para ser mais seletiva
    /// com conexões anti-causais
    pub const STDP_TAU_MINUS: f64 = 15.0;

    /// Constante de tempo do eligibility trace em timesteps (~200ms)
    ///
    /// Referência: Izhikevich (2007) - eligibility traces biológicos
    /// têm duração de 100-500ms para permitir credit assignment tardio
    pub const ELIGIBILITY_TRACE_TAU: f64 = 200.0;

    /// Taxa de recuperação de recursos sinápticos (STP) em timesteps
    ///
    /// Referência: Modelo Tsodyks-Markram, recuperação típica de 100-200ms
    pub const STP_RECOVERY_TAU: f64 = 150.0;

    /// Intervalo para aplicar homeostase (em timesteps)
    ///
    /// Otimizado por grid-search para balancear responsividade
    /// e estabilidade
    pub const HOMEO_INTERVAL: i64 = 9;

    /// Intervalo para aplicar competição lateral (em timesteps)
    pub const COMPETITION_INTERVAL: i64 = 10;

    /// Intervalo para normalização competitiva de pesos
    pub const NORMALIZATION_INTERVAL: i64 = 100;
}

// ============================================================================
// TAXAS DE APRENDIZADO (LEARNING RATES)
// ============================================================================

/// Módulo com constantes de aprendizado
pub mod learning {
    /// Taxa de aprendizado base para STDP
    ///
    /// Valor otimizado para convergência estável com redes
    /// de tamanho típico (20-100 neurônios)
    pub const BASE_LEARNING_RATE: f64 = 0.008;

    /// Amplitude de LTP (Long-Term Potentiation)
    ///
    /// Referência: Estudos de STDP mostram ratio LTP/LTD de 2-3:1
    /// para aprendizado estável
    pub const STDP_A_PLUS: f64 = 0.015;

    /// Amplitude de LTD (Long-Term Depression)
    ///
    /// Mais fraco que LTP (ratio 2.5:1) para favorever
    /// fortalecimento de conexões causais
    pub const STDP_A_MINUS: f64 = 0.006;

    /// Taxa de aprendizado para iSTDP (inibitório)
    ///
    /// Referência: Vogels et al. (2011) - balanceamento E/I
    pub const ISTDP_LEARNING_RATE: f64 = 0.001;

    /// Incremento do eligibility trace por correlação pré-pós
    pub const TRACE_INCREMENT: f64 = 0.15;

    /// Fração de recursos sinápticos usada por spike (STP)
    ///
    /// 15% de depleção por uso - balanceia depressão/facilitação
    pub const STP_USE_FRACTION: f64 = 0.15;

    /// Taxa de decaimento da facilitação sináptica
    pub const STP_FACILITATION_DECAY: f64 = 0.95;

    /// Taxa de decaimento de pesos (weight decay)
    pub const WEIGHT_DECAY: f64 = 0.0001;

    /// Limite superior para pesos sinápticos (clamp)
    pub const WEIGHT_CLAMP: f64 = 2.5;

    /// Fator alpha para média móvel exponencial de memória
    pub const MEMORY_ALPHA: f64 = 0.02;
}

// ============================================================================
// PARÂMETROS HOMEOSTÁTICOS
// ============================================================================

/// Módulo com constantes de homeostase
pub mod homeostasis {
    /// Taxa de disparo alvo (15% dos timesteps)
    ///
    /// Referência: Neurônios corticais disparam ~0.1-0.3 spikes/s
    /// em média, representando ~10-20% do tempo ativo
    pub const TARGET_FIRING_RATE: f64 = 0.15;

    /// Taxa de aprendizado homeostático (eta)
    ///
    /// Valor otimizado por grid-search para convergência
    /// sem oscilações
    pub const HOMEO_ETA: f64 = 0.1627;

    /// Threshold BCM (Bienenstock-Cooper-Munro)
    ///
    /// Separa regime de LTP vs LTD baseado em atividade recente
    pub const META_THRESHOLD: f64 = 0.12;

    /// Taxa de atualização do meta-threshold BCM
    pub const META_ALPHA: f64 = 0.005;

    /// Razão de ajuste de pesos na homeostase
    pub const HOMEO_WEIGHT_RATIO: f64 = 0.65;

    /// Razão de ajuste de threshold na homeostase
    pub const HOMEO_THRESHOLD_RATIO: f64 = 0.35;

    /// Taxa alvo para iSTDP
    pub const ISTDP_TARGET_RATE: f64 = 0.15;
}

// ============================================================================
// PARÂMETROS ENERGÉTICOS
// ============================================================================

/// Módulo com constantes de energia/metabolismo
pub mod energy {
    /// Energia máxima do neurônio (normalizada)
    pub const MAX_ENERGY: f64 = 100.0;

    /// Custo energético por spike
    ///
    /// ~10% da energia máxima por disparo
    pub const ENERGY_COST_FIRE: f64 = 10.0;

    /// Custo de manutenção por timestep
    pub const ENERGY_COST_MAINTENANCE: f64 = 0.1;

    /// Taxa de recuperação de energia por timestep
    pub const ENERGY_RECOVERY_RATE: f64 = 10.0;

    /// Reserva de energia de emergência
    pub const ENERGY_RESERVE: f64 = 20.0;

    /// Reserva máxima de emergência
    pub const MAX_RESERVE: f64 = 50.0;

    /// Taxa de preenchimento da reserva
    pub const RESERVE_FILL_RATE: f64 = 0.5;

    /// Threshold para usar reserva de emergência
    pub const RESERVE_USE_THRESHOLD: f64 = 20.0;

    /// Threshold de energia para plasticidade completa
    ///
    /// Abaixo disso, plasticidade é reduzida proporcionalmente
    pub const PLASTICITY_ENERGY_THRESHOLD: f64 = 50.0;
}

// ============================================================================
// PARÂMETROS DE MEMÓRIA
// ============================================================================

/// Módulo com constantes de memória e consolidação
pub mod memory {
    /// Taxa de decaimento das synaptic tags
    pub const TAG_DECAY_RATE: f64 = 0.008;

    /// Threshold para captura de memória (synaptic tagging)
    ///
    /// Tags acima deste valor disparam consolidação STM→LTM
    pub const CAPTURE_THRESHOLD: f64 = 0.15;

    /// Sensibilidade à dopamina para consolidação
    ///
    /// Amplifica relevância de eventos com reward
    pub const DOPAMINE_SENSITIVITY: f64 = 5.0;

    /// Threshold de estabilidade para proteção LTM
    pub const STABILITY_THRESHOLD: f64 = 0.7;

    /// Threshold de relevância para ativação LTM
    pub const LTM_RELEVANCE_THRESHOLD: f64 = 0.5;

    /// Força de atração para memórias consolidadas
    pub const ATTRACTION_STRENGTH: f64 = 0.1;

    /// Capacidade do spike history para STDP
    pub const SPIKE_HISTORY_CAPACITY: usize = 100;
}

// ============================================================================
// PARÂMETROS DE CURIOSIDADE/MOTIVAÇÃO INTRÍNSECA
// ============================================================================

/// Módulo com constantes de motivação intrínseca
pub mod curiosity {
    /// Erro de predição médio inicial (não-zero para normalização)
    pub const INITIAL_AVG_PREDICTION_ERROR: f64 = 0.1;

    /// Alpha para média móvel exponencial do erro
    pub const EMA_ALPHA: f64 = 0.01;

    /// Escala padrão da recompensa de curiosidade
    pub const CURIOSITY_SCALE: f64 = 0.1;

    /// Threshold mínimo de surpresa para gerar reward
    pub const SURPRISE_THRESHOLD: f64 = 0.01;

    /// Taxa de habituação (decaimento para estímulos repetidos)
    ///
    /// 0.995 = decaimento lento, permite re-exploração eventual
    pub const HABITUATION_RATE: f64 = 0.995;

    /// Tamanho máximo do mapa de habituação
    pub const HABITUATION_MAP_MAX_SIZE: usize = 10000;

    /// Threshold para cleanup do mapa de habituação
    pub const HABITUATION_CLEANUP_THRESHOLD: f64 = 0.1;

    /// Fator de discretização para hash de estados
    pub const STATE_DISCRETIZATION_FACTOR: f64 = 100.0;

    /// Tamanho do histórico de erros/rewards
    pub const HISTORY_SIZE: usize = 1000;
}

// ============================================================================
// PARÂMETROS DE REDE
// ============================================================================

/// Módulo com constantes estruturais da rede
pub mod network {
    /// Razão padrão de neurônios inibitórios (20%)
    ///
    /// Referência: Córtex cerebral tem ~20-25% de neurônios inibitórios
    pub const DEFAULT_INHIBITORY_RATIO: f64 = 0.20;

    /// Threshold padrão para disparo
    pub const DEFAULT_THRESHOLD: f64 = 0.15;

    /// Força da competição lateral
    pub const COMPETITION_STRENGTH: f64 = 0.3;

    /// Taxa de decaimento do nível de alerta
    pub const ALERT_DECAY_RATE: f64 = 0.05;

    /// Threshold de novidade para disparar alerta
    pub const NOVELTY_ALERT_THRESHOLD: f64 = 0.05;

    /// Sensibilidade do alerta a novidade
    pub const ALERT_SENSITIVITY: f64 = 1.0;

    /// Peso inicial para conexões excitatórias
    pub const INITIAL_EXCITATORY_WEIGHT: f64 = 0.05;

    /// Peso inicial para conexões inibitórias
    pub const INITIAL_INHIBITORY_WEIGHT: f64 = 0.3;

    /// Tamanho limite do spike buffer para STDP
    pub const SPIKE_BUFFER_LIMIT: usize = 1000;

    /// Tamanho do histórico de firing rate
    pub const FR_HISTORY_SIZE: usize = 1000;
}

// ============================================================================
// PARÂMETROS DE SONO/CONSOLIDAÇÃO
// ============================================================================

/// Módulo com constantes de sono e consolidação
pub mod sleep {
    /// Nível de alerta durante sono
    pub const SLEEP_ALERT_LEVEL: f64 = 0.3;

    /// Ruído de replay durante sono
    pub const SLEEP_REPLAY_NOISE: f64 = 0.05;

    /// Seletividade mínima para entrar em sono
    pub const MIN_SELECTIVITY_TO_SLEEP: f64 = 0.03;

    /// Fator de taxa de aprendizado durante sono
    pub const SLEEP_LEARNING_RATE_FACTOR: f64 = 0.0;

    /// Fator metabólico durante sono
    pub const SLEEP_METABOLIC_FACTOR: f64 = 1.5;
}

// ============================================================================
// PARÂMETROS DE PREDIÇÃO (PREDICTIVE CODING)
// ============================================================================

/// Módulo com constantes de codificação preditiva
pub mod predictive {
    /// Taxa de aprendizado para modelo generativo
    pub const GENERATIVE_LEARNING_RATE: f64 = 0.01;

    /// Taxa de aprendizado para atualização de estados
    pub const STATE_LEARNING_RATE: f64 = 0.1;

    /// Número de iterações de inferência por update
    pub const INFERENCE_ITERATIONS: usize = 5;

    /// Precisão mínima (floor)
    pub const MIN_PRECISION: f64 = 0.1;

    /// Precisão máxima (ceiling)
    pub const MAX_PRECISION: f64 = 10.0;

    /// Tamanho do histórico de free energy
    pub const FREE_ENERGY_HISTORY_SIZE: usize = 1000;

    /// Clamp para pesos do modelo generativo
    pub const GENERATIVE_WEIGHT_CLAMP: f64 = 3.0;
}

// ============================================================================
// PARÂMETROS DE WORKING MEMORY
// ============================================================================

/// Módulo com constantes de memória de trabalho
pub mod working_memory {
    /// Capacidade padrão (7±2 de Miller)
    pub const DEFAULT_CAPACITY: usize = 7;

    /// Capacidade mínima
    pub const MIN_CAPACITY: usize = 5;

    /// Capacidade máxima
    pub const MAX_CAPACITY: usize = 9;

    /// Força da recorrência para manutenção
    pub const RECURRENT_STRENGTH: f64 = 0.85;

    /// Taxa de decaimento dos slots
    pub const DECAY_RATE: f64 = 0.02;

    /// Força da inibição lateral entre slots
    pub const LATERAL_INHIBITION: f64 = 0.08;
}

// ============================================================================
// PARÂMETROS ADAPTATIVOS
// ============================================================================

/// Módulo com constantes do sistema adaptativo
pub mod adaptive {
    /// Cooldown entre intervenções adaptativas (timesteps)
    pub const INTERVENTION_COOLDOWN: i64 = 100;

    /// Threshold de firing rate para rede "morta"
    pub const DEAD_NETWORK_THRESHOLD: f64 = 0.001;

    /// Threshold de firing rate para excitação descontrolada
    pub const RUNAWAY_THRESHOLD: f64 = 0.9;

    /// Threshold de energia para risco de depleção
    pub const ENERGY_DEPLETION_THRESHOLD: f64 = 30.0;

    /// Severidade mínima para ação corretiva
    pub const MIN_SEVERITY_FOR_ACTION: u8 = 9;
}

// ============================================================================
// FUNÇÕES DE CONVENIÊNCIA
// ============================================================================

/// Retorna informações da versão
pub fn version_info() -> String {
    format!("{} v{} - {}", PROJECT_NAME, VERSION, DESCRIPTION)
}

/// Verifica se um valor está dentro da faixa biológica plausível
pub fn is_biologically_plausible_fr(fr: f64) -> bool {
    fr >= 0.01 && fr <= 0.5
}

/// Verifica se energia está em nível crítico
pub fn is_energy_critical(energy: f64) -> bool {
    energy < adaptive::ENERGY_DEPLETION_THRESHOLD
}

// ============================================================================
// TESTES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_asymmetry() {
        // tau_plus deve ser maior que tau_minus
        assert!(timing::STDP_TAU_PLUS > timing::STDP_TAU_MINUS);

        // LTP deve ser mais forte que LTD
        assert!(learning::STDP_A_PLUS > learning::STDP_A_MINUS);
    }

    #[test]
    fn test_homeostasis_ratios() {
        // Ratios devem somar 1.0
        let ratio_sum = homeostasis::HOMEO_WEIGHT_RATIO + homeostasis::HOMEO_THRESHOLD_RATIO;
        assert!((ratio_sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_energy_balance() {
        // Taxa de recuperação deve ser maior que custo de manutenção
        assert!(energy::ENERGY_RECOVERY_RATE > energy::ENERGY_COST_MAINTENANCE);
    }

    #[test]
    fn test_working_memory_bounds() {
        // Capacidade padrão dentro dos limites
        assert!(working_memory::DEFAULT_CAPACITY >= working_memory::MIN_CAPACITY);
        assert!(working_memory::DEFAULT_CAPACITY <= working_memory::MAX_CAPACITY);
    }

    #[test]
    fn test_biological_plausibility() {
        // Firing rate alvo deve ser biologicamente plausível
        assert!(is_biologically_plausible_fr(homeostasis::TARGET_FIRING_RATE));
    }
}
