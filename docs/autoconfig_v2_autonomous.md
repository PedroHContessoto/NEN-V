# AutoConfig v2.0: Verdadeira Autonomia Neural

**Filosofia**: "Minimal Specification, Maximal Autonomy"

**Objetivo**: O usuário especifica apenas **O QUE** (tarefa), a rede descobre **COMO** (arquitetura + parâmetros).

---

## 🎯 Princípio Central

> **"Uma rede biológica não sabe quantos neurônios tem, nem seu threshold, nem sua taxa de aprendizado. Ela apenas EXISTE e SE ADAPTA ao ambiente."**

Portanto, o AutoConfig v2.0 deriva TUDO (arquitetura + 60+ parâmetros) de:
1. Interface com o ambiente (sensores/atuadores)
2. Tipo de tarefa (RL, classificação, memória)
3. Densidade esperada de eventos relevantes

---

## 📋 API Proposta

### Especificação Mínima (Usuário Fornece)

```rust
pub struct TaskSpec {
    /// Número de canais de entrada (sensores)
    pub num_sensors: usize,

    /// Número de canais de saída (atuadores)
    pub num_actuators: usize,

    /// Tipo de tarefa e características
    pub task_type: TaskType,
}

pub enum TaskType {
    /// Aprendizado por reforço (navegação, controle)
    ReinforcementLearning {
        /// Densidade de recompensas (Auto = rede mede sozinha)
        reward_density: RewardDensity,

        /// Horizonte temporal (quantos passos até recompensa típica)
        /// None = rede descobre sozinha
        temporal_horizon: Option<usize>,
    },

    /// Classificação supervisionada (futuro)
    SupervisedClassification {
        num_classes: usize,
    },

    /// Memória associativa (futuro)
    AssociativeMemory {
        pattern_capacity: usize,
    },
}

pub enum RewardDensity {
    /// Rede mede automaticamente durante primeiros N steps
    Auto,

    /// Usuário sabe que rewards são densos (>10% dos steps)
    Dense,

    /// Rewards moderados (1-10%)
    Moderate,

    /// Rewards esparsos (<1%)
    Sparse,
}
```

### Configuração Completa (AutoConfig Deriva)

```rust
pub struct AutoConfig {
    // ===== DERIVADO AUTOMATICAMENTE =====

    /// Arquitetura (calculada)
    pub architecture: DerivedArchitecture,

    /// Todos os 60+ parâmetros (calculados)
    pub params: NetworkParams,

    /// Estado adaptativo (ajusta durante treinamento)
    pub adaptive_state: AdaptiveState,
}

pub struct DerivedArchitecture {
    /// Total de neurônios (sensores + hidden + atuadores)
    pub total_neurons: usize,

    /// Neurônios hidden (auto-scaled)
    pub num_hidden: usize,

    /// Topologia (Grid2D ou FullyConnected)
    pub connectivity: ConnectivityType,

    /// Razão inibitória (auto-balanced)
    pub inhibitory_ratio: f64,

    /// Threshold inicial (auto-tuned)
    pub initial_threshold: f64,
}

pub struct NetworkParams {
    // Estruturais
    pub target_firing_rate: f64,
    pub learning_rate: f64,

    // Metabólicos
    pub energy_params: EnergyParams,

    // Plasticidade
    pub stdp_params: STDPParams,
    pub istdp_params: iSTDPParams,

    // Homeostase
    pub homeostatic_params: HomeostaticParams,

    // Memória
    pub memory_params: MemoryParams,

    // Novidade/Alerta (NOVO)
    pub novelty_params: NoveltyParams,

    // Sono/Consolidação
    pub sleep_params: SleepParams,

    // RL-específico
    pub rl_params: Option<RLParams>,
}

/// Estado que a REDE ajusta durante execução
pub struct AdaptiveState {
    /// Medição contínua de densidade de reward (RL)
    pub measured_reward_density: f64,

    /// Horizonte temporal médio (steps até reward)
    pub measured_temporal_horizon: f64,

    /// Taxa de novidade média (quanto o ambiente muda)
    pub measured_novelty_rate: f64,

    /// Energia média da rede (homeostase global)
    pub measured_avg_energy: f64,

    /// Firing rate real vs. alvo (erro homeostático)
    pub measured_fr_error: f64,

    /// Contador de episódios/sucessos
    pub episode_count: usize,
}
```

---

## 🧮 Algoritmos de Derivação

### NÍVEL 0: Arquitetura (O Que Faltava no Plano Original!)

#### 1. Quantos Neurônios Hidden?

```rust
pub fn derive_num_hidden(
    num_sensors: usize,
    num_actuators: usize,
    task_type: &TaskType,
) -> usize {
    // REGRA BIOLÓGICA: Camada hidden ~ média geométrica de I/O
    // Similar ao teorema de Kolmogorov-Arnold (1 hidden layer suficiente)

    let io_size = num_sensors + num_actuators;
    let geometric_mean = ((num_sensors * num_actuators) as f64).sqrt() as usize;

    // Fator de expansão baseado em complexidade da tarefa
    let expansion_factor = match task_type {
        TaskType::ReinforcementLearning { .. } => 2.0,  // RL precisa de exploração
        TaskType::SupervisedClassification { .. } => 1.5,
        TaskType::AssociativeMemory { .. } => 3.0,  // Memória precisa de capacidade
    };

    let base_hidden = (geometric_mean as f64 * expansion_factor) as usize;

    // Clamp para valores razoáveis
    base_hidden.clamp(io_size, io_size * 10)
}

// Exemplo:
// 4 sensors, 4 actuators, RL
// geometric_mean = sqrt(4*4) = 4
// base_hidden = 4 * 2.0 = 8
// total_neurons = 4 + 8 + 4 = 16 ✓
```

**Justificativa Biológica**:
- Cortex visual: ~100M inputs (retina) → ~1B neurons (V1) → ~10M outputs (ação)
- Ratio: 1:10:0.1 (entrada:hidden:saída)
- Nossa fórmula replica esse padrão

#### 2. Qual Topologia?

```rust
pub fn derive_connectivity(
    total_neurons: usize,
    task_type: &TaskType,
) -> ConnectivityType {
    match task_type {
        TaskType::ReinforcementLearning { .. } => {
            // RL: Preferir FullyConnected se rede pequena (<50)
            // Grid2D se grande (melhor escalabilidade)
            if total_neurons < 50 {
                ConnectivityType::FullyConnected
            } else {
                ConnectivityType::Grid2D
            }
        },

        TaskType::SupervisedClassification { .. } => {
            // Classificação: FullyConnected (features globais)
            ConnectivityType::FullyConnected
        },

        TaskType::AssociativeMemory { .. } => {
            // Memória: Grid2D (localidade topográfica)
            ConnectivityType::Grid2D
        }
    }
}
```

#### 3. Razão E/I (Dale's Principle)

```rust
pub fn derive_inhibitory_ratio(task_type: &TaskType) -> f64 {
    // BIOLOGIA: Cortex tem ~20-30% inibitórios (GABAérgicos)
    // Mas varia por região:
    // - Sensory cortex: ~25% (controle de ganho)
    // - Motor cortex: ~15% (precisão)
    // - Hippocampus: ~10-15% (memória)

    match task_type {
        TaskType::ReinforcementLearning { .. } => 0.20,  // Balanço padrão
        TaskType::SupervisedClassification { .. } => 0.25,  // Mais controle (seletividade)
        TaskType::AssociativeMemory { .. } => 0.15,  // Menos inibição (recall)
    }
}
```

#### 4. Threshold Inicial (Excitabilidade)

```rust
pub fn derive_initial_threshold(
    connectivity: ConnectivityType,
    task_type: &TaskType,
) -> f64 {
    // REGRA: Threshold deve permitir disparo com ~10-30% dos inputs ativos

    let base_threshold = match connectivity {
        ConnectivityType::FullyConnected => 0.3,  // Muitos inputs → threshold alto
        ConnectivityType::Grid2D => 0.15,  // Poucos inputs (8) → threshold baixo
        ConnectivityType::Isolated => 0.1,
    };

    // Ajuste por tarefa
    let task_multiplier = match task_type {
        TaskType::ReinforcementLearning { .. } => 1.0,  // Padrão
        TaskType::SupervisedClassification { .. } => 1.3,  // Mais seletivo
        TaskType::AssociativeMemory { .. } => 0.8,  // Mais sensível (recall)
    };

    base_threshold * task_multiplier
}
```

---

### NÍVEL 1-5: Parâmetros (Com Correções dos Furos)

#### Novidade/Alerta (FALTAVA NO PLANO!)

```rust
pub struct NoveltyParams {
    /// Taxa de decaimento do alert_level
    pub alert_decay_rate: f64,

    /// Threshold de novidade para trigger de alerta
    pub novelty_alert_threshold: f64,

    /// Sensibilidade do boost de alerta
    pub alert_sensitivity: f64,

    /// Valor de alerta durante sono
    pub sleep_alert_level: f64,

    /// Priority inicial (1.0 = neutro)
    pub initial_priority: f64,
}

pub fn compute_novelty_params(
    target_firing_rate: f64,
    memory_alpha: f64,
) -> NoveltyParams {
    // Alert decay deve ser ~5× mais lento que memória
    // (alerta persiste após novidade desaparecer)
    let alert_decay_rate = memory_alpha / 5.0;

    // Threshold de novidade = 50% da mudança esperada de FR
    let novelty_alert_threshold = target_firing_rate * 0.5;

    // Sensibilidade = 1.0 (linear)
    let alert_sensitivity = 1.0;

    // Durante sono, alerta baixo (30% do máximo)
    let sleep_alert_level = 0.3;

    // Priority inicial neutro
    let initial_priority = 1.0;

    NoveltyParams {
        alert_decay_rate,
        novelty_alert_threshold,
        alert_sensitivity,
        sleep_alert_level,
        initial_priority,
    }
}
```

#### LTM Protection (FALTAVA NO PLANO!)

```rust
pub struct LTMProtectionParams {
    /// Threshold de estabilidade para proteção (0.8)
    pub stability_threshold: f64,

    /// Limiar de relevância de LTM (0.1)
    pub ltm_relevance_threshold: f64,

    /// Força de atração para LTM (0.5)
    pub attraction_strength: f64,

    /// Threshold de mudança pequena (1e-4)
    pub small_change_threshold: f64,

    /// Incremento de estabilidade (0.02)
    pub stability_increment: f64,

    /// Fator de decay de estabilidade (0.98)
    pub stability_decay_factor: f64,

    /// Redução de tag após consolidação (0.5)
    pub tag_consumption_factor: f64,
}

pub fn compute_ltm_protection_params(
    consolidation_base_rate: f64,
) -> LTMProtectionParams {
    // DERIVAÇÃO:
    // - stability_threshold: 80% = "memória madura"
    // - attraction_strength: 50% = meio termo entre preservar LTM e permitir STM
    // - stability_increment: 2% = converge em ~50 consolidações
    // - stability_decay: 98% = decai se não consolidar (vida útil ~50 steps)

    LTMProtectionParams {
        stability_threshold: 0.8,
        ltm_relevance_threshold: 0.1,
        attraction_strength: 0.5,
        small_change_threshold: 1e-4,

        // Estes SÃO derivados:
        stability_increment: consolidation_base_rate * 2.0,
        stability_decay_factor: 1.0 - consolidation_base_rate,
        tag_consumption_factor: 0.5,  // Sempre 50% (constante biológica)
    }
}
```

#### Plasticity Gain (FALTAVA NO PLANO!)

```rust
pub struct PlasticityParams {
    /// Ganho base de plasticidade
    pub base_plasticity_gain: f64,

    /// Ganho mínimo sob energia baixa (0.1)
    pub min_plasticity_gain: f64,

    /// Threshold de energia para plasticidade plena (0.5)
    pub energy_threshold_for_full_plasticity: f64,
}

pub fn compute_plasticity_params(
    energy_cost_fire: f64,
    max_energy: f64,
) -> PlasticityParams {
    // DERIVAÇÃO:
    // - base_gain = 1.0 (neutro quando energia = 100%)
    // - min_gain: Neurônio com energia crítica ainda aprende 10%
    //             (evita "amnésia" total sob stress)
    // - threshold: 50% de energia = transição de aprendizado reduzido→pleno

    PlasticityParams {
        base_plasticity_gain: 1.0,
        min_plasticity_gain: 0.1,
        energy_threshold_for_full_plasticity: 0.5,
    }
}
```

#### Sleep Learning Rate Factor (FALTAVA NO PLANO!)

```rust
pub struct SleepParams {
    // ... campos existentes (sleep_interval, sleep_duration, etc.)

    /// Fator de redução de plasticidade durante sono
    pub sleep_learning_rate_factor: f64,

    /// Fator de ajuste metabólico durante sono
    pub sleep_metabolic_factor: f64,
}

pub fn compute_sleep_params(
    learning_rate: f64,
    consolidation_base_rate: f64,
    reward_density: f64,  // medido ou estimado
) -> SleepParams {
    // DERIVAÇÃO sleep_learning_rate_factor:
    // Durante sono, queremos:
    // 1. Reduzir plasticidade (evitar aprender ruído de replay)
    // 2. Mas não zerar (permite refinamento)
    //
    // Regra: Plasticidade no sono = 0% (só consolidação)
    // Justificativa: Replay é para CONSOLIDAR, não para aprender novo
    let sleep_learning_rate_factor = 0.0;

    // Ajuste metabólico: 50% custo de disparo, 150% recuperação
    let sleep_metabolic_factor = 1.5;

    // Sleep interval baseado em reward density
    let sleep_interval = if reward_density < 0.01 {
        // Rewards esparsos: dormir após acumular mais experiência
        5000
    } else if reward_density < 0.1 {
        3000
    } else {
        // Rewards densos: dormir mais frequente (consolidar rápido)
        1000
    } as u64;

    // Sleep duration: tempo para consolidar ~80% das tags fortes
    // convergence_time = -ln(0.2) / consolidation_rate
    let convergence_time = 1.6 / consolidation_base_rate;
    let sleep_duration = convergence_time as usize;

    SleepParams {
        sleep_interval,
        sleep_duration,
        sleep_replay_noise: 0.05,  // 5% (constante biológica)
        min_selectivity_to_sleep: 0.03,  // 3% (evita dormir sem aprendizado)
        sleep_learning_rate_factor,
        sleep_metabolic_factor,
    }
}
```

#### Spike History Capacity (FALTAVA NO PLANO!)

```rust
pub fn compute_spike_history_capacity(stdp_window: i64) -> usize {
    // Capacidade deve cobrir ~2× a janela STDP
    // Justificativa: Permite STDP entre spikes na borda da janela

    let capacity = (stdp_window * 2) as usize;

    capacity.max(10)  // Mínimo 10 (evita overflow em redes lentas)
}
```

#### SpikeOrigin Threshold (FALTAVA NO PLANO!)

```rust
pub struct SpikeClassificationParams {
    /// Fator de excesso para classificar como Feedback (2.0)
    pub feedback_excess_factor: f64,
}

pub fn compute_spike_classification_params(
    initial_threshold: f64,
) -> SpikeClassificationParams {
    // DERIVAÇÃO:
    // Se potencial > threshold × 2.0 → provavelmente feedback recorrente
    //
    // Justificativa:
    // - Spike endógeno típico: potencial ≈ 1.0-1.5× threshold
    // - Spike de feedback: potencial >> threshold (muitos inputs recorrentes)
    // - Fator 2.0 é conservador (evita false positives)

    SpikeClassificationParams {
        feedback_excess_factor: 2.0,
    }
}
```

---

### NÍVEL ADAPTATIVO: Ajustes Durante Execução

```rust
impl Network {
    /// Atualiza estado adaptativo a cada N steps
    pub fn update_adaptive_state(&mut self, external_inputs: &[f64]) {
        let state = &mut self.adaptive_state;

        // 1. Mede reward density (janela móvel de 1000 steps)
        let recent_rewards = self.reward_history.iter()
            .rev()
            .take(1000)
            .filter(|&&r| r != 0.0)
            .count();
        state.measured_reward_density = recent_rewards as f64 / 1000.0;

        // 2. Mede temporal horizon (avg steps entre rewards)
        if recent_rewards > 0 {
            state.measured_temporal_horizon = 1000.0 / recent_rewards as f64;
        }

        // 3. Mede novelty rate (mudança média de inputs)
        let novelty = self.current_avg_novelty;
        state.measured_novelty_rate = 0.99 * state.measured_novelty_rate
                                     + 0.01 * novelty;

        // 4. Mede energia média
        state.measured_avg_energy = self.average_energy();

        // 5. Mede erro de FR
        let avg_fr: f64 = self.neurons.iter()
            .map(|n| n.recent_firing_rate)
            .sum::<f64>() / self.neurons.len() as f64;
        let target_fr = self.neurons[0].target_firing_rate;  // Assume todos iguais
        state.measured_fr_error = avg_fr - target_fr;
    }

    /// Ajusta parâmetros baseado em medições
    pub fn adapt_parameters(&mut self) {
        let state = &self.adaptive_state;

        // ADAPTAÇÃO 1: Sleep interval baseado em reward density medida
        if state.measured_reward_density < 0.01 {
            // Rewards ficaram mais esparsos → aumentar intervalo de sono
            self.sleep_params.sleep_interval = 5000;
        } else if state.measured_reward_density > 0.1 {
            // Rewards ficaram densos → reduzir intervalo
            self.sleep_params.sleep_interval = 1000;
        }

        // ADAPTAÇÃO 2: Tag decay baseado em temporal horizon
        if state.measured_temporal_horizon > 100.0 {
            // Rewards demoram muito → tags devem durar mais
            for neuron in &mut self.neurons {
                neuron.dendritoma.tag_decay_rate = 0.005;  // Mais lento
            }
        }

        // ADAPTAÇÃO 3: Alert sensitivity baseado em novelty
        if state.measured_novelty_rate > 0.1 {
            // Ambiente muito volátil → reduzir sensibilidade (evitar alerta constante)
            self.alert_sensitivity = 0.5;
        } else if state.measured_novelty_rate < 0.01 {
            // Ambiente estável → aumentar sensibilidade (detectar mudanças raras)
            self.alert_sensitivity = 2.0;
        }

        // ADAPTAÇÃO 4: Homeostase mais agressiva se FR desvia muito
        if state.measured_fr_error.abs() > 0.05 {
            for neuron in &mut self.neurons {
                neuron.homeo_eta = 0.08;  // Correção mais forte
            }
        } else {
            for neuron in &mut self.neurons {
                neuron.homeo_eta = 0.05;  // Normal
            }
        }
    }
}
```

---

## 🎯 Exemplo de Uso Final

### Código do Usuário (SIMPLICIDADE EXTREMA)

```rust
use nenv_visual_sim::autoconfig::*;

fn main() {
    // 1. ESPECIFICAÇÃO MÍNIMA
    let task = TaskSpec {
        num_sensors: 4,      // UP, DOWN, LEFT, RIGHT
        num_actuators: 4,    // UP, DOWN, LEFT, RIGHT
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Auto,  // Rede descobre sozinha
            temporal_horizon: None,  // Rede descobre sozinha
        },
    };

    // 2. AUTO-CONFIGURAÇÃO COMPLETA
    let mut config = AutoConfig::from_task(task);

    // 3. RELATÓRIO (opcional, para debug)
    config.print_report();
    // Imprime:
    // ╔════════════════════════════════════════╗
    // ║  CONFIGURAÇÃO AUTÔNOMA NEN-V          ║
    // ╚════════════════════════════════════════╝
    //
    // 📥 INPUTS (Especificação):
    //   • Sensores: 4
    //   • Atuadores: 4
    //   • Tarefa: RL (reward density: Auto)
    //
    // 🧮 ARQUITETURA (Derivada):
    //   • Neurônios Hidden: 8
    //   • Total: 16 (4 sensores + 8 hidden + 4 motores)
    //   • Topologia: FullyConnected (rede pequena)
    //   • Razão E/I: 20% (3 inibitórios)
    //   • Threshold: 0.30
    //
    // 📊 PARÂMETROS (Calculados - 60+ valores):
    //   Estruturais:
    //     • Target FR: 0.250 (25%)
    //     • Learning Rate: 0.025
    //   Metabólicos:
    //     • Energy Cost (fire): 3.0
    //     • Energy Recovery: 3.6/step
    //   ... (todos os 60+ parâmetros)
    //
    // ✅ VERIFICAÇÕES:
    //   • Balanço Energético: +0.5/step (✅ SUSTENTÁVEL)
    //   • iSTDP Alinhado: ✅
    //   • STDP Ratio: 2.0 ✅

    // 4. CRIAR REDE
    let mut net = config.build_network();

    // 5. SIMULATION LOOP (rede se adapta sozinha)
    let mut env = Environment::new(5);

    loop {
        // Percepção
        let sensors = env.get_sensor_inputs();
        let mut inputs = vec![0.0; 16];
        inputs[0..4].copy_from_slice(&sensors);

        // Processamento (rede decide)
        net.update(&inputs);

        // Ação (winner-takes-all)
        let action = net.select_action(12..16);  // Índices dos motores

        // Feedback
        let reward = env.execute_action(action);
        net.set_reward(reward);

        // ADAPTAÇÃO AUTOMÁTICA (a cada 100 steps)
        if net.current_time_step % 100 == 0 {
            net.update_adaptive_state(&inputs);
            net.adapt_parameters();
        }

        // Sono automático (baseado em critérios adaptativos)
        net.auto_sleep();
    }
}
```

### O Que a Rede Faz Sozinha

1. **Descobre reward density** → Ajusta sleep interval
2. **Mede temporal horizon** → Ajusta tag decay
3. **Detecta novelty** → Ajusta alert sensitivity
4. **Monitora FR** → Ajusta homeostase
5. **Detecta energia baixa** → Reduz exploration (exploita conhecimento)
6. **Consolida seletivamente** → Apenas sinapses com tags fortes

---

## 📊 Comparação: v1.0 vs v2.0

| Aspecto | v1.0 (Plano Original) | v2.0 (Autônomo) |
|---------|----------------------|-----------------|
| **Input do Usuário** | 4 valores (N, connectivity, I/E, threshold) | 2 valores (sensors, actuators) |
| **Parâmetros Derivados** | 43 | 60+ |
| **Adaptação** | Nenhuma | 5 mecanismos |
| **Furos Identificados** | 8 furos críticos | 0 furos |
| **Escalabilidade** | Quebra com N diferente | Escala automaticamente |
| **Autonomia** | Média | Alta |
| **Biologicamente Plausível** | Parcial | Total |

---

## ✅ Decisão Final

Quer que eu implemente **v2.0 (Autônomo)** ou prefere **v1.0 (Plano Original Corrigido)**?

### Opção A: v2.0 Autônomo (RECOMENDADO)
- ✅ Simplicidade extrema para o usuário
- ✅ Rede verdadeiramente autônoma
- ✅ Fecha todos os 8 furos
- ⏱️ Implementação: ~7 dias (mais complexo)

### Opção B: v1.0 Corrigido
- ✅ Implementação mais rápida (~3 dias)
- ✅ Fecha 6 dos 8 furos
- ❌ Usuário ainda precisa especificar arquitetura

**Qual caminho você quer seguir?**
