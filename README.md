# NEN-V: Neuromorphic Energy-based Neural Virtual Model v2.0

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)

Uma implementação biologicamente plausível de rede neural em Rust, com mecanismos de aprendizado inspirados em neurociência.

## 🧠 Visão Geral

O NEN-V (Neuromorphic Energy-based Neural Virtual Model) é uma biblioteca Rust que implementa redes neurais spiking com características biologicamente plausíveis:

- **STDP Assimétrico**: Spike-Timing-Dependent Plasticity com janelas temporais otimizadas
- **Homeostase Multi-escala**: Synaptic scaling, metaplasticidade BCM, threshold adaptativo
- **Sistema Energético**: Metabolismo neural com glia e reservas energéticas
- **Working Memory**: Pool de memória de trabalho com dinâmica de atrator
- **Codificação Preditiva**: Hierarquia preditiva e Active Inference
- **Curiosidade Intrínseca**: Exploração autônoma baseada em surpresa
- **Neuromodulação**: Dopamina, norepinefrina, acetilcolina, serotonina

## 📦 Instalação

Adicione ao seu `Cargo.toml`:

```toml
[dependencies]
nenv_v2 = "2.0.0"
```

## 🚀 Início Rápido

### Criação Manual da Rede

```rust
use nenv_v2::prelude::*;

// Cria rede com 20 neurônios
let mut network = Network::new(
    20,                              // Número de neurônios
    ConnectivityType::FullyConnected, // Topologia
    0.2,                             // 20% inibitórios
    0.15,                            // Threshold de disparo
);

network.set_learning_mode(LearningMode::STDP);

// Loop de simulação
for step in 0..1000 {
    let inputs = vec![0.5; 20];  // Inputs externos
    network.update(&inputs);
    
    let stats = network.get_stats();
    println!("Step {}: FR={:.2}%", step, stats.firing_rate * 100.0);
}
```

### Usando AutoConfig (Recomendado)

```rust
use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};

// Define tarefa de Reinforcement Learning
let task = TaskSpec {
    num_sensors: 8,
    num_actuators: 4,
    task_type: TaskType::ReinforcementLearning {
        reward_density: RewardDensity::Auto,
        temporal_horizon: Some(100),
    },
};

// AutoConfig deriva automaticamente 80+ parâmetros
let config = AutoConfig::from_task(task);
config.print_report();

// Cria rede otimizada
let mut network = config.build_network().expect("Configuração válida");
```

### Working Memory + Curiosidade

```rust
use nenv_v2::working_memory::WorkingMemoryPool;
use nenv_v2::intrinsic_motivation::CuriosityModule;

// Working Memory (7±2 slots como no cérebro humano)
let mut wm = WorkingMemoryPool::new(7, 64);
let pattern = vec![0.5; 64];
wm.encode(pattern, 0);

// Curiosidade Intrínseca para exploração
let mut curiosity = CuriosityModule::new(64, 4);
let state = vec![0.5; 64];
let action = vec![1.0, 0.0, 0.0, 0.0];
let next_state = vec![0.6; 64];

let intrinsic_reward = curiosity.compute_intrinsic_reward(
    &state, &action, &next_state
);
println!("Recompensa intrínseca: {:.4}", intrinsic_reward);
```

## 📚 Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEN-V v2.0 ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         PROCESSAMENTO                                │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│  │  │  Sensorial  │  │   Hidden    │  │         Atuadores           │  │   │
│  │  │  (Input)    │──│   Layer     │──│         (Output)            │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│  │        │                │                        │                   │   │
│  │        ▼                ▼                        ▼                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              WORKING MEMORY POOL (7±2 slots)                │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       PLASTICIDADE                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐    │  │
│  │  │    STDP      │  │ Eligibility  │  │   Predição/Modelo       │    │  │
│  │  │  Adaptativo  │◄─┤   Traces     │◄─┤   Interno               │    │  │
│  │  └──────────────┘  └──────────────┘  └─────────────────────────┘    │  │
│  │                           │                                          │  │
│  │                           ▼                                          │  │
│  │                 ┌─────────────────────┐                              │  │
│  │                 │   Neuromodulação    │                              │  │
│  │                 │   Diferencial       │                              │  │
│  │                 └─────────────────────┘                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        MOTIVAÇÃO                                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │  │
│  │  │ Curiosidade │  │ Saciedade/  │  │   Reward Extrínseco        │  │  │
│  │  │ Intrínseca  │──┤ Necessidade │──┤   (Ambiente)               │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Módulos

### Core

| Módulo | Descrição |
|--------|-----------|
| `nenv` | Neurônio individual (NENV) com dendritoma, glia e axônio |
| `dendritoma` | Sistema sináptico com STDP, eligibility traces, STP |
| `glia` | Metabolismo energético com reservas e adaptação |
| `network` | Orquestração de múltiplos neurônios |
| `neuromodulation` | Sistema de neuromodulação (DA, NE, ACh, 5-HT) |

### Cognitivos (v2.0)

| Módulo | Descrição |
|--------|-----------|
| `working_memory` | Pool de memória de trabalho com dinâmica de atrator |
| `predictive` | Hierarquia preditiva e Active Inference |
| `intrinsic_motivation` | Curiosidade intrínseca e exploração autônoma |

### Configuração

| Módulo | Descrição |
|--------|-----------|
| `autoconfig` | Configuração automática baseada na tarefa |

## 📊 Mecanismos Biológicos

### Plasticidade Sináptica
- ✅ STDP Assimétrico (tau_plus > tau_minus)
- ✅ iSTDP (Inhibitory STDP)
- ✅ Eligibility Traces (3-factor learning)
- ✅ Short-Term Plasticity (facilitação/depressão)
- ✅ Synaptic Tagging and Capture

### Homeostase
- ✅ Synaptic Scaling
- ✅ Intrinsic Plasticity (threshold adaptativo)
- ✅ Metaplasticidade BCM
- ✅ Controlador PID global

### Metabolismo
- ✅ Sistema energético com reserva
- ✅ Energy-gated learning
- ✅ Adaptação metabólica

### Dinâmicas de Rede
- ✅ Competição lateral (winner-take-all suave)
- ✅ Normalização competitiva
- ✅ Ciclos de sono/consolidação

## 📈 Priorização de Implementação

| Prioridade | Componente | Status | Impacto |
|------------|------------|--------|---------|
| 🔴 Alta | Working Memory | ✅ Completo | Crítico |
| 🔴 Alta | Predição/Modelo | ✅ Completo | Crítico |
| 🟡 Média | Curiosidade Intrínseca | ✅ Completo | Alto |
| 🟡 Média | Replay Estruturado | 🔄 Parcial | Alto |
| 🟢 Baixa | Atenção Top-Down | 📋 Planejado | Médio |

## 🧪 Testes

```bash
# Todos os testes
cargo test

# Testes específicos
cargo test working_memory
cargo test predictive
cargo test curiosity

# Com output detalhado
cargo test -- --nocapture
```

## 📖 Exemplos

```bash
# Rede básica
cargo run --example basic_network

# Agente RL
cargo run --example rl_agent

# Exploração com curiosidade
cargo run --example curiosity_exploration
```

## 📚 Referências Científicas

- **STDP**: Bi & Poo (1998), Markram et al. (1997)
- **Eligibility Traces**: Izhikevich (2007)
- **Predictive Coding**: Rao & Ballard (1999), Friston (2010)
- **Curiosity/ICM**: Pathak et al. (2017)
- **Homeostase**: Turrigiano (2008)
- **BCM**: Bienenstock, Cooper & Munro (1982)

## 📄 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor, leia o [CONTRIBUTING.md](CONTRIBUTING.md) antes de submeter PRs.

---

**Filosofia Central**: A rede não deve ser "programada" para ser inteligente; deve ter os **mecanismos corretos** para que inteligência **emerja** da interação com o ambiente.
