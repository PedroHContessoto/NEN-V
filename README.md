# NEN-V: Neuromorphic Energy-based Neural Virtual Model v2.0

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)
![Tests](https://img.shields.io/badge/tests-146%20passing-brightgreen.svg)

Uma implementação biologicamente plausível de rede neural spiking em Rust, com mecanismos de aprendizado inspirados em neurociência computacional.

## 🧠 Visão Geral

O NEN-V (Neuromorphic Energy-based Neural Virtual Model) é uma biblioteca Rust que implementa redes neurais spiking com características biologicamente plausíveis. Diferente de redes neurais artificiais tradicionais, o NEN-V simula mecanismos neurofisiológicos reais:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           NEN-V v2.0 ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                         COGNITIVE LAYER                                    │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐   │ │
│  │  │ Working Memory  │  │   Predictive    │  │    Intrinsic            │   │ │
│  │  │ Pool (7±2)      │◄─┤   Hierarchy     │◄─┤    Motivation           │   │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                          │
│  ┌───────────────────────────────────▼───────────────────────────────────────┐ │
│  │                         NEURAL NETWORK                                     │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐   │ │
│  │  │  Sensory    │───▶│   Hidden    │───▶│        Motor               │   │ │
│  │  │  Neurons    │    │   Layers    │    │        Neurons             │   │ │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────────┘   │ │
│  │         │                 │                          │                    │ │
│  │         └─────────────────┼──────────────────────────┘                    │ │
│  │                           ▼                                               │ │
│  │              ┌─────────────────────────┐                                  │ │
│  │              │     Neuromodulation     │                                  │ │
│  │              │  DA · NE · ACh · 5-HT   │                                  │ │
│  │              └─────────────────────────┘                                  │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                          │
│  ┌───────────────────────────────────▼───────────────────────────────────────┐ │
│  │                         PLASTICITY LAYER                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │     STDP     │  │ Eligibility  │  │   Homeo-     │  │    Energy    │  │ │
│  │  │  Asymmetric  │◄─┤   Traces     │◄─┤   stasis     │◄─┤   Gating     │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Principais Características

| Componente | Descrição |
|------------|-----------|
| **STDP Assimétrico** | Spike-Timing-Dependent Plasticity com janelas temporais biologicamente calibradas |
| **Homeostase Multi-escala** | Synaptic scaling, metaplasticidade BCM, threshold adaptativo |
| **Sistema Energético** | Metabolismo neural com glia, reservas energéticas e energy-gated learning |
| **Working Memory** | Pool de memória de trabalho (7±2 slots) com dinâmica de atrator |
| **Codificação Preditiva** | Hierarquia preditiva com Active Inference e Free Energy Principle |
| **Curiosidade Intrínseca** | Exploração autônoma via ICM (Intrinsic Curiosity Module) |
| **Neuromodulação** | Dopamina, Norepinefrina, Acetilcolina, Serotonina |
| **Eligibility Traces** | Three-factor learning para credit assignment temporal |

## 📦 Instalação

### Como Dependência

```toml
[dependencies]
nenv_v2 = { git = "https://github.com/seu-usuario/nenv_v2.git" }
```

### Build Local

```bash
git clone https://github.com/seu-usuario/nenv_v2.git
cd nenv_v2
cargo build --release
```

### Verificar Instalação

```bash
cargo test
# Deve passar todos os 146+ testes
```

## 🚀 Início Rápido

### Criação Manual da Rede

```rust
use nenv_v2::prelude::*;

fn main() {
    // Cria rede com 100 neurônios
    let mut network = Network::new(
        100,                              // Número de neurônios
        ConnectivityType::SmallWorld,     // Topologia
        0.2,                              // 20% inibitórios
        0.15,                             // Threshold de disparo
    );

    network.set_learning_mode(LearningMode::STDP);

    // Loop de simulação
    for step in 0..10000 {
        let inputs = generate_inputs(step);
        network.update(&inputs);

        // Aplica reward externo (ex: do ambiente)
        if step % 100 == 0 {
            network.apply_reward(0.5);
        }

        let stats = network.get_stats();
        if step % 1000 == 0 {
            println!("Step {}: FR={:.1}% Energy={:.1}%",
                     step,
                     stats.firing_rate * 100.0,
                     stats.avg_energy);
        }
    }
}
```

### Usando AutoConfig (Recomendado)

```rust
use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};

fn main() {
    // Define tarefa de Reinforcement Learning
    let task = TaskSpec {
        num_sensors: 8,
        num_actuators: 4,
        task_type: TaskType::ReinforcementLearning {
            reward_density: RewardDensity::Sparse,
            temporal_horizon: Some(100),
        },
    };

    // AutoConfig deriva automaticamente 80+ parâmetros
    let config = AutoConfig::from_task(task);
    config.print_report();

    // Cria rede otimizada para a tarefa
    let mut network = config.build_network().expect("Configuração válida");
}
```

### Working Memory + Predictive Coding

```rust
use nenv_v2::working_memory::WorkingMemoryPool;
use nenv_v2::predictive::{PredictiveHierarchy, PredictiveConfig};

fn main() {
    // Working Memory (7±2 slots como cognição humana)
    let mut wm = WorkingMemoryPool::new(7, 64);

    // Armazena padrão na memória de trabalho
    let pattern = vec![0.5; 64];
    wm.encode(pattern.clone(), 0);

    // Recupera por similaridade
    if let Some(retrieved) = wm.retrieve_by_similarity(&pattern) {
        println!("Padrão recuperado com sucesso!");
    }

    // Hierarquia Preditiva
    let config = PredictiveConfig::default();
    let mut hierarchy = PredictiveHierarchy::new(vec![64, 32, 16], config);

    // Processa observação
    let observation = vec![0.5; 64];
    let free_energy = hierarchy.process(&observation);
    println!("Free Energy: {:.4}", free_energy);
}
```

### Curiosidade Intrínseca

```rust
use nenv_v2::intrinsic_motivation::CuriosityModule;

fn main() {
    let mut curiosity = CuriosityModule::new(64, 4);

    let state = vec![0.5; 64];
    let action = vec![1.0, 0.0, 0.0, 0.0];  // One-hot
    let next_state = vec![0.6; 64];

    // Calcula recompensa intrínseca baseada em surpresa
    let intrinsic_reward = curiosity.compute_intrinsic_reward(
        &state, &action, &next_state
    );

    println!("Recompensa intrínseca: {:.4}", intrinsic_reward);
    // Maior surpresa = maior reward = mais exploração
}
```

## 📁 Estrutura do Projeto

```
nenv_v2/
├── src/
│   ├── lib.rs                    # Entry point da biblioteca
│   ├── nenv.rs                   # Neurônio individual (NENV)
│   ├── dendritoma.rs             # Sistema sináptico
│   ├── glia.rs                   # Metabolismo energético
│   ├── network.rs                # Orquestração de rede
│   ├── neuromodulation.rs        # Sistema neuromodulador
│   ├── working_memory.rs         # Memória de trabalho
│   ├── predictive.rs             # Codificação preditiva
│   ├── intrinsic_motivation.rs   # Curiosidade intrínseca
│   ├── constants.rs              # Constantes centralizadas
│   ├── sparse.rs                 # Matriz esparsa de conectividade
│   ├── lru_cache.rs              # Cache LRU para habituação
│   ├── plasticity/               # Módulo de plasticidade
│   │   ├── mod.rs
│   │   ├── stdp.rs               # STDP implementation
│   │   ├── eligibility.rs        # Eligibility traces
│   │   ├── short_term.rs         # STP (facilitação/depressão)
│   │   └── normalization.rs      # Normalização sináptica
│   └── autoconfig/               # Auto-configuração
│       ├── mod.rs
│       ├── task.rs               # Especificação de tarefas
│       ├── architecture.rs       # Derivação de arquitetura
│       ├── params.rs             # Derivação de parâmetros
│       └── adaptive.rs           # Adaptação online
│
├── examples/
│   ├── basic_network.rs          # Exemplo básico
│   ├── rl_agent.rs               # Agente RL
│   └── curiosity_exploration.rs  # Exploração com curiosidade
│
├── simulations/
│   └── realtime_environment/     # Simulação interativa
│       ├── main.rs               # Grid world navigation
│       └── README.md
│
├── experiments/
│   └── hyperparameter_search/    # Otimização de hiperparâmetros
│       ├── main.rs               # CLI principal
│       ├── mod.rs                # Módulo
│       ├── param_space.rs        # Espaço de 45+ parâmetros
│       ├── search.rs             # Algoritmos de busca
│       ├── evaluation.rs         # Sistema de benchmarks
│       └── orchestrator.rs       # Coordenação de experimentos
│
├── Cargo.toml
└── README.md
```

## 🔧 Módulos Principais

### Core Neural

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| **NENV** | `nenv.rs` | Neurônio individual com integração, disparo e período refratário |
| **Dendritoma** | `dendritoma.rs` | Sistema sináptico com STDP, eligibility traces, STP |
| **Glia** | `glia.rs` | Metabolismo energético, reservas, adaptação |
| **Network** | `network.rs` | Orquestração multi-neurônio com competição e homeostase |
| **Neuromodulation** | `neuromodulation.rs` | DA, NE, ACh, 5-HT com dinâmicas realistas |

### Sistemas Cognitivos

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| **Working Memory** | `working_memory.rs` | Pool de memória (7±2), dinâmica de atrator, decay |
| **Predictive** | `predictive.rs` | Hierarquia preditiva, Free Energy, Active Inference |
| **Intrinsic Motivation** | `intrinsic_motivation.rs` | ICM, RND, exploração autônoma |

### Plasticidade

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| **STDP** | `plasticity/stdp.rs` | Assimétrico, triplet, voltage-dependent |
| **Eligibility** | `plasticity/eligibility.rs` | Three-factor learning, traces temporais |
| **STP** | `plasticity/short_term.rs` | Facilitação e depressão de curto prazo |
| **Normalization** | `plasticity/normalization.rs` | Synaptic scaling, weight normalization |

### Configuração

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| **AutoConfig** | `autoconfig/` | Derivação automática de 80+ parâmetros |
| **Constants** | `constants.rs` | Constantes biológicas centralizadas |

## 🎮 Simulações

### Realtime Environment

Simulação completa de navegação em grid world que testa todos os componentes:

```bash
# Modo demonstração (recomendado)
cargo run --release --bin realtime_sim -- --demo

# Modo rápido
cargo run --release --bin realtime_sim -- --fast

# Modo benchmark (sem visualização)
cargo run --release --bin realtime_sim -- --benchmark
```

**Características:**
- Grid 2D com comida, perigos e obstáculos
- Agente neural com todos os sistemas integrados
- Visualização em tempo real no terminal
- Métricas detalhadas (firing rate, energia, neuromoduladores)

```
╔═══════════════════════════════════╗
║ · · 🧱 · · · 🍎 · · · · · · · · ║
║ · · · · · · · · · · 💀 · · · · ║
║ · · · · · 🧱 · · · · · · · · · ║
║ · · 🍎 · · · · · · · · · · · · ║
║ · · · · · · · 🤖 · · · · · 🧱 · ║
╚═══════════════════════════════════╝

┌──────────────────────────────────────────┐
│ Episode:   15 | Step:    234             │
│ Reward:  +0.990 | Total:  +12.45         │
│ Food:  12 | Danger:   2                  │
├──────────────────────────────────────────┤
│ Firing Rate:  8.50% | Energy: 78.3%      │
│ Dopamine:  0.450 | NE:  0.320            │
│ Exploration: 15.2% | WM Slots:  4        │
│ Free Energy:    2.34                     │
└──────────────────────────────────────────┘
```

## 🔬 Otimização de Hiperparâmetros

Sistema completo de busca inteligente com 45+ parâmetros otimizáveis:

```bash
# Bayesian Optimization (recomendado)
cargo run --release --bin hyperopt -- --strategy bayesian --trials 100

# Random Search rápido
cargo run --release --bin hyperopt -- --strategy random --trials 50

# Evolutionary Search
cargo run --release --bin hyperopt -- --strategy evolutionary --trials 200 --population 30

# Teste rápido
cargo run --release --bin hyperopt -- --quick

# Ver ajuda
cargo run --release --bin hyperopt -- --help
```

### Estratégias de Busca

| Estratégia | Descrição | Uso Recomendado |
|------------|-----------|-----------------|
| **Bayesian** | Gaussian Process + UCB | Melhor para poucos trials (<200) |
| **Evolutionary** | Algoritmo genético | Bom para espaços discretos |
| **Random** | Amostragem uniforme | Baseline rápido |
| **Grid** | Busca exaustiva | Poucos parâmetros (<5) |

### Categorias de Parâmetros

```
┌─────────────────────────────────────────────────────────────┐
│                    PARAMETER SPACE                          │
├─────────────────────────────────────────────────────────────┤
│ timing         :   6 parameters (STDP windows, traces)      │
│ learning       :   7 parameters (rates, LTP/LTD)            │
│ homeostasis    :   6 parameters (target FR, adaptation)     │
│ energy         :   4 parameters (costs, recovery)           │
│ memory         :   5 parameters (consolidation, tags)       │
│ curiosity      :   3 parameters (scale, habituation)        │
│ network        :   4 parameters (topology, weights)         │
│ working_memory :   3 parameters (capacity, decay)           │
│ predictive     :   2 parameters (inference, learning)       │
│ competition    :   2 parameters (strength, interval)        │
│ sleep          :   2 parameters (replay, noise)             │
│ stp            :   1 parameters (use fraction)              │
├─────────────────────────────────────────────────────────────┤
│ Total: 45 parameters                                        │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Testes

```bash
# Todos os testes (146+)
cargo test

# Testes específicos por módulo
cargo test working_memory
cargo test predictive
cargo test curiosity
cargo test stdp
cargo test homeostasis

# Testes com output detalhado
cargo test -- --nocapture

# Testes do hyperopt
cargo test --bin hyperopt

# Testes da simulação
cargo test --bin realtime_sim
```

## 📖 Exemplos

```bash
# Rede básica com STDP
cargo run --example basic_network

# Agente de Reinforcement Learning
cargo run --example rl_agent

# Exploração autônoma com curiosidade
cargo run --example curiosity_exploration
```

## 📊 Mecanismos Biológicos Implementados

### Plasticidade Sináptica ✅

- [x] **STDP Assimétrico** - tau_plus > tau_minus (Bi & Poo, 1998)
- [x] **iSTDP** - Inhibitory STDP para balanço E/I
- [x] **Eligibility Traces** - Three-factor learning (Izhikevich, 2007)
- [x] **Short-Term Plasticity** - Facilitação e depressão
- [x] **Synaptic Tagging** - Consolidação de memória
- [x] **Heterosynaptic Plasticity** - Competição entre sinapses

### Homeostase ✅

- [x] **Synaptic Scaling** - Normalização multiplicativa
- [x] **Intrinsic Plasticity** - Threshold adaptativo
- [x] **Metaplasticidade BCM** - Sliding threshold
- [x] **Controlador PID** - Regulação global de atividade

### Metabolismo ✅

- [x] **Sistema Energético** - ATP/reservas por neurônio
- [x] **Energy-Gated Learning** - Plasticidade dependente de energia
- [x] **Glia** - Suporte metabólico e sinalização
- [x] **Adaptação Metabólica** - Eficiência com experiência

### Cognição ✅

- [x] **Working Memory** - 7±2 slots, dinâmica de atrator
- [x] **Predictive Coding** - Hierarquia com Free Energy
- [x] **Active Inference** - Ação para reduzir surpresa
- [x] **Curiosidade Intrínseca** - ICM, RND, exploração

### Neuromodulação ✅

- [x] **Dopamina** - Reward prediction error
- [x] **Norepinefrina** - Arousal, exploração
- [x] **Acetilcolina** - Atenção, encoding
- [x] **Serotonina** - Humor, temporal discounting

## 🔮 Roadmap

### v2.1 (Planejado)
- [ ] Atenção Top-Down com gating
- [ ] Replay estruturado durante sono
- [ ] Memória episódica com hipocampo simulado
- [ ] Integração com ambientes OpenAI Gym

### v2.2 (Futuro)
- [ ] Multi-área cortical
- [ ] Oscillations (gamma, theta)
- [ ] Sparse distributed representations
- [ ] GPU acceleration (CUDA/Metal)

## 📚 Referências Científicas

### Plasticidade
- **STDP**: Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons.
- **Eligibility Traces**: Izhikevich, E.M. (2007). Solving the distal reward problem.
- **BCM**: Bienenstock, E.L., Cooper, L.N. & Munro, P.W. (1982). Theory for the development of neuron selectivity.

### Codificação Preditiva
- **Predictive Coding**: Rao, R.P. & Ballard, D.H. (1999). Predictive coding in the visual cortex.
- **Free Energy**: Friston, K. (2010). The free-energy principle: a unified brain theory?
- **Active Inference**: Friston, K. et al. (2017). Active inference and epistemic value.

### Motivação Intrínseca
- **ICM**: Pathak, D. et al. (2017). Curiosity-driven exploration by self-supervised prediction.
- **RND**: Burda, Y. et al. (2018). Exploration by random network distillation.

### Homeostase
- **Synaptic Scaling**: Turrigiano, G.G. (2008). The self-tuning neuron.

### Working Memory
- **Capacity**: Miller, G.A. (1956). The magical number seven, plus or minus two.
- **Attractor Dynamics**: Compte, A. et al. (2000). Synaptic mechanisms and network dynamics.

## 📄 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Add nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

### Guidelines
- Siga o estilo de código existente (rustfmt)
- Adicione testes para novas funcionalidades
- Atualize documentação conforme necessário
- Mantenha compatibilidade com versões anteriores

---

<div align="center">

**Filosofia Central**

*A rede não deve ser "programada" para ser inteligente; deve ter os **mecanismos corretos** para que inteligência **emerja** da interação com o ambiente.*

---

Made with 🧠 and ❤️ in Rust

</div>
