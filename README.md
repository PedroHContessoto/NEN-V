# NEN-V: Neuromorphic Energy-based Neural Virtual Model

Uma implementação em Rust de rede neural biologicamente plausível, com mecanismos de aprendizado inspirados em neurociência.

## 🧠 Características Biológicas

- **STDP (Spike-Timing-Dependent Plasticity)**: Aprendizado temporal baseado em causalidade
- **iSTDP**: Plasticidade inibitória para equilíbrio Excitatório/Inibitório (E/I)
- **Homeostase Sináptica**: Auto-regulação de taxa de disparo
- **Consolidação de Memória**: Transferência STM → LTM durante ciclos de sono
- **Modulação Glial**: Controle metabólico e energético
- **Synaptic Tagging & Capture**: Consolidação seletiva baseada em relevância
- **Dopamina-like Signaling**: Modulação de aprendizado por recompensa

## 📁 Estrutura do Projeto

```
nenv_visual_sim/
├── src/                          # Biblioteca Core NEN-V
│   ├── lib.rs                    # Ponto de entrada da biblioteca
│   ├── nenv.rs                   # Neurônio individual (NENV)
│   ├── dendritoma.rs             # Sistema dendrítico + aprendizado sináptico
│   ├── glia.rs                   # Modulação glial e metabólica
│   └── network.rs                # Orquestração de múltiplos neurônios
│
└── simulations/                  # Experimentos científicos
    └── gridworld_sensorimotor/   # Aprendizado sensorimotor em GridWorld
        ├── main.rs               # Loop principal da simulação
        ├── environment.rs        # Ambiente GridWorld
        └── visuals.rs            # Visualização em tempo real
```

### 🎯 Filosofia de Organização

**`src/`**: Contém apenas a **biblioteca core** do modelo NEN-V, reutilizável em diferentes contextos.

**`simulations/`**: Cada subpasta é um **experimento científico independente** que usa a biblioteca core. Permite executar múltiplas simulações sem misturar código.

## 🚀 Como Usar

### Compilar e Rodar a Simulação GridWorld

```bash
# Modo debug (mais lento, com checks)
cargo run --bin gridworld_sensorimotor

# Modo release (otimizado, ~10x mais rápido)
cargo run --bin gridworld_sensorimotor --release
```

### Usar a Biblioteca NEN-V em Outro Projeto

```rust
use nenv_visual_sim::network::{Network, ConnectivityType, LearningMode};

fn main() {
    let mut net = Network::new(
        20,                            // 20 neurônios
        ConnectivityType::FullyConnected,
        0.2,                           // 20% inibitórios
        0.15,                          // Threshold de disparo
    );

    net.set_learning_mode(LearningMode::STDP);
    net.set_weight_decay(0.002);

    // Loop de simulação
    let inputs = vec![0.0; 20];
    net.update(&inputs);

    println!("Energia média: {:.1}%", net.average_energy());
}
```

## 📊 Simulação GridWorld Sensorimotor

### Descrição

Um agente (rede neural de 20 neurônios) aprende a navegar em um grid e coletar comida usando apenas:
- **4 sensores direcionais** (UP, DOWN, LEFT, RIGHT)
- **4 motores** (movimento nas 4 direções)
- **Aprendizado por reforço** via sinal de dopamina

### Configuração Atual

```rust
Neurônios: 20 (4 sensoriais + 12 internos + 4 motores)
Topologia: FullyConnected
Aprendizado: STDP (a_plus=0.012, a_minus=0.006)
Weight Decay: 0.002
Recompensa (comida): +1.0
Punição (parede): -1.0
```

### Ciclos de Sono

A cada **3000 steps**, se a rede tiver aprendizado significativo (seletividade > 0.03) e experiência (≥3 sucessos), ela entra em **modo sono** por 500 steps:
- Replay espontâneo de padrões aprendidos
- Consolidação STM → LTM
- Plasticity reduzida
- Visualização do replay neural

### Métricas

- **Score**: Quantas vezes comeu
- **Seletividade**: Contraste entre pesos corretos e ruído
- **Energia**: Custo metabólico de cada ação
- **Exploration Rate**: Taxa de exploração aleatória

## 🔬 Criando uma Nova Simulação

```bash
# 1. Criar nova pasta
mkdir -p simulations/nova_simulacao

# 2. Criar main.rs
cat > simulations/nova_simulacao/main.rs <<EOF
use nenv_visual_sim::network::{Network, ConnectivityType, LearningMode};

fn main() {
    let mut net = Network::new(10, ConnectivityType::Grid2D, 0.2, 0.5);
    net.set_learning_mode(LearningMode::STDP);

    // Seu experimento aqui...
}
EOF

# 3. Adicionar ao Cargo.toml
[[bin]]
name = "nova_simulacao"
path = "simulations/nova_simulacao/main.rs"

# 4. Rodar
cargo run --bin nova_simulacao --release
```

## 📚 Referências Científicas

- **STDP**: Bi & Poo (1998) - "Synaptic modifications in cultured hippocampal neurons"
- **iSTDP**: Vogels et al. (2011) - "Inhibitory Plasticity Balances Excitation and Inhibition"
- **Synaptic Tagging**: Frey & Morris (1997) - "Synaptic tagging and long-term potentiation"
- **Memory Consolidation**: Walker & Stickgold (2004) - "Sleep-dependent learning and memory consolidation"

## 📝 Licença

MIT License - Veja `LICENSE` para detalhes.

## 👤 Autor

Pedro H. Contessoto

---

🤖 *Estrutura organizada com [Claude Code](https://claude.com/claude-code)*
