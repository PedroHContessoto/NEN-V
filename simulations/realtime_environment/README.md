# Simulação Realtime: Ambiente de Navegação

Uma simulação completa que testa todos os componentes da rede neural NEN-V v2.0 em um ambiente de navegação 2D.

## Características

### Ambiente
- **Grid World**: Ambiente 2D discreto com visualização no terminal
- **Comida**: Itens que dão reward positivo ao serem coletados
- **Perigo**: Zonas que causam penalidade
- **Obstáculos**: Bloqueiam movimento e visão

### Agente Neural
- **Rede Spiking**: Rede neural com STDP e homeostase
- **Working Memory**: Mantém contexto de estados recentes
- **Predictive Coding**: Hierarquia preditiva para antecipação
- **Curiosidade Intrínseca**: Exploração motivada por novidade
- **Neuromodulação**: Dopamina e norepinefrina modulam aprendizado
- **Eligibility Traces**: Credit assignment para rewards tardios

### Sensores
- **Raycasting**: 8 direções × 3 canais (comida, perigo, obstáculo)
- **Posição**: Coordenadas normalizadas do agente
- **Proximidade**: Distância até comida/perigo mais próximo

### Ações
- **4 direções**: Cima, Baixo, Esquerda, Direita
- **Exploração ε-greedy**: Com decaimento adaptativo

## Execução

### Modo Demonstração (recomendado para primeira execução)
```bash
cargo run --release --bin realtime_sim -- --demo
```

### Modo Rápido
```bash
cargo run --release --bin realtime_sim -- --fast
```

### Modo Benchmark (sem visualização)
```bash
cargo run --release --bin realtime_sim -- --benchmark
```

### Modo Padrão (interativo)
```bash
cargo run --release --bin realtime_sim
```

## Configuração

Edite `SimConfig` em `main.rs` para personalizar:

```rust
SimConfig {
    grid_size: (15, 15),           // Tamanho do grid
    num_food: 5,                    // Quantidade de comida
    num_danger: 3,                  // Zonas de perigo
    num_obstacles: 10,              // Obstáculos
    food_reward: 1.0,               // Reward por comida
    danger_penalty: -0.5,           // Penalidade por perigo
    movement_cost: -0.01,           // Custo por movimento
    food_respawn_interval: 50,      // Intervalo de respawn
    max_steps_per_episode: 500,     // Steps por episódio
    num_episodes: 100,              // Número de episódios
    enable_visualization: true,     // Mostrar grid no terminal
    frame_delay_ms: 50,             // Delay entre frames
}
```

## Métricas

A simulação reporta:
- **Reward por episódio**: Total de recompensa acumulada
- **Comida coletada**: Número de itens de comida obtidos
- **Perigos atingidos**: Vezes que o agente entrou em zona de perigo
- **Taxa de disparo**: Atividade média da rede neural
- **Energia média**: Nível metabólico dos neurônios
- **Níveis de neuromoduladores**: Dopamina e norepinefrina
- **Free Energy**: Surpresa total da hierarquia preditiva
- **Taxa de exploração**: Probabilidade de ação aleatória

## Estrutura do Código

```
realtime_environment/
├── main.rs          # Código principal
│   ├── SimConfig    # Configuração da simulação
│   ├── Environment  # Ambiente de navegação
│   ├── NeuralAgent  # Agente com rede neural
│   ├── Metrics      # Sistema de métricas
│   └── Tests        # Testes unitários
└── README.md        # Este arquivo
```

## Diagrama de Fluxo

```
┌─────────────────────────────────────────────────────────────┐
│                    LOOP DE SIMULAÇÃO                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Environment │───►│  Observation │───►│ NeuralAgent  │  │
│  │             │    │  (28 dims)   │    │              │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│         ▲                                      │           │
│         │                                      ▼           │
│         │              ┌──────────────────────────┐        │
│         │              │   Predictive Coding      │        │
│         │              │   Working Memory         │        │
│         │              │   Spiking Network        │        │
│         │              │   Curiosity Module       │        │
│         │              └──────────────────────────┘        │
│         │                           │                      │
│         │                           ▼                      │
│  ┌──────┴──────┐    ┌──────────────┐                      │
│  │   Reward    │◄───│    Action    │                      │
│  │   (r, done) │    │   (0-3)      │                      │
│  └─────────────┘    └──────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Exemplo de Saída

```
╔═══════════════════════════════════╗
║ · · 🧱 · · · 🍎 · · · · · · · · ║
║ · · · · · · · · · · 💀 · · · · ║
║ · · · · · 🧱 · · · · · · · · · ║
║ · · 🍎 · · · · · · · · · · · · ║
║ · · · · · · · 🤖 · · · · · 🧱 · ║
║ · · · · · · · · · · · · · · · ║
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

## Requisitos

- Rust 1.70+
- Terminal com suporte a Unicode (para emojis)
- Recomendado: Terminal com cores ANSI

## Licença

Parte do projeto NEN-V v2.0
