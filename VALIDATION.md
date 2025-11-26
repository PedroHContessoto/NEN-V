# NEN-V v2.0 - Documentação de Validação

## Estrutura do Projeto

```
nenv_v2/
├── Cargo.toml              # Configuração do projeto Rust
├── README.md               # Documentação principal
├── examples/
│   ├── basic_network.rs    # Exemplo de rede básica
│   ├── rl_agent.rs         # Exemplo de agente RL
│   └── curiosity_exploration.rs  # Exemplo de curiosidade
└── src/
    ├── lib.rs              # Ponto de entrada da biblioteca
    ├── autoconfig.rs       # Configuração automática unificada
    ├── nenv.rs             # Neurônio NENV
    ├── dendritoma.rs       # Sistema sináptico
    ├── glia.rs             # Metabolismo energético
    ├── network.rs          # Orquestração da rede
    ├── neuromodulation.rs  # Sistema de neuromodulação
    ├── working_memory.rs   # Memória de trabalho
    ├── predictive.rs       # Codificação preditiva
    └── intrinsic_motivation.rs  # Curiosidade intrínseca
```

## Módulos e Suas Funções

### Core (Implementados e Integrados)

| Módulo | Linhas | Função | Status |
|--------|--------|--------|--------|
| `lib.rs` | ~250 | Ponto de entrada, re-exportações | ✅ Completo |
| `autoconfig.rs` | ~1200 | Configuração automática unificada | ✅ Completo |
| `nenv.rs` | ~950 | Neurônio individual | ✅ Completo |
| `dendritoma.rs` | ~890 | Sistema sináptico (STDP, STP, traces) | ✅ Completo |
| `glia.rs` | ~410 | Metabolismo energético | ✅ Completo |
| `network.rs` | ~940 | Orquestração da rede | ✅ Completo |
| `neuromodulation.rs` | ~440 | Sistema de neuromodulação | ✅ Completo |

### Cognitivos (v2.0 - Novos)

| Módulo | Linhas | Função | Status |
|--------|--------|--------|--------|
| `working_memory.rs` | ~610 | Pool de memória de trabalho | ✅ Completo |
| `predictive.rs` | ~700 | Hierarquia preditiva, Active Inference | ✅ Completo |
| `intrinsic_motivation.rs` | ~750 | Curiosidade intrínseca, RND | ✅ Completo |

## Integrações Entre Módulos

### Fluxo Principal

```
TaskSpec (entrada do usuário)
    │
    ▼
AutoConfig::from_task()
    │
    ├── DerivedArchitecture (sensores, hidden, atuadores)
    │
    └── NetworkParams (80+ parâmetros derivados)
            │
            ▼
        Network::new() com neurônios NENV
            │
            ├── Dendritoma (plasticidade)
            ├── Glia (metabolismo)
            └── NeuromodulationSystem
                    │
                    ▼
            WorkingMemoryPool (contexto)
                    │
                    ▼
            PredictiveHierarchy (predição)
                    │
                    ▼
            CuriosityModule (exploração)
```

### Dependências Entre Módulos

```
lib.rs
├── dendritoma (nenhuma dependência interna)
├── glia (nenhuma dependência interna)
├── nenv → dendritoma, glia
├── network → nenv, neuromodulation
├── neuromodulation (nenhuma dependência interna)
├── working_memory (nenhuma dependência interna)
├── predictive (nenhuma dependência interna)
├── intrinsic_motivation (nenhuma dependência interna)
└── autoconfig → network (e todos os core)
```

## Parâmetros Derivados Automaticamente

O sistema AutoConfig deriva mais de 80 parâmetros a partir de apenas 3 inputs:
- `num_sensors`
- `num_actuators`
- `task_type`

### Categorias de Parâmetros

1. **Arquitetura**: neurônios, topologia, razão E/I, threshold
2. **Energia**: max_energy, custos, recuperação
3. **STDP**: janela, tau+/tau-, amplitudes LTP/LTD
4. **Homeostase**: target_fr, eta, intervalo, BCM
5. **Eligibility**: tau, increment
6. **STP**: recovery_tau, use_fraction
7. **Competição**: strength, interval
8. **Working Memory**: capacity, recurrence
9. **Curiosidade**: scale, threshold

## Filosofia de Design

### Princípios Fundamentais

1. **Minimal Intervention**: Sistema externo só fornece rewards
2. **Emergent Specialization**: Funções emergem, não são pré-designadas
3. **Self-Regulation**: Todos os parâmetros têm loops de feedback
4. **Intrinsic Drives**: Curiosidade e homeostase como motivadores
5. **No Hidden Supervision**: Sem conhecimento do "correto"

### O Que NÃO Fazer

| Anti-padrão | Justificativa |
|-------------|---------------|
| Backpropagation | Sinal de erro global não-biológico |
| Labels explícitos | Não disponíveis em ambiente natural |
| Curriculum forçado | Remove autonomia de exploração |
| Regularização externa | Sistema deve auto-regular |
| Reset de pesos | Rede deve lidar com própria estabilidade |

## Verificação de Compatibilidade

### Checklist de Integração

- [x] `lib.rs` exporta todos os módulos
- [x] `autoconfig` usa tipos de `network`
- [x] `network` usa `nenv` e `neuromodulation`
- [x] `nenv` usa `dendritoma` e `glia`
- [x] `dendritoma` tem métodos para autoconfig
- [x] Testes unitários em todos os módulos
- [x] Exemplos funcionais

### Uso Recomendado

```rust
use nenv_v2::prelude::*;  // Importação rápida

// ou específico:
use nenv_v2::autoconfig::{AutoConfig, TaskSpec, TaskType, RewardDensity};
use nenv_v2::working_memory::WorkingMemoryPool;
use nenv_v2::intrinsic_motivation::CuriosityModule;
use nenv_v2::predictive::PredictiveHierarchy;
```

## Total do Projeto

- **13 arquivos Rust** (10 módulos + 3 exemplos)
- **~8,071 linhas de código**
- **Cobertura completa** das lacunas identificadas no relatório

---

*Projeto NEN-V v2.0 - Neuromorphic Energy-based Neural Virtual Model*
*Desenvolvido com foco em plausibilidade biológica e autonomia adaptativa*
