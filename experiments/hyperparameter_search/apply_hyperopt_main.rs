//! # Apply Hyperopt Results CLI
//!
//! Ferramenta para aplicar resultados do hyperopt ao derivation.rs
//!
//! ```bash
//! cargo run --release --bin apply_hyperopt
//! cargo run --release --bin apply_hyperopt experiments/results/mega_full_results.txt
//! ```

use std::env;
use std::path::Path;

// Re-use the module from hyperparameter_search
#[path = "apply_hyperopt.rs"]
mod apply_hyperopt;

use apply_hyperopt::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║             APPLY HYPEROPT → AUTOCONFIG                          ║");
    println!("║                                                                  ║");
    println!("║  Transfere parâmetros otimizados para o sistema de derivação     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let args: Vec<String> = env::args().collect();

    let results_path = if args.len() > 1 {
        args[1].clone()
    } else {
        // Tenta encontrar o arquivo mais recente
        let default_paths = [
            "experiments/results/mega_full_results.txt",
            "experiments/results/hyperopt_results.txt",
            "experiments/results/quick_test_results.txt",
        ];

        let mut found_path = None;
        for path in default_paths {
            if Path::new(path).exists() {
                found_path = Some(path.to_string());
                break;
            }
        }

        match found_path {
            Some(p) => p,
            None => {
                eprintln!("❌ Nenhum arquivo de resultados encontrado!");
                eprintln!("\nUso: apply_hyperopt <caminho_para_results.txt>");
                eprintln!("\nExemplo:");
                eprintln!("  cargo run --release --bin apply_hyperopt experiments/results/mega_full_results.txt");
                std::process::exit(1);
            }
        }
    };

    let derivation_path = "src/autoconfig/derivation.rs";

    match apply_hyperopt_to_derivation(&results_path, derivation_path) {
        Ok(_) => {
            println!("\n✅ Código gerado com sucesso!");
            println!("\n📋 PRÓXIMOS PASSOS:");
            println!("   1. Revise o código gerado acima");
            println!("   2. Copie as funções relevantes para {}", derivation_path);
            println!("   3. Execute: cargo test");
            println!("   4. Execute: cargo run --release --bin deep_diagnostic");
        }
        Err(e) => {
            eprintln!("❌ Erro: {}", e);
            std::process::exit(1);
        }
    }
}
