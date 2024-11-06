

use crate::rete_neurale_mlp::rete_neurale::*;
use std::sync::{Arc, RwLock};
use lazy_static::lazy_static;

mod rete_neurale_mlp;


// Uso di Arc<RwLock> per rendere il puntatore thread-safe
lazy_static! {
    static ref RETE: Arc<RwLock<Option<ReteNeurale>>> = Arc::new(RwLock::new(None));
    static ref INPUT_ADDESTRAMENTO: Arc<RwLock<Vec<InputAddestramento>>> = Arc::new(RwLock::new(vec![]));
}

fn main() {
    const TEST_ADDESTRA_NUOVA_RETE: bool = !true; 
    let mut rete: ReteNeurale;
    if TEST_ADDESTRA_NUOVA_RETE {
        println!("[*]  -- TEST ADDESTRA NUOVA RETE -- ");
        let tasso_apprendimento = 0.01;

        let strati: Vec<Strato> = vec![
            Strato {
                neuroni: 2,
                funzione_attivazione: Arc::new(Nessuna)
            },
            Strato {
                neuroni: 16,
                funzione_attivazione: Arc::new(Sigmoide)
            },
            Strato {
                neuroni: 1,
                funzione_attivazione: Arc::new(Sigmoide)//Arc::new(LeakyReLU { alpha: 0.05 }),
            },
        ];

        rete = ReteNeurale::nuova(strati, tasso_apprendimento);
    } else {
        println!("[*]-- TEST CARICA RETE ESISTENTE ----------- ");
        rete = ReteNeurale::carica("rete_neurale.txt");
    }

    let dati_addestramento = [
        InputAddestramento { input: vec![0.0, 1.0], output: vec![0.0] },
        InputAddestramento { input: vec![1.0, 0.0], output: vec![0.0] },
        InputAddestramento { input: vec![1.0, 1.0], output: vec![1.0] },
        InputAddestramento { input: vec![0.0, 0.0], output: vec![1.0] },
    ];

    if TEST_ADDESTRA_NUOVA_RETE {
        println!("[+] Prima dell'addestramento ----------");
        println!("{}", rete);

        for _ in 0..1000000 {
            for set in dati_addestramento.iter() {
                rete.addestra(set.input.clone(), set.output.clone());
            }
        }
        println!("[-] Dopo addestramento ----------------");
    } else {
        println!("[-] Rete  Caricata ----------------");
    }

    println!("{}", rete);

    println!("[!] test apprendimento ------------------");
    for set in dati_addestramento.iter() {
        let uscita = rete.elabora(set.input.clone());
        println!("Input: {:?}, Previsto: {:?}, Uscita: {:?}", set.input, set.output, uscita);
    }

    if TEST_ADDESTRA_NUOVA_RETE {
        let _risultato = rete.salva_pesi_txt("rete_neurale.txt");
        if let Err(e) = _risultato {
            println!("{:?}", e);
        }
    }

    println!("[*]-- FINE TEST --------------------------");
}