
mod rete_neurale_mlp;
use std::time::Instant;

use nalgebra::DVector;

use crate::rete_neurale_mlp::rete_neurale::*;


fn main() {
    let dimensioni_strati = vec![2, 10, 1];  // Input, due livelli nascosti, output
    let tasso_apprendimento = 0.01;

    let funzione_attivazione = Sigmoide; // Puoi cambiare la funzione di attivazione

    let mut rete = ReteNeurale::nuova(
        dimensioni_strati,
        tasso_apprendimento,
        funzione_attivazione,
    );
/* 
   let _ok = rete.carica_pesi_txt("rete_neurale.txt");
   if let Err(s) = _ok {
        panic!("{s}");
   }
*/
  println!("Prima dell'addestramento ");
    println!("{}",rete);

    // Dati di addestramento
    let dati_addestramento = [
        InputAddestranto {input:vec![0.0, 0.0], output: vec![0.0]},
        InputAddestranto {input:vec![0.0, 1.0], output: vec![1.0]},
        InputAddestranto {input:vec![1.0, 0.0], output: vec![1.0]},
        InputAddestranto {input:vec![1.0, 1.0], output: vec![0.0]}
    ];

  // /* 
    // Addestramento della rete neurale
    for _ in 0..1000000 {
        for set in dati_addestramento.iter() {
            rete.addestra(set.input.clone(), set.output.clone());
        }
    }
    println!("Dopo addestramento ");
    println!("{}",rete);
  // */
    // Test della rete
    for set in dati_addestramento.iter() {
        let uscita = rete.elabora(set.input.clone());
        println!("Input: {:?}, Previsto: {:?}, Uscita: {:?}", set.input, set.output, uscita);
    }
  

 /* 
    let _risultato = rete.salva_pesi_txt("rete_neurale.txt");
    if let Err(e) = _risultato {
        println!("{:?}",e);
    }
*/ 
}