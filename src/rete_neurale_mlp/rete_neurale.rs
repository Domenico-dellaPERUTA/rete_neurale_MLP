use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::fmt::{Display,Debug, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Write};

/// Esempio di una semplice coppia di input-output del Set di Addestramento di una Rete Neurale.
pub struct InputAddestranto {
    pub input: Vec<f64>,
    pub output: Vec<f64>
}

/// Trait per le funzioni di attivazione generiche.
/// Le funzioni di attivazione devono implementare questi metodi.
pub trait FunzioneAttivazione {
    /// Calcola il valore della funzione di attivazione.
    ///
    /// # Argomenti
    ///
    /// * `x` - Un valore di tipo `f64` per il quale calcolare l'attivazione.
    ///
    /// # Ritorna
    ///
    /// Il valore della funzione di attivazione.
    fn attiva(&self, x: f64) -> f64;

    /// Calcola la derivata della funzione di attivazione.
    ///
    /// # Argomenti
    ///
    /// * `x` - Un valore di tipo `f64` per il quale calcolare la derivata dell'attivazione.
    ///
    /// # Ritorna
    ///
    /// Il valore della derivata della funzione di attivazione.
    fn derivata(&self, x: f64) -> f64;
}

/// Implementazione della funzione Sigmoide.
/// La sigmoide è una funzione di attivazione comune che mappa i valori in un intervallo tra 0 e 1.
pub struct Sigmoide;

impl FunzioneAttivazione for Sigmoide {
    fn attiva(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivata(&self, x: f64) -> f64 {
        let s = self.attiva(x);
        s * (1.0 - s)
    }
}
#[derive(Debug)]
/// Implementazione della funzione ReLU (Rectified Linear Unit).
/// La ReLU restituisce il valore di input se è positivo, altrimenti restituisce 0.
pub struct ReLU;

impl FunzioneAttivazione for ReLU {
    fn attiva(&self, x: f64) -> f64 {
        x.max(0.0)
    }

    fn derivata(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Implementazione della funzione Leaky ReLU.
/// La Leaky ReLU permette una pendenza piccola per i valori negativi per evitare neuroni morti.
pub struct LeakyReLU {
    /// Parametro alpha per la pendenza nei valori negativi.
    pub alpha: f64,
}

impl FunzioneAttivazione for LeakyReLU {
    fn attiva(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            self.alpha * x
        }
    }

    fn derivata(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            self.alpha
        }
    }
}

/// Implementazione della funzione tanh (Tangente Iperbolica).
/// La tanh mappa i valori in un intervallo tra -1 e 1.
pub struct Tanh;

impl FunzioneAttivazione for Tanh {
    fn attiva(&self, x: f64) -> f64 {
        x.tanh()
    }

    fn derivata(&self, x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }
}

/// Implementazione della funzione Softplus.
/// La Softplus è una versione liscia della ReLU e ha la proprietà di essere differenziabile ovunque.
pub struct Softplus;

impl FunzioneAttivazione for Softplus {
    fn attiva(&self, x: f64) -> f64 {
        (1.0 + x.exp()).ln()
    }

    fn derivata(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Implementazione della funzione Swish.
/// La Swish è una funzione di attivazione che combina caratteristiche della ReLU e della sigmoide.
pub struct Swish;

impl FunzioneAttivazione for Swish {
    fn attiva(&self, x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    fn derivata(&self, x: f64) -> f64 {
        let sigmoide = 1.0 / (1.0 + (-x).exp());
        sigmoide + x * sigmoide * (1.0 - sigmoide)
    }
}

/*
    +---------------------------------------------------------------------------------------+
    |                               Classe Rete Neurale                                     |
    +---------------------------------------------------------------------------------------+
 */
#[derive(Debug)]
/// Struttura della rete neurale generica che supporta più strati nascosti.
pub struct ReteNeurale<F: FunzioneAttivazione> {
    strati: Vec<DMatrix<f64>>,          // I pesi di ogni strato (organizzati come connessioni tra i livelli)
    funzione_attivazione: F,            // La funzione di attivazione
    tasso_apprendimento: f64            // Il tasso di apprendimentox
}

/// Permette la stampa della rete
/// 
/// Esempio:
/// let rete: ReteNeurale<ReLU>;
/// .....
/// println!("{rete}");
impl<F: FunzioneAttivazione>  Display for ReteNeurale<F> {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result { //[*]
        let mut testo = String::new();
        testo += "Pesi Rete Neurale MLP\n"; // Separatore di strato
        let mut i=0;
        for strato in &self.strati {
            testo += format!("\nconnessioni livello [{}] - [{}]\n",i,i+1).as_str(); 
            for riga in strato.row_iter() {
                let riga_str = riga.iter()
                    .map(|valore| valore.to_string())
                    .collect::<Vec<String>>()
                    .join("\t");
                testo += riga_str.as_str() ;
                testo += "\n";
            }
           i += 1;
        }
        write!(f, "{}\n", testo)
    }
}
impl<F: FunzioneAttivazione> ReteNeurale<F> {
    /// Crea una nuova rete neurale con il numero di livelli nascosti specificato.
    ///
    /// # Argomenti
    ///
    /// * `dimensioni_strati` - Un vettore che specifica il numero di neuroni in ogni strato, incluso input e output.
    /// 
    ///     Esempio:
    ///       let dimensioni_strati = vec![2, 3, 2, 1];  // Input, due livelli nascosti, output
    /// 
    /// * `tasso_apprendimento` - Il tasso di apprendimento per l'algoritmo di backpropagation.
    /// * `funzione_attivazione` - La funzione di attivazione da utilizzare nella rete.
    pub fn nuova(
        dimensioni_strati: Vec<usize>,
        tasso_apprendimento: f64,
        funzione_attivazione: F,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut strati = Vec::with_capacity(dimensioni_strati.len() - 1);

        for i in 0..dimensioni_strati.len() - 1 {
            let pesi = DMatrix::from_fn(dimensioni_strati[i + 1], dimensioni_strati[i], |_, _| rng.gen_range(-1.0..1.0));
            strati.push(pesi);
        }

        ReteNeurale {
            strati,
            funzione_attivazione,
            tasso_apprendimento
        }
    }

    /// Propagazione in avanti attraverso la rete.
    ///
    /// # Argomenti
    ///
    /// * `input` - Vettore di input per la rete neurale.
    ///
    /// # Ritorna
    ///
    /// Un vettore di vettori contenenti le uscite di ogni strato.
    fn propagazione_avanti(&self, input: &DVector<f64>) -> Vec<DVector<f64>> {
        let mut uscite = Vec::with_capacity(self.strati.len() + 1);
        let mut attivazione_corrente = input.clone();
        uscite.push(attivazione_corrente.clone());

        for pesi in &self.strati {
            let input_strato = pesi * attivazione_corrente;
            attivazione_corrente = input_strato.map(|x| self.funzione_attivazione.attiva(x));
            uscite.push(attivazione_corrente.clone());
        }

        uscite
    }

    /// Metodo che interroga la Rete Neutale, elabora i dati di input
    /// e restituisce in output il risultato.
    /// 
    /// # Argomenti
    /// * `input` vettore dei dati in input
    /// 
    /// # Ritorna
    ///  vettore output (risposta)
    /// # Esempio
    /// ```
    ///  // usiamo gli stessi dati di addestramento per confronto ...
    ///  let dati_addestramento = [
    ///     InputAddestranto {input:vec![0.0, 0.0], output: vec![0.0]},
    ///     InputAddestranto {input:vec![0.0, 1.0], output: vec![1.0]},
    ///     InputAddestranto {input:vec![1.0, 0.0], output: vec![1.0]},
    ///     InputAddestranto {input:vec![1.0, 1.0], output: vec![0.0]}
    /// ];
    /// for set in dati_addestramento.iter() {
    ///     let uscita = rete.elabora(set.input.clone());
    ///     println!("Input: {:?}, Previsto: {:?}, Uscita: {:?}", set.input, set.output, uscita);
    /// }
    /// ```
    pub fn elabora(&self, input: Vec<f64>) -> Vec<f64> {
        let uscite = self.propagazione_avanti(&DVector::from_vec(input));
        uscite[uscite.len() - 1].data.as_vec().to_vec()
    }

    /// Retropropagazione per aggiornare i pesi della rete neurale.
    ///
    /// # Argomenti
    ///
    /// * `uscite` - Le uscite di ogni strato dalla propagazione in avanti.
    /// * `target` - Il vettore dei valori target.
    fn _retropropagazione(&mut self, uscite: Vec<DVector<f64>>, target: &DVector<f64>) {
        let mut errore = target - &uscite[uscite.len() - 1];
        let mut delta = errore.component_mul(&uscite[uscite.len() - 1].map(|x| self.funzione_attivazione.derivata(x)));

        for (i, pesi) in self.strati.iter_mut().enumerate().rev() {
            let uscita_precedente = &uscite[i];
            *pesi += self.tasso_apprendimento * (&delta * uscita_precedente.transpose());

            if i > 0 {
                errore = pesi.transpose() * &delta;
                delta = errore.component_mul(&uscite[i].map(|x| self.funzione_attivazione.derivata(x)));
            }
        }
    }

    /// Addestra la rete neurale su un singolo esempio.
    ///
    /// # Argomenti
    ///
    /// * `input` - Vettore di input per la rete neurale.
    /// * `target` - Vettore dei valori target per il training.
    /// 
    /// # Esempio
    /// ```
    ///  let dati_addestramento = [
    ///     InputAddestranto {input:vec![0.0, 0.0], output: vec![0.0]},
    ///     InputAddestranto {input:vec![0.0, 1.0], output: vec![1.0]},
    ///     InputAddestranto {input:vec![1.0, 0.0], output: vec![1.0]},
    ///     InputAddestranto {input:vec![1.0, 1.0], output: vec![0.0]}
    /// ];
    /// for _ in 0..1000000 {
    ///     for set in dati_addestramento.iter() {
    ///         rete.addestra(set.input.clone(), set.output.clone());
    ///     }
    /// }
    /// ```
    /// 
    pub fn addestra(&mut self, input: Vec<f64>, target: Vec<f64>) {
        
        let uscite = self.propagazione_avanti(&DVector::from_vec(input));
        self._retropropagazione(uscite,&DVector::from_vec(target));
    }
    /// Salva i pesi della rete neurale in un file di testo.
    ///
    /// # Argomenti
    ///
    /// * `file_path` - Il percorso del file di testo in cui salvare i pesi.
    ///
    /// # Ritorna
    ///
    /// Un `Result` che indica se l'operazione ha avuto successo o meno.
    pub fn salva_pesi_txt(&self, file_path: &str) -> Result<(), Error> {
        let mut file = File::create(file_path)?;

        for strato in &self.strati {
            for riga in strato.row_iter() {
                let riga_str = riga.iter()
                    .map(|valore| valore.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");
                writeln!(file, "{}", riga_str)?;
            }
            writeln!(file, "---")?; // Separatore di strato
        }

        Ok(())
    }

    /// Metodo che inverte una matrice formata come vettore di vettori
    fn trasponi<T: Clone>(matrice: Vec<Vec<T>>) -> Vec<Vec<T>> {
        if matrice.is_empty() {
            return vec![]; // Se la matrice è vuota, ritorna una matrice vuota.
        }
    
        let numero_righe = matrice.len();
        let numero_colonne = matrice[0].len();
    
        // Creiamo una nuova matrice trasposta con colonne vuote.
        let mut matrice_trasposta: Vec<Vec<T>> = vec![vec![]; numero_colonne];
    
        // Per ogni riga nella matrice originale
        for riga in matrice {
            // Inseriamo ogni elemento nella colonna corrispondente della nuova matrice trasposta.
            for (i, elemento) in riga.into_iter().enumerate() {
                matrice_trasposta[i].push(elemento);
            }
        }
    
        matrice_trasposta
    }


    /// Carica i pesi della rete neurale da un file di testo.
    ///
    /// # Argomenti
    ///
    /// * `file_path` - Il percorso del file di testo da cui caricare i pesi.
    ///
    /// # Ritorna
    ///
    /// Un `Result` che indica se l'operazione ha avuto successo o meno.
    pub fn carica_pesi_txt(&mut self, file_path: &str) -> Result<(), Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut strati = Vec::new();
        let mut attuale_strato:Vec<Vec<f64>> = Vec::new();
        

        for line in reader.lines() {
            let linea = line?;
            if linea.trim() == "---" {
                let num_righe = attuale_strato.len();
                let connessioni =attuale_strato.get(0);
                let mut num_colonne: usize=0;
                
                if let Some(x) = connessioni {
                    num_colonne = x.len();
                }else{
                    return Err(Error::from(ErrorKind::Other));
                }
                let dati_strato = DMatrix::from_vec(
                    num_righe,
                    num_colonne,
                    Self::trasponi(attuale_strato).clone().into_iter().flatten().collect(),
                );
                strati.push(dati_strato);
                attuale_strato = Vec::new();
                
            } else {
                let riga: Vec<f64> = linea.split_whitespace()
                    .map(|valore| valore.parse::<f64>().unwrap())
                    .collect();
                attuale_strato.push(riga);
            }
        }

        self.strati = strati;
        Ok(())
    }
}

