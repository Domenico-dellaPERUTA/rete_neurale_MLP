use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::fmt::{Display,Debug, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Write};
use std::sync::Arc;

const _FILE_INFO_RETE :          &str = "[#] ";
const _FILE_INFO_APPRENDIMENTO:  &str = "[+] ";
const _FILE_INFO_ATTIVAZIONE:    &str = "[*] ";
const _FILE_STRATO:              &str = "---";

#[derive(Clone)]
/// Coppia di input-output del Set di Addestramento di una Rete Neurale.
pub struct InputAddestramento {
    pub input: Vec<f64>,
    pub output: Vec<f64>
}
/// Informazioni relative al numero di neuroni e alla funzione di ativazione di uno strato.
pub struct Strato {
    pub neuroni: usize,
    pub funzione_attivazione: Arc<dyn FunzioneAttivazione + Send + Sync>
}

/// Trait per le funzioni di attivazione generiche.
/// Le funzioni di attivazione devono implementare questi metodi.
pub trait FunzioneAttivazione  {
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

    /// Metodo per ottenere il nome della funzione di attivazione
    fn nome(&self) -> &str;
    ///  Metodo per ottenere il nome della funzione di attivazione abbreviato
    fn sigla(&self) -> &str;
    /// Parametro opzionale
    fn alfa(&self) -> f64;
}


/// Implementazione della funzione Sigmoide.
/// La sigmoide è una funzione di attivazione comune che mappa i valori in un intervallo tra 0 e 1.
#[derive(Clone)]
pub struct Sigmoide;

impl FunzioneAttivazione for Sigmoide {
    fn attiva(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivata(&self, x: f64) -> f64 {
        let s = self.attiva(x);
        s * (1.0 - s)
    }
    fn nome(&self) -> &str {
        "Sigmoide"
    }
    fn sigla(&self) -> &str {
        "Sigmoide"
    }
    fn alfa(&self) -> f64 {
        0.0
    }
}

/// Implementazione della funzione ReLU (Rectified Linear Unit).
/// La ReLU restituisce il valore di input se è positivo, altrimenti restituisce 0.
#[derive(Debug,Clone)]
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
    fn nome(&self) -> &str {
        "Rectified Linear Unit"
    }
    fn sigla(&self) -> &str {
        "ReLU"
    }
    fn alfa(&self) -> f64 {
        0.0
    }
}

/// Implementazione della funzione Leaky ReLU.
/// La Leaky ReLU permette una pendenza piccola per i valori negativi per evitare neuroni morti.
#[derive(Clone)]
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

    fn nome(&self) -> &str {
        "Leaky Rectified Linear Unit"
    }
    fn sigla(&self) -> &str {
        "LeakyReLU"
    }
    fn alfa(&self) -> f64 {
        self.alpha
    }
}

/// Implementazione della funzione tanh (Tangente Iperbolica).
/// La tanh mappa i valori in un intervallo tra -1 e 1.
#[derive(Clone)]
pub struct Tanh;

impl FunzioneAttivazione for Tanh {
    fn attiva(&self, x: f64) -> f64 {
        x.tanh()
    }

    fn derivata(&self, x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }

    fn nome(&self) -> &str {
        "Tangente Iperbolica"
    }
    fn sigla(&self) -> &str {
        "Tanh"
    }
    fn alfa(&self) -> f64 {
        0.0
    }
}

/// Implementazione della funzione Softplus.
/// La Softplus è una versione liscia della ReLU e ha la proprietà di essere differenziabile ovunque.
#[derive(Clone)]
pub struct Softplus;

impl FunzioneAttivazione for Softplus {
    fn attiva(&self, x: f64) -> f64 {
        (1.0 + x.exp()).ln()
    }

    fn derivata(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    fn nome(&self) -> &str {
        "Softplus"
    }
    fn sigla(&self) -> &str {
        "Softplus"
    }
    fn alfa(&self) -> f64 {
        0.0
    }
}

/// Implementazione della funzione Swish.
/// La Swish è una funzione di attivazione che combina caratteristiche della ReLU e della sigmoide.
#[derive(Clone)]
pub struct Swish;

impl FunzioneAttivazione for Swish {
    fn attiva(&self, x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    fn derivata(&self, x: f64) -> f64 {
        let sigmoide = 1.0 / (1.0 + (-x).exp());
        sigmoide + x * sigmoide * (1.0 - sigmoide)
    }
    fn nome(&self) -> &str {
        "Swish"
    }
    fn sigla(&self) -> &str {
        "Swish"
    }
    fn alfa(&self) -> f64 {
        0.0
    }
}

/// Nessuana funzione di attivazione, si applica solo sui nodi di input.
/// 
#[derive(Clone)]
pub struct Nessuna;

impl FunzioneAttivazione for Nessuna {
    fn attiva(&self, x: f64) -> f64 {
        x
    }

    fn derivata(&self, x: f64) -> f64 {
        0.0
    }
    fn nome(&self) -> &str {
        "Nessua Funzione di Attivazione"
    }
    fn sigla(&self) -> &str {
        "Null"
    }
    fn alfa(&self) -> f64 {
        0.0
    }
}


/// Implementazione della funzione Lineare.
/// La funzione Lineare restituisce il valore in input senza alcuna trasformazione.
#[derive(Clone)]
pub struct Lineare;

impl FunzioneAttivazione for Lineare {
    fn attiva(&self, x: f64) -> f64 {
        x
    }

    fn derivata(&self, x: f64) -> f64 {
        1.0
    }

    fn nome(&self) -> &str {
        "Lineare"
    }
    fn sigla(&self) -> &str {
        "Lineare"
    }
    fn alfa(&self) -> f64 {
        0.0
    }
}

/*
    +---------------------------------------------------------------------------------------+
    |                               Classe Rete Neurale                                     |
    +---------------------------------------------------------------------------------------+
 */

/// Struttura della rete neurale generica che supporta più strati nascosti.
#[derive(Clone)]
pub struct ReteNeurale {
    strati: Vec<DMatrix<f64>>,          // I pesi di ogni strato (organizzati come connessioni tra i livelli)
    funzioni_attivazione: Vec<Arc<dyn FunzioneAttivazione + Send + Sync>>,  // Le funzioni di attivazione in ordine per strati
    tasso_apprendimento: f64 ,           // Il tasso di apprendimentox,
    dimensioni_strati:Vec<usize>
}

/// Permette la stampa della rete
/// 
/// Esempio:
/// let rete: ReteNeurale<ReLU>;
/// .....
/// println!("{rete}");
impl Display for ReteNeurale {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut testo = String::new();
        testo += "Pesi Rete Neurale MLP\n";
        let mut nomi_funz_attivazione = String::new();
        
        for funzione_attivazione in self.funzioni_attivazione.clone().into_iter()  {
            nomi_funz_attivazione += &(funzione_attivazione.nome().to_string()+ "; ");
        }
        testo += format!("Funzioni di attivazione per livello: {:?}\n", nomi_funz_attivazione).as_str(); // Stampa il nome della funzione di attivazione

        let mut i = 0;
        for strato in &self.strati {
            testo += format!("\nConnessioni livello [{}] - [{}]\n", i, i + 1).as_str();
            for riga in strato.row_iter() {
                let riga_str = riga.iter()
                    .map(|valore| valore.to_string())
                    .collect::<Vec<String>>()
                    .join("\t");
                testo += riga_str.as_str();
                testo += "\n";
            }
            i += 1;
        }
        write!(f, "{}\n", testo)
    }
}
impl ReteNeurale {
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
    pub fn nuova_rete_uniforme(
        dimensioni_strati: Vec<usize>,
        tasso_apprendimento: f64,
        funzione_attivazione:Arc<dyn FunzioneAttivazione + Send + Sync>
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut strati = Vec::with_capacity(dimensioni_strati.len() - 1);

        for i in 0..dimensioni_strati.len() - 1 {
            let pesi = DMatrix::from_fn(dimensioni_strati[i + 1], dimensioni_strati[i], |_, _| rng.gen_range(-1.0..1.0));
            strati.push(pesi);
        }
        let funzioni_attivazione = vec![funzione_attivazione];
        ReteNeurale {
            strati,
            funzioni_attivazione,
            tasso_apprendimento,
            dimensioni_strati
        }
    }

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
    /// * `funzioni_attivazione` - Lista delle funzioni di attivazione per singoli strati.
    /// 
    pub fn nuova( info_strati: Vec<Strato>, tasso_apprendimento: f64 ) -> Self {
        let mut funzioni_attivazione:Vec<Arc<dyn FunzioneAttivazione + Send + Sync>> = Vec::new();
        let mut dimensioni_strati= Vec::new();
        let mut primo_strato = true;
        for info_strato in info_strati.into_iter() {
            dimensioni_strati.push(info_strato.neuroni);
            if primo_strato {
                primo_strato = false;
                funzioni_attivazione.push(Arc::new(Nessuna));
            }else{
                funzioni_attivazione.push(info_strato.funzione_attivazione);
            }
        }
        let mut rng = rand::thread_rng();
        let mut strati = Vec::with_capacity(dimensioni_strati.len() - 1);

        for i in 0..dimensioni_strati.len() - 1 {
            let pesi = DMatrix::from_fn(dimensioni_strati[i + 1], dimensioni_strati[i], |_, _| rng.gen_range(-1.0..1.0));
            strati.push(pesi);
        }
        
        ReteNeurale {
            strati,
            funzioni_attivazione,
            tasso_apprendimento,
            dimensioni_strati
        }
    }


    /// Crea una rete da un file contiene i pesi e le informazioni della rete, da un file txt precedentemente creato.
    /// 
    pub fn carica(file_txt: &str) -> Self {
        let mut rete = Self::nuova_rete_uniforme(vec![0], 0.0, Arc::new(Sigmoide));
        rete.carica_pesi_txt(file_txt).unwrap();
        rete
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
        let mut i=1; // la prima funzione di attivazione è nulla (perché non si applica allo strato degli input [i != 0] )
        for pesi in &self.strati {
            let input_strato = pesi * attivazione_corrente;
            attivazione_corrente = input_strato.map( |x| self.funzioni_attivazione[i].attiva(x) );
            uscite.push(attivazione_corrente.clone());

            // l'indice dipende dal numero di funzioni di attivazioni presenti 
            if i < self.funzioni_attivazione.len() - 1 {
                i += 1;
            }else if self.funzioni_attivazione.len() > 1 {
                i = 1; // salta il primo strato nullo di default (strato input)
            }else {
                i = 0; // si verifica nel caso in cui e stato assegnato una solo funzione di attivazione (per tugli gli strati)
            }
            
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
    ///     InputAddestramento {input:vec![0.0, 0.0], output: vec![0.0]},
    ///     InputAddestramento {input:vec![0.0, 1.0], output: vec![1.0]},
    ///     InputAddestramento {input:vec![1.0, 0.0], output: vec![1.0]},
    ///     InputAddestramento {input:vec![1.0, 1.0], output: vec![0.0]}
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
        let mut j = self.funzioni_attivazione.len()-1;
        let mut errore = target - &uscite[uscite.len() - 1];
        let mut delta = errore.component_mul(&uscite[uscite.len() - 1].map(|x| self.funzioni_attivazione[j].derivata(x)));
            
        for (i, pesi) in self.strati.iter_mut().enumerate().rev() {
            
            // l'indice 'j' dipende dal numero di funzioni di attivazioni presenti 
            if self.funzioni_attivazione.len() == 1 {
                j = 0; // caso in cui vie una sola funzione di attivazione per tutti gli strati
            }else {
                // caso con multi funzioni di attivazione (eccetto per l'input)
                if j > 1 { // salta la prima funzione di attivazione associata agli input di default Nulla!!
                    j -= 1;
                } else {
                    j = self.funzioni_attivazione.len() - 1;
                }
            }
            let uscita_precedente = &uscite[i];
            *pesi += self.tasso_apprendimento * (&delta * uscita_precedente.transpose());

            if i > 0 {
                errore = pesi.transpose() * &delta;
                delta = errore.component_mul(&uscite[i].map(|x| self.funzioni_attivazione[j].derivata(x)));
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
    ///     InputAddestramento {input:vec![0.0, 0.0], output: vec![0.0]},
    ///     InputAddestramento {input:vec![0.0, 1.0], output: vec![1.0]},
    ///     InputAddestramento {input:vec![1.0, 0.0], output: vec![1.0]},
    ///     InputAddestramento {input:vec![1.0, 1.0], output: vec![0.0]}
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
        writeln!( file, "{} {}",_FILE_INFO_APPRENDIMENTO, self.tasso_apprendimento )?;
        
        let mut nomi_funz_attivazione = String::new();
        
        for funzione_attivazione in self.funzioni_attivazione.clone().into_iter()  {
            if funzione_attivazione.sigla() != "LeakyReLU" {
                nomi_funz_attivazione += &(funzione_attivazione.sigla().to_string()+ "; ");
            }else{
                nomi_funz_attivazione += &(funzione_attivazione.sigla().to_string()+ "_" + funzione_attivazione.alfa().to_string().as_str() +"; ");
            }
        }
        writeln!( file, "{} {}",_FILE_INFO_ATTIVAZIONE, nomi_funz_attivazione.as_str())?;
        writeln!( file, "{} {}", 
            _FILE_INFO_RETE, 
            format!("{:?}", self.dimensioni_strati ).replace("[", "").replace("]", "")
        )?;

        for strato in &self.strati {
            
            for riga in strato.row_iter() {
                let riga_str = riga.iter()
                    .map(|valore| valore.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");
                writeln!(file, "{}", riga_str)?;
            }
            writeln!(file, "{}", _FILE_STRATO )?; // Separatore di strato
        }

        Ok(())
    }

    /**
     * Restituisce i singoli pesi delle connessioni tragli strati della rete neurale.
     * I pesi sono organizzati come un vettori tridimensionale di varori in virgola mobile.
     * Puo essere visto come un vettore delle connessioni tragli strati organizzati come matrici.
     */
    pub fn pesi_connessioni(&self) -> Vec<Vec<Vec<f64>>> {
        let mut strati_rete = vec![];

        for strato in &self.strati {
            let mut connessioni = vec![];
            for riga in strato.row_iter() {
                let riga_connessioni = riga.iter()
                    .map(|valore| *valore)
                    .collect::<Vec<f64>>();
                connessioni.push(riga_connessioni.clone());
            }
            strati_rete.push(connessioni);
           
        }

       strati_rete
    }


    /// Metodo che inverte una matrice formata come vettore di vettori
    fn trasponi<T: Clone>(matrice: Vec<Vec<T>>) -> Vec<Vec<T>> {
        if matrice.is_empty() {
            return vec![]; // Se la matrice è vuota, ritorna una matrice vuota.
        }
    
        let _numero_righe = matrice.len();
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
        
        self.funzioni_attivazione.clear();
        
        for line in reader.lines() {
            let linea = line?;
            
            if linea.starts_with(_FILE_INFO_APPRENDIMENTO) {
                let tasso = linea.replace(_FILE_INFO_APPRENDIMENTO, "")
                                    .split(" ")
                                    .collect::<Vec<&str>>().get(1).unwrap()
                                    .to_string().parse::<f64>().unwrap();
                if tasso > 0.0 {
                     self.tasso_apprendimento = tasso;
                }
            } else if linea.starts_with(_FILE_INFO_ATTIVAZIONE) {
                let nomi_funzioni = linea.replace(_FILE_INFO_ATTIVAZIONE, "").trim().to_string();
                for nome_funzione in nomi_funzioni.split("; ").into_iter() {
                    let nome_funzione_modificato = nome_funzione.to_string().replace(";", "").replace(" ", "");
                    let mut _nome_funzione = nome_funzione_modificato.as_str();  
                    if _nome_funzione.trim() != "" {
                        let funzione_attivazione: Arc<dyn FunzioneAttivazione + Send + Sync>= if _nome_funzione.starts_with("LeakyReLU_") {
                            let alfa = _nome_funzione.split("_").collect::<Vec<&str>>()[1].parse::<f64>().unwrap();
                            Arc::new(LeakyReLU { alpha: alfa })
                        } else {
                            match _nome_funzione {
                                "Sigmoide"  => Arc::new(Sigmoide),
                                "ReLU"      => Arc::new(ReLU),
                                "Tanh"      => Arc::new(Tanh),
                                "Softplus"  => Arc::new(Softplus),
                                "Swish"     => Arc::new(Swish),
                                "Lineare"     => Arc::new(Lineare),
                                _           => Arc::new(Nessuna),
                            }
                        };
                        self.funzioni_attivazione.push(funzione_attivazione);
                    }
                }

            } else if linea.starts_with(_FILE_INFO_RETE) {
                let strati = linea.replace(_FILE_INFO_RETE, "").trim()
                        .split(", ")
                        .map( |cifra| cifra.to_string().parse::<usize>().unwrap() )
                        .collect::<Vec<usize>>();
                if strati.len() > 0 {
                    self.dimensioni_strati = strati;    
                    
                    let mut info_strati = Vec::new();
                    let mut i: usize = 0;
                    for neuroni_strato in  self.dimensioni_strati.clone().into_iter() {
                        info_strati.push( 
                            Strato {
                                neuroni: neuroni_strato,
                                funzione_attivazione: self.funzioni_attivazione[i].clone()
                            }
                        );
                        if i < self.funzioni_attivazione.len()  {
                            i += 1;
                        }else {
                            i = 0; // nel caso   self.funzioni_attivazione.len() <  self.dimensioni_strati.len()
                        }
                    }                                                             
                    Self::nuova(info_strati, self.tasso_apprendimento );
                }
            } else if linea.trim() == _FILE_STRATO {
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

    /// Dimensione dei vari strati.
    pub fn strati (&self) ->  Vec<usize> {
        self.dimensioni_strati.to_vec()
    }

    /// Nome della funzione di attivazione degli strati
    /// 
    /// # Argomenti
    ///
    /// * `indice` - è riferito alla posizione nella lista delle funzioni di attivazione associata allo strato,
    /// di norma corrisponde all'indice dello strato (se la lista è uguale o maggiore del numero degli strati)
    /// 
    pub fn funzione_attivazione (&self,indice:usize) ->  &str {
        self.funzioni_attivazione[indice].nome()
    }

    ///Lista di tutte le funzioni di attivazione per strati, dall'input verso l'output.
    /// 
    pub fn lista_funzioni_attivazioni(&self) -> Vec<String> {
        let mut lista: Vec<String> =  vec![];
        for funzione_attivazione in self.funzioni_attivazione.clone().into_iter()  {
            lista.push(funzione_attivazione.sigla().to_string());
        }
        return lista;
    }

    /// Tasso di apprendimento.
    pub fn tasso_apprendimento (&self) ->  f64 {
        self.tasso_apprendimento
    }
}

