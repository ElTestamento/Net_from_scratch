use plotlars::{Plot, ScatterPlot, Text};
use polars::prelude::*;

use autodiff::grad;
use polars::error::constants::TRUE;
use polars::export::arrow::types::simd::f16x32;
use rand::RngExt;

// Beispielhaft unter Verwendung der autodiff-Library
fn backprop(net_output : f64, target : f64){
    //
    let dE_output = 2.0*(net_output - target); // ist die Ableitung meiner LOSS, hier MSE...OHNE n, da mathematisch für backward nicht nötig.
    //wenn ReLU genutzt wird: differenzial 1 wenn ReLU aktiviert, 0 wenn ReLU nicht aktiviert
    let d_relu = if net_output > 0.0 { 1.0 } else { 0.0 };
    let dE_dnet = dE_output * d_relu;
    //Hier weiter!!!!
}

fn plot_it(inputs : &Vec<i32>, targets : &Vec<f64>, plot_name : &str){

    let mut df = DataFrame::new(vec![
        Series::new("x_achse".into(), inputs.clone()).into(),
        Series::new("y_achse".into(), targets.clone()).into(),
    ]).unwrap();

    ScatterPlot::builder()
        .data(&df)
        .x("x_achse")
        .y("y_achse")
        .plot_title(Text::from("Zielfunktion")
        .font("Arial")
        .size(20)
    )
        .x_title(Text::from("Eingaben"))
        .y_title(Text::from("Y Werte (x * 2 + 5)\n X-Werte (x^2)"))
        .build()
        .write_html(plot_name); // Speichert die Datei im Projektordner!

    if webbrowser::open(plot_name).is_ok() {
        println!("Erfolg! Der Graph sollte sich jetzt in deinem Browser geöffnet haben.");
    } else {
        println!("Das automatische Öffnen wurde blockiert. Bitte öffne '{}' manuell.", "mein_plot.html");
    }
}

fn calculate_target(x_feature_input_vec : &Vec<i32>, y_feature_input_vec : &Vec<i32>)-> Vec<f64>{
    let mut vec_of_target = Vec::new();
    for i in 0..x_feature_input_vec.len(){
        let curr_target:f64 = 3.0 * x_feature_input_vec[i] as f64 + 2.0 * y_feature_input_vec[i] as f64 + 4.0;
        vec_of_target.push(curr_target);
    }
    vec_of_target
}

fn mse(net_preds: Vec<f64>, g_truth: &Vec<f64>)->f64 {
    let n = net_preds.len() as f64;
    let mut pred_sum_to_the_power = 0.0;
    for i in 0..net_preds.len() {
        pred_sum_to_the_power += ((net_preds[i] - g_truth[i]).powi(2));
    }
    let err = (pred_sum_to_the_power / n);
    err
}

fn main() {
    println!("Hello, Net!");
    //wir skalieren stufenweise

    struct Node {
        weights: Vec<f64>,
        input: Vec<f64>,
        bias: f64,
        output_activation: f64,
        output_fw: f64,
    }

    impl Node {
        fn new() -> Node {
            Node {
                weights: vec![0.3; 0],
                input: vec![0.1; 0],
                bias: 0.0,
                output_fw: 0.0,
                output_activation: 0.0,
            }
        }
        fn add_weights(&mut self, rand_seed: f64 ,amount: usize) {
            self.weights.extend(std::iter::repeat_n(rand_seed, amount));
        }
        fn ingest_data(&mut self, input: f64) {
            self.input.push(input);
        }
        fn forward(&mut self) { //hier werden nacheinander die inputs der Vorschicht mit den jeweiligen Gewichten linear aufadiert und dann mit dem bias addiert und an den Output weitergereciht.
            let mut output_per_linear_term = 0.0;
            for i in 0..self.weights.len() {
                output_per_linear_term += self.weights[i] * self.input[i];
            }
            self.output_fw = output_per_linear_term + self.bias;
        }

        fn relu(&mut self) {
            match self.output_fw {
                x if self.output_fw > 0.0 => self.output_activation = x,
                _ => self.output_activation = 0.0,
            }
        }
        fn sigmoid(&mut self) {
            self.output_activation = 1.0 / (1.0 + (-self.output_fw).exp());
        }
    }

    struct Layer {
        nodes: Vec<Node>,
        use_activation_function: bool,
        activation_function: String,
    }
    impl Layer {
        fn new() -> Layer {
            Layer {
                nodes: Vec::new(),
                use_activation_function: false,
                activation_function: String::from("relu"),
            }
        }
        fn add_activation_function(&mut self, activation_function: String) {
            match activation_function.as_str() {
                "sigmoid" => {self.activation_function = String::from("sigmoid"); self.use_activation_function = true;},
                "relu" => {self.activation_function = String::from("relu"); self.use_activation_function = true;},
                _ => self.use_activation_function = false,
            }
        }
        fn add_hidden_node(&mut self, amount: u8, connect_to_nodes: usize)->usize {
            for _i in 0..amount {
                let mut node = Node::new();
                let initial_random_weight = rand::random_range(0.0..0.4); //beim initialisieren von neuen Nodes werden die gewichte so random initialisiert
                node.add_weights(initial_random_weight, connect_to_nodes);
                self.nodes.push(node);
            }
            amount as usize
        }
        fn get_output_activation(&self) -> Vec<f64> {
            let mut output_vecs: Vec<f64> = Vec::new();
            for node in self.nodes.iter() {
                output_vecs.push(node.output_activation);
            }
            output_vecs
        }
        fn calculation_node_output(&mut self, prev_activations: &[f64]) {
            for node in self.nodes.iter_mut() {
                node.input.clear();
                for i in 0..prev_activations.len() {
                    node.ingest_data(prev_activations[i]);
                }
                node.forward();
                if self.use_activation_function == true {
                    match self.activation_function.as_str() {
                        "relu" => { node.relu() },
                        "sigmoid" => { node.sigmoid() },
                        _ => { node.output_activation = node.output_fw; }
                    }
                }else{
                        node.output_activation = node.output_fw;
                }
            }
        }
    }

    //Wir bauen die Daten!
    let data_count = 20;
    //Probe_Input und Vorbereitung der Datenproben:
    let x_features_vec_of_inputs: Vec<i32> = (1..data_count).map(|x: i32| x.pow(2)).collect(); //Merkmal x bauen
    let y_features_vec_of_inputs: Vec<i32> = (1..data_count).map(|x| (x * 2 + 5)).collect(); //Merkmal y bauen

    let dataframe_2d: Vec<Vec<i32>> = x_features_vec_of_inputs
        .iter().copied()
        .zip(y_features_vec_of_inputs.iter().copied())
        .map(|(x, y)| vec![x, y])
        .collect();
    let dataframe_length = dataframe_2d.len();
    let feature_count = dataframe_2d[0].len(); //Anzahl der Feautures dynamisch gemessen des DF
    println!("{:?}", x_features_vec_of_inputs);
    println!("{:?}", y_features_vec_of_inputs);
    let vec_of_targets = calculate_target(&x_features_vec_of_inputs, &y_features_vec_of_inputs); // output z errechnen -> GT
    println!("{:?}", vec_of_targets);
    let plot_name = "plot1.html";
    let x_achse: Vec<i32> = (1..data_count).collect(); //Merkmal x bauen
    plot_it(&x_achse, &vec_of_targets, plot_name);
    //Daten gebaut und geplottet!

    //Teststruktur als einfaches FC-Forward-Netz------------------------------------
    //Input Layer durch Anzahl der Merkmale definiert hier x und y;
    //dann 3x3 hiodden layer konvergieren auf ein Output.

    let mut amount_of_weights = feature_count;
    let mut l1 = Layer::new();
    amount_of_weights =l1.add_hidden_node(3, amount_of_weights);
    l1.add_activation_function(String::from("relu"));
    let mut l2 = Layer::new();
    amount_of_weights =l2.add_hidden_node(3, amount_of_weights);
    l2.add_activation_function(String::from("relu"));
    let mut l3 = Layer::new();
    amount_of_weights = l3.add_hidden_node(3, amount_of_weights);
    l3.add_activation_function(String::from("relu"));
    let mut l4 = Layer::new();
    _ =l4.add_hidden_node(1, amount_of_weights);

    //wir haben also hier ein Merkmal X-Wert und ein Merkmal Y-Wert, mit dem Ergebnis Z (vec_of_targets).
    const EPOCHS: u32 = 10;

    for epoch in 0..EPOCHS {
        let mut net_output: Vec<f64> = Vec::new();
        for i in 0..(data_count as usize - 1) {
            //Verarbeitung des ersten, also INPUT Layer: Hier braucht es keine Nodes, es repräsentiert die Merkmale. Es finden keine Rechnugnen statt.
            let probe_input = vec![dataframe_2d[i][0] as f64, dataframe_2d[i][1] as f64]; //Merkmal x und y der ersten Probe.

            //Verarbeitung des ERSTEN Layer:
            // hier starten schon die hidden Layer und er nimmt den Input pro Node aus dem Vorlayer(Input-Layer):
            l1.calculation_node_output(&probe_input);

            //Verarbeitung des zweiten Layer:
            let out= l1.get_output_activation();
            l2.calculation_node_output(&out);

            //Verarbeitung des dritte Layer:
            let out= l2.get_output_activation();
            l3.calculation_node_output(&out);

            //Verarbeitung OUTPUT Layer:
            let out= l3.get_output_activation();
            l4.calculation_node_output(&out);//hier nur ein Output im Vec, weil nur ein Output das NEtz verlässt
            net_output.push(l4.nodes[0].output_activation);
        }
        let mse = mse(net_output, &vec_of_targets);
        println!("Epoch: {}\n\n", epoch);
        println!("LOSS_Fn ist MSE");
        println!("LOSS: {}", mse);
        println!("Als nächstes Backpropagation mit Gradienten und Optimierung");

    }
}

// STAND 31.10.:
// Forward-Pass läuft korrekt: Input ist ein Vec<f64> (keine Input-Layer-Nodes mehr),
// jede Schicht liest den kompletten Aktivierungsvektor der Vorschicht (get_output_activation),
// Aktivierung pro Layer konfigurierbar, l4 ohne Aktivierung (linear, für Regression).
// Loss ist konstant über die Epochen -> erwartet, es wird noch nichts gelernt.
//
// NÄCHSTE SCHRITTE:
// 1. Eine Probe per Hand gegen den Print nachrechnen.
// 2. Backprop nur für l4: dE/dw und dE/db auf Papier, dann in Code. Erst danach die Kette nach hinten.
//    Dafür braucht jeder Node seinen letzten Input (hat er: node.input) und die Ableitung seiner Aktivierung.
//
// SPÄTER (nicht dringend):
// - Aktivierung als enum statt String+bool (der "_"-Arm im match ist aktuell unerreichbar)
// - Gewichte innerhalb eines Nodes sind alle identisch (repeat_n) -> Zufallszug in die Schleife
// - Net als Vec<Layer> mit Schleife über die Layer statt l1..l4 von Hand

