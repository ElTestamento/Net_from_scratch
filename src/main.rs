use plotlars::{Plot, ScatterPlot, Text};
use polars::prelude::*;

use autodiff::grad;
use polars::export::arrow::types::simd::f16x32;

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

fn mse(net_preds: Vec<f64>, g_truth: &Vec<f64>, n : f64)->f64 {
    let mut pred_sum_to_the_power = 0.0;
    for i in 0..net_preds.len() {
        pred_sum_to_the_power = ((net_preds[i] - g_truth[i]).powi(2));
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
        fn add_weights(&mut self, amount: usize) {
            let initial_random_weight = rand::random_range(0.0..0.4); //beim initialisieren von neuen Nodes werden die gewichte so random initialisiert
            self.weights.extend(std::iter::repeat_n(initial_random_weight, amount));
        }
        fn ingest_data(&mut self, input: f64) {
            self.input.push(input);
        }
        fn forward(&mut self) { //hier werden nacheinander die inputs der Vorschicht mit den jeweiligen Gewichten linear aufadiert und dann mit dem bias addiert und an den Output weitergereciht.
            let mut output_per_linear_term = 0.0;
            for i in 0..self.weights.len() {
                output_per_linear_term = self.weights[i] * self.input[i];
            }
            self.output_fw = output_per_linear_term + self.bias;
            println!("Er rechnet als Forward-Formel (weights * input = forward_out): {:?} * {:?} = {:?}", self.weights, self.input, self.output_fw);
        }

        fn relu(&mut self) {
            match self.output_fw {
                x if self.output_fw > 0.0 => self.output_activation = x,
                _ => self.output_activation = 0.0,
            }
            println!("Er rechnet als ReLU-Formel (output_fw {:?} > 0.0, dann ReLU-output wie forward_out {:?} sonst RelU-output = 0.0)=> ReLU : {:?} ", self.output_fw,self.output_fw, self.output_activation);
        }
        fn sigmoid(&self, output_fw: f64) -> f64 {
            1.0 / (1.0 + (-output_fw).exp())
        }
    }

    struct Layer {
        nodes: Vec<Node>,
    }
    impl Layer {
        fn new() -> Layer {
            Layer {
                nodes: Vec::new(),
            }
        }
        fn add_input_node(&mut self, features : u8) ->usize {
            for _i in 0..features {
                let mut node = Node::new();
                node.add_weights(features as usize);
                self.nodes.push(node);
            }
            features as usize
        }
        fn add_hidden_node(&mut self, amount: u8, connect_to_nodes: usize)->usize {
            for _i in 0..amount {
                let mut node = Node::new();
                node.add_weights(connect_to_nodes);
                self.nodes.push(node);
            }
            amount as usize
        }
        fn calculation_node_output(&mut self, prev_l: &Layer) {
            for node in self.nodes.iter_mut() {
                node.input.clear();
                for i in 0..prev_l.nodes.len() {
                    node.ingest_data(prev_l.nodes[i].output_activation);
                }
                node.forward();
                node.relu();
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
    //let mut my_net = Net::new();
    let mut input_layer = Layer::new();
    let mut amount_of_weights = input_layer.add_input_node(feature_count as u8);
    let mut l1 = Layer::new();
    amount_of_weights =l1.add_hidden_node(3, amount_of_weights);
    let mut l2 = Layer::new();
    amount_of_weights =l2.add_hidden_node(3, amount_of_weights);
    let mut l3 = Layer::new();
    amount_of_weights = l3.add_hidden_node(3, amount_of_weights);
    let mut l4 = Layer::new();
    amount_of_weights = l4.add_hidden_node(1, 1);

    //wir haben also hier ein Merkmal X-Wert und ein Merkmal Y-Wert, mit dem Ergebnis Z (vec_of_targets).
    const EPOCHS: u32 = 10;

    for epoch in 0..EPOCHS {
        let mut net_output: Vec<f64> = Vec::new();
        for i in 0..(data_count as usize - 1) {
            //Verarbeitung des ersten, also INPUT Layer: Hier braucht es keine Nodes, es repräsentiert die Merkmale. Es finden keine Rechnugnen statt.
            let probe_input = vec![dataframe_2d[i][0], dataframe_2d[i][1]]; //Merkmal x und y der ersten Probe.

            //Daten wandern Merkmal nach Merkmal pro Probe in das Netz
            for i in 0..input_layer.nodes.len() {
                input_layer.nodes[i].ingest_data(probe_input[i] as f64);
                input_layer.nodes[i].output_activation = probe_input[i] as f64;
            }
            //Verarbeitung des ERSTEN Layer:
            // hier starten schon die hidden Layer und er nimmt den Input pro Node aus dem Vorlayer(Input-Layer):
            l1.calculation_node_output(&input_layer);

            //Verarbeitung des zweiten Layer:
            l2.calculation_node_output(&l1);

            //Verarbeitung des dritte Layer:
            l3.calculation_node_output(&l2);

            //Verarbeitung OUTPUT Layer:
            l4.calculation_node_output(&l3);//hier nur ein Output im Vec, weil nur ein Output das NEtz verlässt
            net_output.push(l4.nodes[0].output_activation);
        }
        let mse = mse(net_output, &vec_of_targets, data_count as f64);
        println!("Epoch: {}\n\n", epoch);
        println!("LOSS_Fn ist MSE");
        println!("LOSS: {}", mse);
        println!("Als nächstes Backpropagation mit Gradienten und Optimierung");

    }
}



//
// Hier müssen wir die Definition ganz sauber trennen, sonst gerät die gesamte Backpropagation im Code durcheinander.
// Ein einzelnes Neuron hat nicht nur ein einziges Gewicht, sondern ein eigenes Gewicht für jeden einzelnen Input, den es empfängt.
// Wenn ein Neuron also Signale von 3 Neuronen aus der vorherigen Schicht bekommt, besitzt dieses eine Neuron genau 3 Gewichte (und einen Bias).
// Schauen wir uns das visuell und im Code an, damit es völlig eindeutig wird:
//
// Das mathematische Modell eines Neurons
//
// Wenn ein Neuron Inputs von einer vorherigen Schicht mit 3 Neuronen erhält, sieht seine Berechnung im Forward-Pass so aus:
// \(\text{Output}=\text{ReLU}(w_{1}\cdot x_{1}+w_{2}\cdot x_{2}+w_{3}\cdot x_{3}+b)\)
// • \(x_1, x_2, x_3\) sind die Outputs der 3 Neuronen aus der Schicht davor.
// • \(w_1, w_2, w_3\) sind die 3 Gewichte dieses einen Neurons. Jedes Gewicht gehört zu genau einer Verbindung.


//Das muss ich im Code prüfen!!!!!! Könntesein ,dass meine Implemetierung des forwar-pass falsch ist

