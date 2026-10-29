use plotlars::{Plot, ScatterPlot, Text};
use polars::prelude::*;

use autodiff::grad; // Beispielhaft unter Verwendung der autodiff-Library
fn backprop(net_output : f64, target : f64){
    //
    let dE_output = 2.0*(net_output - target); // ist die Ableitung meiner LOSS, hier MSE...OHNE n, da mathematisch für backward nicht nötig.
    //wenn ReLU genutzt wird: differenzial 1 wenn ReLU aktiviert, 0 wenn ReLU nicht aktiviert
    let d_relu = if net_output > 0.0 { 1.0 } else { 0.0 };
    let dE_dnet = dE_output * d_relu;

    //Hier weiter!!!!

}

fn plot_it(inputs : &Vec<i32>, targets : &Vec<f64>){

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
        .y_title(Text::from("Y Werte (3 * Input + 4)"))
        .build()
        .write_html("mein_plot.html"); // Speichert die Datei im Projektordner!

    if webbrowser::open("mein_plot.html").is_ok() {
        println!("Erfolg! Der Graph sollte sich jetzt in deinem Browser geöffnet haben.");
    } else {
        println!("Das automatische Öffnen wurde blockiert. Bitte öffne '{}' manuell.", "mein_plot.html");
    }
}

fn calculate_target(input_vec : &Vec<i32>)-> Vec<f64>{
    let mut vec_of_target = Vec::new();
    for i in 0..input_vec.len(){
        let curr_target:f64 = 3.0 * input_vec[i] as f64 + 4.0;
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

    struct Nod {
        weights: f64,
        input: f64,
        bias: f64,
        output: f64,

        output_fw: f64,
        output_activation: f64,
    }

    impl Nod {
        fn new(input: f64) -> Nod {
            Nod {
                weights: 0.3,
                input: input,
                bias: 0.0,
                output: 0.0,

                output_fw: 0.0,
                output_activation: 0.0,

            }
        }
        fn forward(&mut self) {
            self.output_fw = self.weights * self.input + self.bias;
        }

        fn relu(&mut self) {
            match self.output_fw {
                x if self.output_fw > 0.0 => self.output_activation = x,
                _ => self.output_activation = 0.0,
            }
        }
        fn sigmoid(&self, output_fw: f64) -> f64 {
            1.0 / (1.0 + (-output_fw).exp())
        }
    }
    struct Layer {
        nodes: Vec<Nod>,
    }
    impl Layer {
        fn new() -> Layer {
            Layer {
                nodes: Vec::new(),
            }
        }
        fn add_node(&mut self, amount: u8, input: f64) {
            for _i in 0..amount {
                let node = Nod::new(input);
                self.nodes.push(node);
            }
        }
    }


struct Netz{
        net : Vec<Layer>,
    }
    impl Netz{
        fn new() -> Netz{
            Netz{
                net : Vec::new(),
            }
        }
    }

    //Probe_Input:
    let vec_of_inputs :Vec<i32> = (1..20).collect();
    println!("{:?}", vec_of_inputs);
    let vec_of_targets = calculate_target(&vec_of_inputs);
    println!("{:?}", vec_of_targets);
    plot_it(&vec_of_inputs,&vec_of_targets);
    let mut my_net = Netz::new();

    const EPOCHS: u32 = 10;

    for i in 0..EPOCHS {
        let mut net_outputs : Vec<f64> = Vec::new();
        let probe_count = vec_of_inputs.len();

        for i in 0..vec_of_inputs.len() {
            let input = vec_of_inputs[i] as f64;
            //Teststruktur als einfaches FC-Forward-Netz------------------------------------
            //Verarbeitung des ERSTEN Layer:
            let mut l1 = Layer::new(); // hier entsteht der erste Layer
            l1.add_node(3, input); //diesem werden Knoten zugefügt (3 Stück und diese Knoten bekommen den Input)
            let mut output_layer1 = 0.0;
            for i in 0..l1.nodes.len() { //hier geht input duch alle Knoten und Aktivierungendieser LAyer
                l1.nodes[i].forward();
                l1.nodes[i].relu();
                println!("{:?}", l1.nodes[i].output_activation);
                output_layer1 += l1.nodes[i].output_activation;
            }
            my_net.net.push(l1);

            //Verarbeitung des zweiten Layer:
            let mut l2 = Layer::new(); // hier entsteht der erste Layer
            l2.add_node(3, output_layer1); //diesem werden Knoten zugefügt (3 Stück und diese Knoten bekommen den Input)
            let mut output_layer2 = 0.0;
            for i in 0..l2.nodes.len() { //hier geht input duch alle Knoten und Aktivierungendieser LAyer
                l2.nodes[i].forward();
                l2.nodes[i].relu();
                println!("{:?}", l2.nodes[i].output_activation);
                output_layer2 += l2.nodes[i].output_activation;
            }
            my_net.net.push(l2);
            //Verarbeitung des dritte Layer:
            let mut l3 = Layer::new(); // hier entsteht der erste Layer
            l3.add_node(3, output_layer2); //diesem werden Knoten zugefügt (3 Stück und diese Knoten bekommen den Input)
            let mut output_layer3 = 0.0;
            for i in 0..l3.nodes.len() { //hier geht input duch alle Knoten und Aktivierungendieser LAyer
                l3.nodes[i].forward();
                l3.nodes[i].relu();
                println!("{:?}", l3.nodes[i].output_activation);
                output_layer3 += l3.nodes[i].output_activation;
            }
            my_net.net.push(l3);
            //Verarbeitung des dritte Layer:
            let mut l4 = Layer::new(); // hier entsteht der erste Layer
            l4.add_node(1, output_layer3); //diesem werden Knoten zugefügt (3 Stück und diese Knoten bekommen den Input)
            let mut output_layer4 = 0.0;
            for i in 0..l4.nodes.len() { //hier geht input duch alle Knoten und Aktivierungendieser LAyer
                l4.nodes[i].forward();
                l4.nodes[i].relu();
                println!("{:?}", l4.nodes[i].output_activation);
                output_layer4 += l4.nodes[i].output_activation;
                net_outputs.push(output_layer4);
            }
            my_net.net.push(l4);
        }
        let mse = mse(net_outputs, &vec_of_targets, probe_count as f64);
        println!("Epoch: {}\n\n", i);
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

