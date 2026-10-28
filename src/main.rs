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
        fn new(input : f64) -> Nod {
            Nod {
                weights: 0.3,
                input: input,
                bias: 0.0,
                output: 0.0,

                output_fw : 0.0,
                output_activation : 0.0,

            }
        }
        fn forward(& mut self)  {
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

    //Probe_Input:
    let input = 2.0;// x kommt ins Netz

    //Teststruktur als einfaches FC-Forward-Netz------------------------------------

    //Verarbeitung des ERSTEN Layer:
    let mut l1 = Layer::new(); // hier entsteht der erste Layer
    l1.add_node(3, input); //diesem werden Knoten zugefügt (3 Stück und diese Knoten bekommen den Input)
    let mut output_layer1 = 0.0;
    for i in 0..l1.nodes.len() {//hier geht input duch alle Knoten und Aktivierungendieser LAyer
        l1.nodes[i].forward();
        l1.nodes[i].relu();
        println!("{:?}", l1.nodes[i].output_activation);
        output_layer1 += l1.nodes[i].output_activation;
    }

    //Verarbeitung des zweiten Layer:
    let mut l2 = Layer::new(); // hier entsteht der erste Layer
    l2.add_node(3, output_layer1); //diesem werden Knoten zugefügt (3 Stück und diese Knoten bekommen den Input)
    let mut output_layer2 = 0.0;
    for i in 0..l2.nodes.len() {//hier geht input duch alle Knoten und Aktivierungendieser LAyer
        l2.nodes[i].forward();
        l2.nodes[i].relu();
        println!("{:?}", l2.nodes[i].output_activation);
        output_layer2 += l2.nodes[i].output_activation;
    }

    //Verarbeitung des dritte Layer:
    let mut l3 = Layer::new(); // hier entsteht der erste Layer
    l3.add_node(3, output_layer2); //diesem werden Knoten zugefügt (3 Stück und diese Knoten bekommen den Input)
    let mut output_layer3 = 0.0;
    for i in 0..l3.nodes.len() {//hier geht input duch alle Knoten und Aktivierungendieser LAyer
        l3.nodes[i].forward();
        l3.nodes[i].relu();
        println!("{:?}", l3.nodes[i].output_activation);
        output_layer3 += l3.nodes[i].output_activation;
    }

    //Verarbeitung des dritte Layer:
    let mut l4 = Layer::new(); // hier entsteht der erste Layer
    l4.add_node(1, output_layer3); //diesem werden Knoten zugefügt (3 Stück und diese Knoten bekommen den Input)
    let mut output_layer4 = 0.0;
    for i in 0..l4.nodes.len() {//hier geht input duch alle Knoten und Aktivierungendieser LAyer
        l4.nodes[i].forward();
        l4.nodes[i].relu();
        println!("{:?}", l4.nodes[i].output_activation);
        output_layer4 += l4.nodes[i].output_activation;
    }

    let residuum = input - output_layer4;
    println!("{:?}", output_layer4);
    println!("Residdum: {:?}",residuum);

    println!("Als nächstes Backpropagation mit Gradienten und Optimierung");
}

