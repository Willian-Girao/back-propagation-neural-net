#include "neurons_helper.h"


/* A layer is made of a list of neurons */
typedef struct lay {
	neuron *n_j;
	struct lay *next;
} hidden_layer;

/* Creates a neuron to later be inserted into a layer */
hidden_layer* new_layer_neuron(neuron *n) {
	hidden_layer* new_layer_elem = (hidden_layer*)malloc(sizeof(hidden_layer*));

	new_layer_elem->n_j = n;
	new_layer_elem->next = NULL;	

	return new_layer_elem;
}

/* Inserts the neuron 'n' into the layer 'root' */
hidden_layer* insert_into_layer(hidden_layer *root, neuron *n) {
	if (!root) {
		return new_layer_neuron(n);
	}

	hidden_layer *aux = root;
	while(aux->next) {
		aux = aux->next;
	}
	aux->next = new_layer_neuron(n);

	return root;
}

/* Generates a whole hidden layer */
hidden_layer* set_layer(int num_neurons, int num_coefs) {

	hidden_layer* layer = NULL;

	for (int j = 0; j < num_neurons; ++j) {
		layer = insert_into_layer(layer, new_neuron(0.0, 0.0, num_coefs));
	}

	return layer;
}

/* Prints the coeficients of all the neurons in a sigle hidden layer */
void print_hidden_layer(hidden_layer* l) {
	hidden_layer* x = l;
	while(x) {
		print_neuron_coefs(x->n_j->coefs);
		x = x->next;
	}
}

/* Prints the output values of all the neurons in a sigle hidden layer */
void print_hidden_layer_neuros_output(hidden_layer* l) {
	hidden_layer* x = l;
	while(x) {
		printf("%.2lf\n", x->n_j->v);
		x = x->next;
	}
}

/* Prints the local derivatives of all the neurons in a sigle hidden layer */
void print_hidden_layer_neuros_g(hidden_layer* l) {
	hidden_layer* x = l;
	while(x) {
		printf("%.2lf\n", x->n_j->g);
		x = x->next;
	}
}

typedef struct layers {
	hidden_layer *layer_k;
	struct layers *next_layer;
	struct layers *previous_layer;
} hidden_layers;

/* Structure representing a output neuron of the netwokr */
typedef struct out_neu {
	double v; /* Neuron's output value */
	struct coefs *coefs; /* This neuron's coeficients */
	struct layers *last_layer; /* A pointer to the last hidden layer that connects to the output neuron */
} output_neuron;

/* Helper function to create/allocate an output neuron */
output_neuron* new_output_neuron(double v, int num_neurons) {
	output_neuron* new_neuron = (output_neuron*)malloc(sizeof(output_neuron*));

	new_neuron->v = v;
	new_neuron->coefs = init_neurons_coefs(num_neurons + 1);
	new_neuron->last_layer = NULL;

	return new_neuron;
}

/* Populates the sctructure utilized to build up the hidden layers set */
hidden_layers* set_layers_element(hidden_layer *l_k, hidden_layers *previous) {
	hidden_layers* layers_set_element = (hidden_layers*)malloc(sizeof(hidden_layers*));

	layers_set_element->layer_k = l_k;
	layers_set_element->previous_layer = previous;
	layers_set_element->next_layer = NULL;

	return layers_set_element;
}

/* Inserts a hidden layer into the correct position ino the set of hidden layers */
hidden_layers* insert_into_hidden_layers_set(hidden_layers *root, hidden_layer *new_layer, output_neuron* out_neuron) {
	if (!root) {
		hidden_layers* aux = set_layers_element(new_layer, NULL);
		out_neuron->last_layer = aux;
		return aux;
	}

	hidden_layers *aux = root;
	while(aux->next_layer) {
		aux = aux->next_layer;
	}
	hidden_layers* last = set_layers_element(new_layer, aux);
	out_neuron->last_layer = last;
	aux->next_layer = last;

	return root;
}

/* Sets up all the hidden layers */
hidden_layers* set_all_hidden_layers(int num_layers, int num_neurons, int num_coefs, output_neuron* out_neuron) {
	hidden_layers *root = NULL;

	for (int k = 0; k < num_layers; ++k) {
		if (!root) { /* 1st layer - number of coeficients needed is proportional to the dimension of the datapoint */
			root = insert_into_hidden_layers_set(root, set_layer(num_neurons, num_coefs), out_neuron);
		} else { /* k-th layer - number of coeficients needed is proportional to the number of neurons in the previous layer */
			root = insert_into_hidden_layers_set(root, set_layer(num_neurons, (num_neurons + 1)), out_neuron);
		}
	}

	return root;
}

/* Prints the coeficients of aech neuron in each hidden layer*/
void print_hidden_layers(hidden_layers* root) {
	hidden_layers* aux = root;
	while(aux) {
		print_hidden_layer(aux->layer_k);
		printf("\n");
		aux = aux->next_layer;
	}
}

/* Prints the output values of aech neuron in each hidden layer*/
void print_neurons_outputs(hidden_layers* root) {
	hidden_layers* aux = root;
	while(aux) {
		print_hidden_layer_neuros_output(aux->layer_k);
		printf("\n");
		aux = aux->next_layer;
	}
}




