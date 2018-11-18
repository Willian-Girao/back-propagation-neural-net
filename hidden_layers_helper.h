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

void print_hidden_layer(hidden_layer* l) {
	hidden_layer* x = l;
	while(x) {
		print_neuron_coefs(x->n_j->coefs);
		x = x->next;
	}
}

typedef struct layers {
	hidden_layer *layer_k;
	struct layers *next_layer;
	struct layers *previous_layer;
} hidden_layers;

hidden_layers* set_layers_element(hidden_layer *l_k, hidden_layers *previous) {
	hidden_layers* layers_set_element = (hidden_layers*)malloc(sizeof(hidden_layers*));

	layers_set_element->layer_k = l_k;
	layers_set_element->previous_layer = previous;
	layers_set_element->next_layer = NULL;

	return layers_set_element;
}

hidden_layers* insert_into_hidden_layers_set(hidden_layers *root, hidden_layer *new_layer) {
	if (!root) {
		return set_layers_element(new_layer, NULL);
	}

	hidden_layers *aux = root;
	while(aux->next_layer) {
		aux = aux->next_layer;
	}
	aux->next_layer = set_layers_element(new_layer, aux);

	return root;
}

hidden_layers* set_all_hidden_layers(int num_layers, int num_neurons, int num_coefs) {
	hidden_layers *root = NULL;

	for (int k = 0; k < num_layers; ++k) {
		hidden_layer* new_layer_k = set_layer(num_neurons, num_coefs);
		print_hidden_layer(new_layer_k);
		printf("\n");
		root = insert_into_hidden_layers_set(root, new_layer_k);
	}

	return root;
}

void print_hidden_layers(hidden_layers* root) {
	hidden_layers* aux = root;
	while(aux) {
		print_hidden_layer(aux->layer_k);
		printf("\n");
		aux = aux->next_layer;
	}
}