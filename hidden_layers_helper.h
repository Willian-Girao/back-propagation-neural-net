#include "neurons_helper.h"

typedef struct lay {
	neuron *n_j;
	struct lay *next;
} hidden_layer;

hidden_layer* new_layer_neuron(neuron *n) {
	hidden_layer* new_layer_elem = (hidden_layer*)malloc(sizeof(hidden_layer*));

	new_layer_elem->n_j = n;
	new_layer_elem->next = NULL;	

	return new_layer_elem;
}

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