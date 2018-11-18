#include "hidden_layers_helper.h"
#include "datapoints_helper.h"

/* Sums all the gradients of a given hidden layer */
double sum_gradients_from_layer(hidden_layer *layer_k) {
	double sum = 0.0;
	hidden_layer *aux_k = layer_k;

	while(aux_k) {
		sum += aux_k->n_j;
		aux_k = aux_k->next;
	}

	return sum;
}

/* HELPER FUNCTION - Updates the coeficients of a single neuron */
void update_neuron_coeficients(neuron *n_j, datapoint *data, double pull) {
	coeficients *aux_y = n_j->coefs;
	datapoint *aux_x = data;

	while (aux_y) { /* Loop through all the neuron's coeficients and he given datapoint dimensions*/
		aux_y += (aux_x->characteristic * (pull - aux_y));

		aux_y = aux_y->next;
		if (!aux_x->next) { /* All the multiplied coeficients have been updated */
			aux_y += 1.0 * pull; /* Last neuron coeficient responds to the pull by the partial derivative of a sum */
			return;
		}
		aux_x = aux_x->next;
	}
}

/* Updates the coeficienst of all the neurons in thegiven layer layer */
void update_coefs(hidden_layer *layer_k, datapoint *data, double pull) {
	hidden_layer *aux_k = layer_k;

	while (aux_k) { /* Loops through each neuron of the given layer */
		update_neuron_coeficients(aux_k->n_j, data, pull);
		aux_k = aux_k->next;
	}
}