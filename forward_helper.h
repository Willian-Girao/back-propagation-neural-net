#include "hidden_layers_helper.h"
#include "datapoints_helper.h"

/* HELPER FUNCTION - makes a neuron do its calculations */
void operate_neuron(neuron *n_j, datapoint *data) {
	coeficients *aux_y = n_j->coefs;
	datapoint *aux_x = data;
	double result = 0.0;

	while (aux_y) { /* Loop through all the neuron's coeficients and the given datapoint dimensions*/
		result += aux_y->y_i * aux_x->x_i;

		aux_y = aux_y->next;
		if (!aux_y->next) { /* All the "multipliables" coeficients have been updated */
			result += aux_y->y_i;
			n_j->v = result;

			return;
		}
		aux_x = aux_x->next;
	}
}

/* HELPER FUNCTION - creates a "vector" containing all the output values of the neurons in the given layer */
datapoint* layer_neurons_outputs(hidden_layer *layer_k) {
	if (!layer_k) {
		return NULL;
	}
	
	hidden_layer *aux_layer = layer_k;
	datapoint* d = NULL;

	while (aux_layer) {
		d = build_datapoint(aux_layer->n_j->v, d);
		aux_layer = aux_layer->next;
	}
	
	return d;
}

/* Forwards the given that point through the given layer's neurons */
void forward_datapoint_inlayer(hidden_layer *layer_k, datapoint *cur_datapoint) {
	hidden_layer *aux_k = layer_k;

	while (aux_k) { /* Loops through each neuron of the given layer */
		operate_neuron(aux_k->n_j, cur_datapoint);
		aux_k = aux_k->next;
	}
}

/* HELPER FUNCTION - calculates the output of the network */
void calc_network_output(output_neuron *out_neuro, datapoint *data) {
	coeficients *aux_y = out_neuro->coefs;
	datapoint *aux_x = data;
	double result = 0.0;

	while (aux_y) { /* Loop through all the neuron's coeficients and the given datapoint dimensions*/
		result += aux_y->y_i * aux_x->x_i;

		aux_y = aux_y->next;
		if (!aux_y->next) { /* All the "multipliables" coeficients have been updated */
			result += aux_y->y_i;
			out_neuro->v = result;

			return;
		}
		aux_x = aux_x->next;
	}
}

/* Forward a given point throghout the entire network */
void network_forward(datapoint *cur_datapoint, hidden_layers *hidden_layers_set, output_neuron* output_neuron) {
	hidden_layers *hidden_set = hidden_layers_set;
	hidden_layers *aux_last_layer = NULL;

	while (hidden_set) {
		if (!hidden_set->previous_layer) { /* Accessing the 1st layer */
			forward_datapoint_inlayer(hidden_set->layer_k, cur_datapoint); /* Make neurons operate with origial datapoint */
		} else { /* Current layer neuron's must operate with the previous layer neuron's outputs */
			forward_datapoint_inlayer(hidden_set->layer_k, layer_neurons_outputs(hidden_set->previous_layer->layer_k));
		}
		aux_last_layer = hidden_set; /* Saves reference to last processed layer */
		hidden_set = hidden_set->next_layer;
	}

	datapoint *aux = layer_neurons_outputs(aux_last_layer->layer_k); /* Gets the outputs of last hidden layer */
	calc_network_output(output_neuron, aux); /* Compute the network's output */
}

