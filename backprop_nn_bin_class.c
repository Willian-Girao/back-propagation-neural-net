#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "backprop_helper.h"

#define STEPSIZE 0.01

/* set_all_hidden_layers(int num_layers, int num_neurons, int num_coefs) */
/* make_layer_respond_to_pull(hidden_layer *layer_k, datapoint *data, double pull) */
/* forward_datapoint_inlayer(hidden_layer *layer_k, datapoint *cur_datapoint) */
/* network_forward(hidden_layers *hidden_layers_set, datapoint *cur_datapoint) */

int main() {
	time_t t;
	srand((unsigned) time(&t));
	int num_coefs = 2;
	int num_hidden_layers = 2;
	int num_neurons = 2;

	/* Creating the hidden layers */
	hidden_layers* hidden_layers_set = set_all_hidden_layers(num_hidden_layers, num_neurons, num_coefs);
	
	/* The output neuron of the network */
	neuron* output_neuron = new_neuron(0.0, 1.0, (num_neurons + 1));

	print_hidden_layers(hidden_layers_set);

	datapoint* d1 = NULL;
	d1 = build_datapoint(2.0, NULL);

	/* Forward the given datapoint through the network */
	network_forward(d1, hidden_layers_set, output_neuron);

	print_neurons_outputs(hidden_layers_set);

	printf("=======================================\n");

	print_neuron_coefs(output_neuron->coefs);
	printf("Networks output: %.2lf\n", output_neuron->v);

	/*
	print_neurons_outputs(hidden_layers_set);

	datapoint* outs = layer_neurons_outputs(hidden_layers_set->layer_k);
	print_datapoint(outs);

	forward_datapoint_inlayer(hidden_layers_set->layer_k, d1);	

	print_hidden_layers(hidden_layers_set);
	d1 = build_datapoint(4.0, d1);
	print_datapoint(d1);

	make_layer_respond_to_pull(hidden_layers_set->layer_k, d1, 1.0);
	print_hidden_layers(hidden_layers_set);
	*/

	return 0;
}