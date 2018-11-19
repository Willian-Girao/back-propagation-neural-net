#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "backprop_helper.h"

#define STEPSIZE 0.01

/* set_all_hidden_layers(int num_layers, int num_neurons, int num_coefs) */
/* make_layer_respond_to_pull(hidden_layer *layer_k, datapoint *data, double pull) */
/* forward_datapoint_inlayer(hidden_layer *layer_k, datapoint *cur_datapoint) */
/* network_forward(hidden_layers *hidden_layers_set, datapoint *cur_datapoint, double stepsize) */

int main() {

	time_t t;
	srand((unsigned) time(&t));

	int num_coefs = 2;
	int num_hidden_layers = 2;
	int num_neurons = 2;

	datapoint* d1 = NULL;
	d1 = build_datapoint(2.0, NULL);

	/* The output neuron of the network */
	output_neuron* out_neuron = new_output_neuron(0.0, num_neurons);

	/* Creating the hidden layers */
	hidden_layers* hidden_layers_set = set_all_hidden_layers(num_hidden_layers, num_neurons, num_coefs, out_neuron);

	printf("Coeficients before forward: \n");
	print_hidden_layers(hidden_layers_set);
	print_neuron_coefs(out_neuron->coefs);

	/* Forward the given datapoint through the network */
	network_forward(d1, hidden_layers_set, out_neuron);

	printf("Outputs during forward: \n");
	printf("\n");
	print_hidden_layer_neuros_output(hidden_layers_set->layer_k);
	printf("\n");
	print_hidden_layer_neuros_output(hidden_layers_set->next_layer->layer_k);

	/* Backprop pass */
	network_backprop(out_neuron, 1.0, STEPSIZE, d1);

	printf("=======================================\n");

	//printf("Networks output: %.2lf\n", out_neuron->v);
	printf("Local derivatives after backprop: \n");
	printf("\n");
	print_hidden_layer_neuros_g(hidden_layers_set->layer_k);
	printf("\n");
	print_hidden_layer_neuros_g(hidden_layers_set->next_layer->layer_k);

	printf("Coeficients after forward: \n");
	print_hidden_layers(hidden_layers_set);
	print_neuron_coefs(out_neuron->coefs);

	/*


	printf("\n");
	print_hidden_layer_neuros_output(hidden_layers_set->layer_k);
	printf("\n");
	print_hidden_layer_neuros_output(hidden_layers_set->next_layer->layer_k);

	printf("\n");
	print_hidden_layer_neuros_g(hidden_layers_set->layer_k);
	printf("\n");
	print_hidden_layer_neuros_g(hidden_layers_set->next_layer->layer_k);








	print_hidden_layers(hidden_layers_set);
	// printf("Output neuron coeficiens: \n");
	// print_neuron_coefs(out_neuron->coefs);


	printf("Last hidden layer neuros outputs: \n");
	print_hidden_layer_neuros_output(hidden_layers_set->next_layer->layer_k);
	printf("Output neuron coeficiens: \n");
	print_neuron_coefs(out_neuron->coefs);
	printf("coeficiens of all neurons in the hiddden layer set: \n");
	print_hidden_layers(hidden_layers_set);


	print_hidden_layer(hidden_layers_set->next_layer->layer_k);
	printf("\n");
	print_hidden_layer(out_neuron->last_layer->layer_k);

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