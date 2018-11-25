#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "network_helper.h"

#define KGRN  "\x1B[32m"
#define KRED  "\x1B[31m"
#define KYEL  "\x1B[33m"
#define RESET "\x1B[0m"

#define STEPSIZE 0.01

int labels[6] = {1, -1, 1, -1, -1, 1};

double eval_accuracy(datapoint* d1, datapoint* d2, datapoint* d3, datapoint* d4, datapoint* d5, datapoint* d6, hidden_layers* backprop_net, output_neuron* out_neuron) {
	double correct = 0.0;
	int out = 0;

	network_forward(d1, backprop_net, out_neuron);
	out = (out_neuron->v > 0.0) ? 1 : -1;
	correct += (out == labels[0]) ? 1 : 0;

	network_forward(d2, backprop_net, out_neuron);
	out = (out_neuron->v > 0.0) ? 1 : -1;
	correct += (out == labels[1]) ? 1 : 0;

	network_forward(d3, backprop_net, out_neuron);
	out = (out_neuron->v > 0.0) ? 1 : -1;
	correct += (out == labels[2]) ? 1 : 0;

	network_forward(d4, backprop_net, out_neuron);
	out = (out_neuron->v > 0.0) ? 1 : -1;
	correct += (out == labels[3]) ? 1 : 0;

	network_forward(d5, backprop_net, out_neuron);
	out = (out_neuron->v > 0.0) ? 1 : -1;
	correct += (out == labels[4]) ? 1 : 0;

	network_forward(d6, backprop_net, out_neuron);
	out = (out_neuron->v > 0.0) ? 1 : -1;
	correct += (out == labels[5]) ? 1 : 0;

	return correct / 6;
}

int main() {

	time_t t;
	srand((unsigned) time(&t));

	int num_coefs = 3;
	int num_hidden_layers = 1;
	int num_neurons = 10;

	printf("Number of hidden layers: ");
	scanf("%d", &num_hidden_layers);

	printf("Neurons in each layer: ");
	scanf("%d", &num_neurons);

	printf("\n");

	datapoint* d1 = NULL;
	datapoint* d2 = NULL;
	datapoint* d3 = NULL;
	datapoint* d4 = NULL;
	datapoint* d5 = NULL;
	datapoint* d6 = NULL;

	printf(KGRN "[Initializing datapoints ]\n" RESET);
	/* Test datapoints */
	d1 = build_datapoint(1.2, d1);
	d1 = build_datapoint(0.7, d1);

	d2 = build_datapoint(-0.3, d2);
	d2 = build_datapoint(-0.5, d2);

	d3 = build_datapoint(3.0, d3);
	d3 = build_datapoint(0.1, d3);

	d4 = build_datapoint(-0.1, d4);
	d4 = build_datapoint(-1.0, d4);

	d5 = build_datapoint(-1.0, d5);
	d5 = build_datapoint(1.1, d5);

	d6 = build_datapoint(2.1, d6);
	d6 = build_datapoint(-3.0, d6);

	printf("%s[Initializing neurons    ]\n", KGRN);
	/* The output neuron of the network */
	output_neuron* out_neuron = new_output_neuron(0.0, num_neurons);

	printf("%s[Initializing connections]\n", KGRN);
	/* Creating the hidden layers */
	hidden_layers* hidden_layers_set = set_all_hidden_layers(num_hidden_layers, num_neurons, num_coefs, out_neuron);

	printf(RESET);
	printf("\n");
	printf("Learning from data...\n");
	/* Forward data and update weights */
	int pause = 0;
	for (int i = 0; i < 400; ++i) {
		int x = random_number(5);

		if (x == 0) 	{
			network_learn(d1, hidden_layers_set, out_neuron, STEPSIZE, labels[0]);
		} else if (x == 1) 	{
			network_learn(d2, hidden_layers_set, out_neuron, STEPSIZE, labels[1]);
		} else if (x == 2) 	{
			network_learn(d3, hidden_layers_set, out_neuron, STEPSIZE, labels[2]);
		} else if (x == 3) 	{
			network_learn(d4, hidden_layers_set, out_neuron, STEPSIZE, labels[3]);
		} else if (x == 4) 	{
			network_learn(d5, hidden_layers_set, out_neuron, STEPSIZE, labels[4]);
		} else if (x == 5) 	{
			network_learn(d6, hidden_layers_set, out_neuron, STEPSIZE, labels[5]);
		}

		if (i % 25 == 0) {
			double accuracy = eval_accuracy(d1, d2, d3, d4, d5, d6, hidden_layers_set, out_neuron);
			printf("Training accuracy at iteration %d: %.2lf\n", i, accuracy);
		}

		//printf("=========%d==========>%lf\n", x, out_neuron->v);
		//printf("%.2lf\n", out_neuron->v);
		//scanf("%d", &pause);
	}

	double accuracy = eval_accuracy(d1, d2, d3, d4, d5, d6, hidden_layers_set, out_neuron);
	printf("\n");
	printf("Overall network accuracy: %.2lf\n", accuracy);

	return 0;
}





	//print_neuron_coefs(out_neuron->coefs);

	// print_hidden_layers(hidden_layers_set);
	// print_neuron_coefs(out_neuron->coefs);

	/*

	printf("Coeficients before forward: \n");
	print_hidden_layers(hidden_layers_set);
	print_neuron_coefs(out_neuron->coefs);

	printf("Local derivatives after backprop: \n");
	printf("\n");
	print_hidden_layer_neuros_g(hidden_layers_set->layer_k);
	printf("\n");
	print_hidden_layer_neuros_g(hidden_layers_set->next_layer->layer_k);

	printf("Coeficients after forward: \n");
	print_hidden_layers(hidden_layers_set);
	print_neuron_coefs(out_neuron->coefs);

	printf("Outputs during forward: \n");
	printf("\n");
	print_hidden_layer_neuros_output(hidden_layers_set->layer_k);
	printf("\n");
	print_hidden_layer_neuros_output(hidden_layers_set->next_layer->layer_k);


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
