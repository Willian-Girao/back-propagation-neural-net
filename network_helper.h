#include "backprop_helper.h"


int random_number(int n) {
	return rand() % (n + 1);
}

void network_learn(datapoint* d1, hidden_layers* backprop_net, output_neuron* out_neuron, double stepsize, int cur_label) {
	/* Forward the given datapoint through the network */
	network_forward(d1, backprop_net, out_neuron);

	double pull = 0.0;

	if (cur_label == 1 && out_neuron->v < 1.0) {
		pull = 1.0;
	} else if (cur_label == -1 && out_neuron->v > -1.0) {
		pull = -1.0;
	}

	//printf("pull: %.2lf\n", pull);

	/* Backprop pass */
	network_backprop(out_neuron, pull, stepsize, d1);
}