#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "forward_helper.h"

#define STEPSIZE 0.01

/* set_all_hidden_layers(int num_layers, int num_neurons, int num_coefs) */

int main() {
	time_t t;
	srand((unsigned) time(&t));

	hidden_layers* hidden_layers_set = set_all_hidden_layers(3, 2, 2);

	printf("=======================================\n");

	print_hidden_layers(hidden_layers_set);

	return 0;
}