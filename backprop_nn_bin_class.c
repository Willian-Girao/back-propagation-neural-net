#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "hidden_layers_helper.h"

#define STEPSIZE 0.01

int main() {
	time_t t;
	srand((unsigned) time(&t));

	hidden_layer* layer1 = set_layer(3, 4);
	
	printf("========================================\n");

	print_hidden_layer(layer1);

	return 0;
}