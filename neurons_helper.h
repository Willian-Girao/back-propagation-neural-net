#include <time.h>
#define COEF_INTERVAL 0.5

/* Structure representing a neuron */
typedef struct units {
	double v; /* Neuron's output value */
	double g; /* Neuros's gradient to be utilized to update its coeficients (pull from above in the chain) */
	struct coefs *coefs; /* This neuron's coeficients */
} neuron;

/* List of coeficients need for the network */
typedef struct coefs {
	double y_i; /* Value */
	struct coefs *next; /* Pointer to other coeficients */
} coeficients;

double initial_random_coef() {
	double random = ((double)rand() / (double)RAND_MAX);
	return (random >= 0.5) ? random * COEF_INTERVAL : -random * COEF_INTERVAL;
}

/* Initialize a neuron's coeficients */
coeficients* init_neurons_coefs(int num_coefs) {

	coeficients* head_coef = (coeficients*)malloc(sizeof(coeficients*));
	head_coef->y_i = initial_random_coef();

	for (int i = 0; i < num_coefs-1; ++i)	{
		coeficients* new_coef = (coeficients*)malloc(sizeof(coeficients*));
		new_coef->y_i = initial_random_coef();

		coeficients* aux = head_coef;
		while(aux->next) {
			aux = aux->next;
		}

		aux->next = new_coef;
	}

	return head_coef;
}

/* Helper function to create/allocate a neuron */
neuron* new_neuron(double v, double g, int num_coefs) {
	neuron* new_neuron = (neuron*)malloc(sizeof(neuron*));

	new_neuron->v = v;
	new_neuron->g = g;
	new_neuron->coefs = init_neurons_coefs(num_coefs);

	return new_neuron;
}

/* Prints the set of coeficients of a neuron */
void print_neuron_coefs(coeficients *n) {
	coeficients* aux = n;
	while(aux) {
		printf("%lf ", aux->y_i);
		aux = aux->next;
	}
	printf("\n");
}