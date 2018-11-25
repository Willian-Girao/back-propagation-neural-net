#include "forward_helper.h"

/* Sums all the gradients of a given hidden layer */
double sum_gradients_from_layer(hidden_layer *layer_k) {
	double sum = 0.0;
	hidden_layer *aux_k = layer_k;

	while(aux_k) {
		sum += aux_k->n_j->g;
		aux_k = aux_k->next;
	}

	return sum;
}

/* HELPER FUNCTION - Updates the coeficients of a single neuron */
void make_neuron_respond_to_pull(neuron *n_j, datapoint *data, double stepsize) {
	coeficients *aux_y = n_j->coefs;
	datapoint *aux_x = data;

	while (aux_y) { /* Loop through all the neuron's coeficients and the given variables */
		aux_y->y_i +=  stepsize * (aux_x->x_i * n_j->g - aux_y->y_i);

		aux_y = aux_y->next;
		if (!aux_y->next) { /* All the "multipliables" coeficients have been updated */
			aux_y->y_i += stepsize * (1.0 * n_j->g); /* Last neuron coeficient responds to the pull by the partial derivative of a sum */
			return;
		}
		aux_x = aux_x->next;
	}
}

/* Updates the coeficienst of all the neurons in the given layer */
void make_layer_respond_to_pull(hidden_layer *layer_k, datapoint *data, double stepsize) {
	hidden_layer *aux_k = layer_k;

	while (aux_k) { /* Loops through each neuron of the given layer */
		make_neuron_respond_to_pull(aux_k->n_j, data, stepsize);
		aux_k = aux_k->next;
	}
}

/* HELPER FUNCTION - updates the output neuron's coeficients */
datapoint* update_output_neuron_coeficients(output_neuron *out_neuron, datapoint *last_layer_outputs, double pull, double stepsize) {
	
	coeficients *out_neuron_aux = out_neuron->coefs;
	datapoint* to_backprop = NULL;

	while (out_neuron_aux) { /* Loop through all the neuron's coeficients and the last layers outputs */
		double before_change = out_neuron_aux->y_i;
		out_neuron_aux->y_i += stepsize * ((last_layer_outputs->x_i * pull) + (-out_neuron_aux->y_i)); /* Update output neuron coeficients */
		//printf("^%lf^\n", stepsize * ((last_layer_outputs->x_i * pull) + (-out_neuron_aux->y_i)));
		to_backprop = build_datapoint((before_change * pull), to_backprop); /* Saves the pull that will be sent to the neurons n_i in the previous layer */

		out_neuron_aux = out_neuron_aux->next;
		if (!out_neuron_aux->next) { /* All the "multipliables" coeficients have been updated */
			//printf("^%lf^\n", stepsize * (1.0 * pull));
			out_neuron_aux->y_i += stepsize * (1.0 * pull); /* Last neuron coeficient responds to the pull by the partial derivative of a sum */

			return to_backprop;
		}
		last_layer_outputs = last_layer_outputs->next;
	}
}

/* Updates all the pulls sent from above for all the neurons in the given layer */
void update_neuron_pull(hidden_layers *layer, datapoint *pulls_from_above) {
	hidden_layer *aux_layer = layer->layer_k;

	while(aux_layer) {
		aux_layer->n_j->g = (aux_layer->n_j->v == 0.0) ? 0.0 : pulls_from_above->x_i;
		//printf("(%.2lf)\n", aux_layer->n_j->g);
		aux_layer = aux_layer->next;
		pulls_from_above = pulls_from_above->next;
	}
}

void set_existing_pull_to_zero(hidden_layer *layer) {
	hidden_layer *layer_aux = layer;

	while (layer_aux) {
		layer_aux->n_j->g = 0.0;
		layer_aux = layer_aux->next;
	}
}

void backpropagate_pull_update(hidden_layers *curr_layer) {
	if (!curr_layer->previous_layer) {
		return;
	}

	/* Resete existing gradients in the previous hidde layer */
	set_existing_pull_to_zero(curr_layer->previous_layer->layer_k);

	hidden_layer *curr_layer_aux = curr_layer->layer_k;
	hidden_layer *previous_layer_aux = curr_layer->previous_layer->layer_k;

	while(curr_layer_aux) {
		double g = curr_layer_aux->n_j->g; /* current layer's nth neuron gradient */
		coeficients *curr_coefs = curr_layer_aux->n_j->coefs; /* curret layer's nth neuron's jth coeficient */

		while (curr_coefs) {
			previous_layer_aux->n_j->g += g * curr_coefs->y_i; /* jth coeficient multiplied by the current neurons g goes to the nth neuron's g in the previous layer */

			curr_coefs = curr_coefs->next;
			if (!curr_coefs->next) { /* All the neurons in the previous layer have been updated by the nth neuron's g in the current layer */
				break;
			}
			previous_layer_aux = previous_layer_aux->next;
		}

		curr_layer_aux = curr_layer_aux->next;
		previous_layer_aux =  curr_layer->previous_layer->layer_k;
	}
}

/* HELPER FUNCTION - updates "backproped_gradients" */
void update_hidden_layers_pulls(hidden_layers *curr_layer) {
	/* Update local derivatives of the layer before this current layer */
	backpropagate_pull_update(curr_layer);

	/* Recursively goes to the laye before this current layer */
	if (curr_layer->previous_layer)	{
		update_hidden_layers_pulls(curr_layer->previous_layer);
	}

	return;
}

/* HELPER FUNCTION - updates the the coeficients of the given layer */
void update_coeficients(hidden_layers *curr_layer, datapoint *last_layer_outputs, double stepsize) {
	hidden_layer *curr_layer_aux = curr_layer->layer_k;
	datapoint *last_layer_outputs_aux = last_layer_outputs;

	while (curr_layer_aux) { /* Visit nth neuron */
		coeficients *coef_aux = curr_layer_aux->n_j->coefs;

		while (coef_aux) { /* Visit nth neuron's jth coeficient */
			double da1 = (last_layer_outputs_aux->x_i * curr_layer_aux->n_j->g);
			double a1 = coef_aux->y_i;
			da1 += -a1;
			//printf("{%lf}\n", stepsize * da1);
			coef_aux->y_i += stepsize * da1;

			coef_aux = coef_aux->next;
			if (!coef_aux->next) {
				//printf("{%lf}\n", stepsize * (1.0 * curr_layer_aux->n_j->g));
				coef_aux->y_i += stepsize * (1.0 * curr_layer_aux->n_j->g);
				break;
			}
			last_layer_outputs_aux = last_layer_outputs_aux->next;
		}
		curr_layer_aux = curr_layer_aux->next;
		last_layer_outputs_aux = last_layer_outputs;
	}
}

void update_all_neurons_coeficients(hidden_layers *last_layer, datapoint* original_datapoint, double stepsize) {
	if (!last_layer->previous_layer) {
		update_coeficients(last_layer, original_datapoint, stepsize);
		return;
	}

	/* Update the the coeficients of the current layer */
	update_coeficients(last_layer, layer_neurons_outputs(last_layer->previous_layer->layer_k), stepsize);

	if (!last_layer->previous_layer->previous_layer) {
		update_all_neurons_coeficients(last_layer->previous_layer, original_datapoint, stepsize);
	} else {
		update_all_neurons_coeficients(last_layer->previous_layer, layer_neurons_outputs(last_layer->previous_layer->previous_layer->layer_k), stepsize);
	}
}

void network_backprop(output_neuron *out_neuron, double pull, double stepsize, datapoint* data) {
	/* Update the output neuron coeficients - 'backproped_derivatives' holds the pulls to be sent to the last hidden layer (before the output neuron) */
	datapoint *backproped_derivatives = update_output_neuron_coeficients(out_neuron, layer_neurons_outputs(out_neuron->last_layer->layer_k), pull, stepsize);

	/* Updates the local gradients in the neurons in the hidden layers */
	update_neuron_pull(out_neuron->last_layer, backproped_derivatives);

	/* Updates all the coeficients of the neurons in the hidden layers */
	update_all_neurons_coeficients(out_neuron->last_layer, data, stepsize);
}
