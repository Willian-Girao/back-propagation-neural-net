
/* Structure to hold a hole vector - a datapoint */
typedef struct input_data {
	double x_i;
	struct input_data *next;
} datapoint;

/* Initializes a datapoint struct */
datapoint* new_datapoint(double value) {
	datapoint *new_data = (datapoint*)malloc(sizeof(datapoint*));

	new_data->x_i = value;
	new_data->next = NULL;

	return new_data;
}

/* Builds the list representing the datapoint (vector) */
datapoint* build_datapoint(double x_i, datapoint *start) {
	if (!start)	{
		return new_datapoint(x_i);
	}

	datapoint *aux_d = start;
	while(aux_d->next) {
		aux_d = aux_d->next;
	}

	aux_d->next = new_datapoint(x_i);

	return start;
}

void print_datapoint(datapoint *x) {
	datapoint *aux = x;
	while(aux) {
		printf("%.2lf ", aux->x_i);
		aux = aux->next;
	}
	printf("\n");
}