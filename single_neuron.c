#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STEPSIZE 0.01

/* Merged representation of neuron output (value) and weight of connection (gradient) */
typedef struct units {
	double value;
	double gradient;
} unit;

/* Helper function to create/allocate neuron */
unit* newUnit(double v, double g) {
	unit* newUnit = (unit*)malloc(sizeof(unit*));

	newUnit->value = v;
	newUnit->gradient = g;

	return newUnit;
}

/* Multiplication gate */
unit* multiplyGate(unit* u0, unit* u1) {
	return newUnit((u0->value * u1->value), 0.0);
}

/* Sum gate */
unit* addGate(unit* u0, unit* u1) {
	return newUnit((u0->value + u1->value), 0.0);
}

/* Helper function to calculate sig(x) */
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

/* Sigmoid gate */
unit* sigmoidGate(unit* u0) {
	return newUnit(sigmoid(u0->value), 0.0);
}


void sigmoidGateBackProp(unit* u, unit* top) {
	double s = sigmoid(u->value);
	u->gradient = (s * (1 - s)) * top->gradient;
}

void addGateBackProp(unit* u0, unit* u1, unit* top) {
	u0->gradient += 1 * top->gradient;	
	u1->gradient += 1 * top->gradient;
}

void multiplyGateBackProp(unit* u0, unit* u1, unit* top) {
	u0->gradient += u1->value * top->gradient;	
	u1->gradient += u0->value * top->gradient;
}

int main () {

	/* Initialize neurons */
	unit *a = newUnit(1.0, 0.0);
	unit *b = newUnit(2.0, 0.0);
	unit *c = newUnit(-3.0, 0.0);
	unit *x = newUnit(-1.0, 0.0);
	unit *y = newUnit(3.0, 0.0);

	/* Forward - calculates f(a, b, c, x, y) = sig(a*x + b*y + c) */
	unit *ax = multiplyGate(a, x);
	unit *by = multiplyGate(b, y);
	unit *axpby = addGate(ax, by);
	unit *axpbypc = addGate(axpby, c);
	unit *s = sigmoidGate(axpbypc);

	printf("circuit output: %lf\n", s->value);

	/* Back Propagate */
	s->gradient = 1.0;
	sigmoidGateBackProp(axpbypc, s);
	addGateBackProp(axpby, c, axpbypc);
	addGateBackProp(ax, by, axpby);
	multiplyGateBackProp(a, x, ax);
	multiplyGateBackProp(b, y, by);

	/* Update weights */
	a->value += STEPSIZE * a->gradient;
	b->value += STEPSIZE * b->gradient;
	c->value += STEPSIZE * c->gradient;
	x->value += STEPSIZE * x->gradient;
	y->value += STEPSIZE * y->gradient;


	/* Forward - calculates f(a, b, c, x, y) = sig(a*x + b*y + c) */
	ax = multiplyGate(a, x);
	by = multiplyGate(b, y);
	axpby = addGate(ax, by);
	axpbypc = addGate(axpby, c);
	s = sigmoidGate(axpbypc);

	printf("circuit output after one backprop: %lf\n", s->value);

	return 0;
}