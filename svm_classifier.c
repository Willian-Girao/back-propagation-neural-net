#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

/* Sigmoid gate */
unit* svmLinearFunctin(unit* u0) {
	return newUnit(u0->value, 0.0);
}

void addGateBackProp(unit* u0, unit* u1, unit* top) {
	u0->gradient += 1 * top->gradient;	
	u1->gradient += 1 * top->gradient;
}

void multiplyGateBackProp(unit* u0, unit* u1, unit* top) {
	u0->gradient += u1->value * top->gradient;	
	u1->gradient += u0->value * top->gradient;
}

/* Datapoints */
double dataPoints[6][2] = {{1.2, 0.7}, {-0.3, -0.5}, {3.0, 0.1}, {-0.1, -1.0}, {-1.0, 1.1}, {2.1, -3}};
int labels[6] = {1, -1, 1, -1, -1, 1};

unit *ax;
unit *by;
unit *axpby;
unit *axpbypc;
unit *s;

/* Initialize neurons */
unit *a;
unit *b;
unit *c;
unit *x;
unit *y;

/* Forwards the inputs */
unit* svm_forward() {
	ax = multiplyGate(a, x);
	by = multiplyGate(b, y);
	axpby = addGate(ax, by);
	axpbypc = addGate(axpby, c);
	return svmLinearFunctin(axpbypc);
}

void svm_backpropagate(int label) {
	a->gradient = 0.0;
	b->gradient = 0.0;
	c->gradient = 0.0;

	s->gradient = 0.0;

	if (label == 1 && s->value < 1) {
		s->gradient = 1.0;
	} 
	if (label == -1 && s->value > -1) {
		s->gradient = -1.0;
	}

	addGateBackProp(axpby, c, s);
	addGateBackProp(ax, by, axpby);
	multiplyGateBackProp(a, x, ax);
	multiplyGateBackProp(b, y, by);

	/* Normalization */
	a->gradient += -a->value;
	b->gradient += -b->value;
}

void svm_coeficient_update() {
	/* Actual updating */
	a->value += STEPSIZE * a->gradient;
	b->value += STEPSIZE * b->gradient;
	c->value += STEPSIZE * c->gradient;
}

double evalPerformance() {
	unit *sol;
	double correct = 0;
	double predicted;

	for (int i = 0; i < 6; ++i) {
		x = newUnit(dataPoints[i][0], 0.0);
		y = newUnit(dataPoints[i][1], 0.0);
		int trueLabel = labels[i];

		sol = svm_forward();
		predicted = (sol->value > 0) ? 1 : -1;

		if (predicted == trueLabel) {
			correct++;
		}
	}

	return correct / 6;
}

int main () {
	time_t t;
	srand((unsigned) time(&t));

	/* Initialize linear coeficients */
	a = newUnit(1.0, 0.0);
	b = newUnit(-2.0, 0.0);
	c = newUnit(-1.0, 0.0);
	
	for (int i = 0; i < 400; ++i) {
		int dataPoint = rand() % 6;

		x = newUnit(dataPoints[dataPoint][0], 0.0);
		y = newUnit(dataPoints[dataPoint][1], 0.0);
		int label = labels[dataPoint];

		/* Forward - calculates f(x, y) = a*x + b*y + c */
		s = svm_forward();
		svm_backpropagate(label);
		svm_coeficient_update();

		if (i % 25 == 0) {
			printf("Training accuracy at iteration %d: %lf\n", i, evalPerformance());
		}
	}

	printf("Final accuracy achived: %lf\n", evalPerformance());

	return 0;
}