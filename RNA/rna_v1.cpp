#if ARDUINO >= 100
  #include "Arduino.h"
#else
  #include "WProgram.h"
#endif

#include <rna_v1.h>

/* Constructeur */
RNA::RNA(int _nips, int _nhid, int _nops)
{
    nb = 2;
    nw = _nhid * (_nips + _nops);
    w = (float*) calloc(nw, sizeof(*w));
    x = w + _nhid * nips;
    b = (float*) calloc(nb, sizeof(*b));
    h = (float*) calloc(_nhid, sizeof(*h));
    o = (float*) calloc(_nops, sizeof(*o));
    nips = _nips;
    nhid = _nhid;
    nops = _nops;
    RNA::wbrand();
}

float* RNA::predict(const float* const in)
{
    RNA::fprop(in);
    return o;
}

// Trains a tinn with an input and target output with a learning rate. Returns target to output error.
float RNA::train(const float* const in, const float* const tg, float rate)
{
    RNA::fprop(in);
    RNA::bprop(in, tg, rate);
    return RNA::toterr(tg, o, nops);
}

// Prints Created Network parameters.
void RNA::printNetwork()
{
	Serial.print( "Nbr entrees : " );Serial.println(nips);
	Serial.print( "Nbr sorties : " );Serial.println(nops);
	Serial.print( "Nbr cachees : " );Serial.println(nhid);
	Serial.print( "Nbr biais   : " );Serial.println(nb);
	Serial.print( "Nbr poids   : " );Serial.println(nw);
	Serial.println( "Poids du reseau   : ");
	for(int i = 0; i < nw; i++)
	{
		Serial.print(w[i],5); 
		Serial.print(" , ");
	}
	Serial.println();

	Serial.println("Biais du reseau   : ");
	for(int i = 0; i < nb; i++) 
	{
		Serial.print(b[i],5); 
		Serial.print(" , ");
	}
}

// Computes error.
float RNA::err(const float a, const float b)
{
    return 0.5f * (a - b) * (a - b);
}

// Returns partial derivative of error function.
float RNA::pderr(const float a, const float b)
{
    return a - b;
}

// Computes total error of target to output.
float RNA::toterr(const float* const tg, const float* const o, const int size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; i++)
        sum += RNA::err(tg[i], o[i]);
    return sum;
}

// Activation function.
float RNA::act(const float a)
{
    return 1.0f / (1.0f + expf(-a));
}

// Returns partial derivative of activation function.
float RNA::pdact(const float a)
{
    return a * (1.0f - a);
}

// Returns floating point random from 0.0 - 1.0.
float RNA::frand()
{
    return rand() / (float) RAND_MAX;
}

// Performs back propagation.
void RNA::bprop(const float* const in, const float* const tg, float rate)
{
    for(int i = 0; i < nhid; i++)
    {
        float sum = 0.0f;
        // Calculate total error change with respect to output.
        for(int j = 0; j < nops; j++)
        {
            const float a = RNA::pderr(o[j], tg[j]);
            const float b = RNA::pdact(o[j]);
            sum += a * b * x[j * nhid + i];
            // Correct weights in hidden to output layer.
            x[j * nhid + i] -= rate * a * b * h[i];
        }
        // Correct weights in input to hidden layer.
        for(int j = 0; j < nips; j++)
            w[i * nips + j] -= rate * sum * RNA::pdact(h[i]) * in[j];
    }
}

// Performs forward propagation.
void RNA::fprop(const float* const in)
{
    // Calculate hidden layer neuron values.
    for(int i = 0; i < nhid; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < nips; j++)
            sum += in[j] * w[i * nips + j];
        h[i] = RNA::act(sum + b[0]);
    }
    // Calculate output layer neuron values.
    for(int i = 0; i < nops; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < nhid; j++)
            sum += h[j] * x[i * nhid + j];
        o[i] = RNA::act(sum + b[1]);
    }
}

// Randomizes tinn weights and biases.
void RNA::wbrand()
{
    for(int i = 0; i < nw; i++) w[i] = frand() - 0.5f;
    for(int i = 0; i < nb; i++) b[i] = frand() - 0.5f;
}
