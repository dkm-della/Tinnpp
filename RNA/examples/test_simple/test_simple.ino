#include <rna_v1.h>

RNA RNA1(2, 5, 1);

void setup()
{
	Serial.begin(115200);
	RNA1.printNetwork();
	float x[]={0.02, 0.7465};
	float y[] = { 1.085 };
	for(int i = 0; i < 50000; i++)
    {
    	float error = RNA1.train(x, y, 0.07);
    }
	Serial.println(*RNA1.predict(x));
}

void loop()
{
	// To Do
}


