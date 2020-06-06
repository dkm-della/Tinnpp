#include <rna_v1.h>

// Network instance
RNA RNA1(2, 5, 1);

void setup()
{
	Serial.begin(115200);
	RNA1.printNetwork();
	float x[]={0.02, 0.7465};
	float y[] = { 0.32415 };
  Serial.println("   Training the Network    ");
	for(int i = 0; i < 5000; i++)
  {
    	float error = RNA1.train(x, y, 0.07);
     //Serial.println(error, 5);
  }
  Serial.println("   Evaluating the Network    ");  
  float x1[]={0.018, 0.74};
	Serial.println(*RNA1.predict(x1), 5);
}

void loop()
{
	// To Do
}

