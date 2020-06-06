#ifndef RNA_v1_h
#define RNA_v1_h
#define LIBRARY_VERSION	1.0.0

class RNA
{
  private:
    // L'ensemble des poids du reseau.
    float* w;
    // Poids de la couche cachee a la couche de sortie.
    float* x;
    // Biais.
    float* b;
    // Couche cachee.
    float* h;
    // couche de sortie.
    float* o;
    // Nombre des biais.
    int nb;
    // Nombre des poids.
    int nw;
    // Nombre des entrees.
    int nips;
    // Nombre des neurones de la couche cachee.
    int nhid;
    // Nombre de sorties.
    int nops;

    // Fonctions utilitaires
    float frand();
    void  wbrand();
    float err(const float , const float );
    float pderr(const float , const float );
    float toterr(const float* const , const float* const , const int );
    float act(const float );
    float pdact(const float );
    void  bprop(const float* const , const float* const , float );
    void  fprop(const float* const );


  public:

	RNA(int , int , int ); 	// * constructeur.  Defini nnens, nnncc, nnsos
        					//   Nombre de neurones correspendant aux entrees
                            //   Nombre de neurones correspendant a la couche cachee
							//   Nombre de neurones correspendant aux sorties

	float* predict(const float*);
	float  train(const float* , const float* , float );
	void   printNetwork();
};
#endif