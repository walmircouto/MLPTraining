/* C3SL Labs - UFPR */
/* LADES Icibe - UFRA */
/* PPGInf - Graduate Program in Informatics - UFPR */
/* Classifying Unstructured Models into Metamodels Using Multi Layer Perceptrons */
/* Developed by Emerson Morais and Walmir Couto */
/* It was last updated on July 15 2019 */

// Libraries calling and compilation directives
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Constants define
#define NCAMOCULTA 5 // hidden layers amount
#define OCULTA 5 // hidden layer neurons amount
#define ENTRADA 72  // input layer neurons amount
#define SAIDA 1  // output layer neurons amount
#define EXEMPLOS 4000 // training entry examples amount

//Global Variables Statement
    float
          W[30][30][30],
          BIAS[30][30], 
          X[EXEMPLOS][72], //input vector
          D[EXEMPLOS][1], //output vector
          ERRO[EXEMPLOS][30],
          NI[30][30],
          PHI[30][30],
          ERRODES, //desirable or acceptable error value
          ETA,  //learning rate
          ERROINST, // Instant Error
          ERROMG=0, //Global Average Error
          PHILINHA[30][30], //philinha vector for correction
          DELTA[30][30]; //delta vector for weight and bias correction
    int   K,
          I,
          J,
          A,
          B,
          VAR,
          EPOCAS, //training epoch amount 
          FUNCAO;
    
//Definição dos módulos utilizados no programa
void ProcessoIterativo();
void EntradaBiasePesosIniciais();
void SaidaBiasePesosIniciais();
void EntraEpocas();
void PreencheVetorEntrada();
void PreencheVetorSaida();
void PreencheOutrasConfiguracoes();


//Programa Principal
int main(int argc, const char* argv[])
{
    EntradaBiasePesosIniciais();
    SaidaBiasePesosIniciais();
    EntraEpocas();
    PreencheVetorEntrada();
    PreencheVetorSaida();
    PreencheOutrasConfiguracoes();
    ProcessoIterativo();
    printf ("........................\n");
    getchar();
    return 0;
}
//Término do Programa Principal

/* Função de Processo Iterativo*/
void ProcessoIterativo()
{
    printf("Iniciando o processo iterativo...\n");
    for(A=0; A<EPOCAS; A++)
    {
       for(B=0; B<EXEMPLOS; B++)
       {

                //Calcula ni e phi da camada de entrada
                for(I=0; I<OCULTA; I++)
                {
                         NI[0][I]=0; // k de camada e i de neurônio
                         for(J=0; J<ENTRADA; J++)
                         {
                                  NI[0][I] = NI[0][I] + W[0][J][I]*X[B][J];
                         }
                         NI[0][I] = NI[0][I] + BIAS[0][I];
                         //printf("NI[1][%d] = %f\n", I+1, NI[0][I]);
                         switch(FUNCAO)
                         {
                                       case 1:
                                            if(NI[0][I]>0) PHI[0][I]=1;
                                            else PHI[0][I]=0;
                                            break;
                                       case 2:
                                            PHI[0][I]=1/(1 + exp(-NI[0][I]));
                                            break;
                         }//fim do switch case
                         //printf("PHI[1][%d] = %f\n", I+1, PHI[0][I]);
                  }//fim do for i

                  //Calcula ni e phi das camadas restantes
                  for(K=1; K<NCAMOCULTA+1; K++)
                  {
                      if (K==NCAMOCULTA) VAR = SAIDA; //última camada - ni da camada de saída
                      else VAR = OCULTA; //camadas intermediárias - ni das camadas intermediárias
                      for(I=0; I<VAR; I++)
                      {
                               NI[K][I]=0;
                               for(J=0; J<OCULTA; J++)
                               {
                                       NI[K][I] = NI[K][I] + W[K][J][I]*PHI[K-1][J];
                               }
                               NI[K][I] = NI[K][I] + BIAS[K][I];
                               //printf("NI[%d][%d] = %f\n", K+1, I+1, NI[K][I]);
                               switch(FUNCAO)
                               {
                               case 1:
                                    if(NI[K][I]>0) PHI[K][I]=1;
                                    else PHI[K][I]=0;
                                    break;
                               case 2:
                                    PHI[K][I]=1/(1 + exp(-NI[K][I]));
                                    break;
                               }//fim do case
                               if (K==NCAMOCULTA) printf("PHI[%d][%d] = %f\n", K+1, I+1, PHI[K][I]);
                      } //fim do for i
                  } // fim do K
                  /* Cálculo do Erros */
                  for(K=1; K<NCAMOCULTA+1; K++)
                  {
                        for(I=0; I<SAIDA; I++)
                        {
                                 ERRO[K][I]= D[B][I] - PHI[K][I];
                                 // printf("DESEJADO[%d][%d] = %f\n", B+1, I+1, D[B][I]);
                                 // printf("ERRO[%d][%d] = %f\n", K+1, I+1, ERRO[K][I]);
                        }
                        ERROINST=0;
                        for(I=0; I<SAIDA; I++)
                        {
                                 ERROINST = ERROINST + ERRO[K][I]*ERRO[K][I]/2;
                        }
                        ERROMG = (ERROMG*(A*EXEMPLOS + B) + ERROINST)/(A*EXEMPLOS + (B+1));
                        //printf("ERROMG = %f\n", ERROMG);
                        if (ERROMG < ERRODES)
                        break;
                        /* Fim Cálculo do Erros */
                  } // fim do K

       /* retropropagaÁ„o do erro */

             /* cálculo de philinha e delta */

             /* cálculo de philinha e delta da última camada*/
             for(I=0; I<SAIDA; I++)
             {
                  PHILINHA[NCAMOCULTA][I]= exp(-NI[NCAMOCULTA][I])/((1 + exp(-NI[NCAMOCULTA][I]))*(1 + exp(-NI[NCAMOCULTA][I])));
                  DELTA[NCAMOCULTA][I]= -ERRO[NCAMOCULTA][I]*PHILINHA[NCAMOCULTA][I];
                  //printf("PHILINHA[%d][%d] = %f\n", NCAMOCULTA+1, I+1, PHILINHA[NCAMOCULTA][I]);
                  //printf("DELTA[%d][%d] = %f\n", NCAMOCULTA+1, I+1, DELTA[NCAMOCULTA][I]);
             }
             /* cálculo de philinha e delta das camadas intermediárias*/

             for(K=1; K<NCAMOCULTA; K++)
             {
                for(I=0; I<OCULTA; I++)
                  {
                       PHILINHA[K][I]= exp(-NI[K][I])/((1 + exp(-NI[K][I]))*(1 + exp(-NI[K][I])));
                       //printf("PHILINHA[%d][%d] = %f\n", K+1, I+1, PHILINHA[K][I]);
                       DELTA[K][I]= 0;
                       for(J=0; J<OCULTA; J++)
                       {
                            DELTA[K][I]= DELTA[K][I] + PHILINHA[K][I]*DELTA[K+1][J]*W[K+1][I][J];
                       }
                       //printf("DELTA[%d][%d] = %f\n", K+1, I+1, DELTA[K][I]);
                   }
             }

             /* cálculo de philinha e delta da primeira camada*/
             for(I=0; I<OCULTA; I++)
             {
                       PHILINHA[0][I]= exp(-NI[0][I])/((1 + exp(-NI[0][I]))*(1 + exp(-NI[0][I])));
                       //printf("PHILINHA[1][%d] = %f\n", I+1, PHILINHA[0][I]);
                       DELTA[0][I]= 0;
                       for(J=0; J<OCULTA; J++)
                       {
                             DELTA[0][I]= DELTA[0][I] + PHILINHA[0][I]*DELTA[1][J]*W[1][I][J];
                       }
                       //printf("DELTA[1][%d] = %f\n", I+1, DELTA[0][I]);
             }

/* Ajuste dos pesos e dos bias  */

              //ajuste das outras camadas
              for(K=1; K<NCAMOCULTA+1; K++)
              {
                      if (K==NCAMOCULTA) VAR = SAIDA; //última camada - ni da camada de saída
                      else VAR = OCULTA; //camadas intermediárias - ni das camadas intermediárias
                      for(I=0; I<VAR; I++)
                      {
                                for(J=0; J<OCULTA; J++)
                                {
                                         W[K][J][I]= W[K][J][I] - ETA*DELTA[K][I]*PHI[K-1][J];
                                         //printf("W[%d][%d][%d] = %f\n", K+1, J+1, I+1, W[K][J][I] );
                                }
                                BIAS[K][I]= BIAS[K][I] - ETA*DELTA[K][I]*PHI[K-1][I];
                                //printf("BIAS[%d][%d] = %f\n", K+1, I+1, BIAS[K][I] );
                      }
              }
              //ajuste da primeira camada
              for(I=0; I<OCULTA; I++)
              {
                  for(J=0; J<ENTRADA; J++)
                  {
                      W[0][J][I]= W[0][J][I] - ETA*DELTA[0][I]*X[B][J];
                      //printf("W[1][%d][%d] = %f\n", J+1, I+1, W[0][J][I] );
                  }
                  BIAS[0][I]= BIAS[0][I] - ETA*DELTA[0][I]*X[B][I];
                  //printf("BIAS[1][%d] = %f\n", I+1,  BIAS[0][I] );
              }


//??????

         if(ERROMG < ERRODES)
         {
              printf("Finalizado pelo erro em %d epocas de treinamento!\n", X);
              break;
         }
     } // fim do B
  } // fim do A
}
/* Término Função de Processo Iterativo*/


/* Função de Entrada de Bias e Pesos Iniciais*/
void EntradaBiasePesosIniciais()
{
/*
     printf("Bias e pesos iniciais...\n");

     //Entrada dos Pesos Iniciais e Bias
     for(K=0; K<NCAMOCULTA+1; K++)
     {
         if (K==0) //primeira camada
         {
            printf("ENTRADA e 1 a. CAMADA OCULTA...\n");
            for(I=0; I<ENTRADA; I++)
            {
                for(J=0; J<OCULTA; J++)
                {
                       //printf("Informe o peso entre o neuronio %d da camada de entrada e o neuronio %d da 1a. camada oculta :", I+1, J+1);
                       printf("Informe W[%d][%d][%d]:", K+1, I+1, J+1);
                       scanf("%f", &W[K][I][J]);
                }
             }
             for(I=0; I<OCULTA; I++)
             {
                 //printf("Informe o bias do neuronio %d da 1a. camada oculta :", I+1);
                 printf("Informe BIAS[%d][%d]:", K+1, I+1);
                 scanf("%f", &BIAS[K][I]);
             }
         }
         else if (K==NCAMOCULTA)   //ultima camada
              {
                 printf("%d a. CAMADA OCULTA e SAIDA\n", K);
                 for(I=0; I<OCULTA; I++)
                 {
                       for(J=0; J<SAIDA; J++)
                       {
                                //printf("Informe o peso entre o neuronio %d da %d a. camada oculta e o neuronio %d da camada de saida :", I+1, K, J+1);
                                printf("Informe W[%d][%d][%d]:", K+1, I+1, J+1);
                                scanf("%f", &W[K][I][J]);
                       }

                  }
                  for(I=0; I<SAIDA; I++)
                  {
                           //printf("Informe o bias do neuronio %d da camada de saida :", I+1);
                           printf("Informe BIAS[%d][%d]:", K+1, I+1);
                           scanf("%f", &BIAS[K][I]);
                   }
              }
              else //camadas ocultas
              {
                  printf("%d a. CAMADA OCULTA e %d a. CAMADA OCULTA\n", K, K+1);
                  for(I=0; I<OCULTA; I++)
                  {
                        for(J=0; J<OCULTA; J++)
                        {
                              //printf("Informe o peso entre o neuronio %d da %d a. camada oculta e o neuronio %d da %d a. camada oculta :", I+1, K, J+1, K+1);
                              printf("Informe W[%d][%d][%d]:", K+1, I+1, J+1);
                              scanf("%f", &W[K][I][J]);
                        }
                  }
                  for(I=0; I<OCULTA; I++)
                  {
                           //printf("Informe o bias do neuronio %d da %d camada oculta :", I+1, K+1);
                           printf("Informe BIAS[%d][%d]:", K+1, I+1);
                           scanf("%f", &BIAS[K][I]);
                  }
              }
}
   */
   W[0][0][0] = 0.7;
   W[0][0][1] = -0.8;
   W[0][1][0] = 0.5;
   W[0][1][1] = -0.6;
   W[1][0][0] = -0.3;
   W[1][0][1] = 0.4;
   W[1][1][0] = 0.5;
   W[1][1][1] = -0.7;
   W[2][0][0] = 0.8;
   W[2][1][0] = -0.5;
   BIAS[0][0] = 0.5;
   BIAS[0][1] = -0.7;
   BIAS[1][0] = 0.6;
   BIAS[1][1] = -0.8;
   BIAS[2][0] = 0.4;
   
   printf("PESOS E BIAS PREENCHIDOS AUTOMATICAMENTE\n");
}
/* Término Função de Entrada de Bias e Pesos Iniciais*/



/* Função de Saída de Bias e Pesos Iniciais*/
void SaidaBiasePesosIniciais()
{
/*
     //Apresentação dos Pesos Iniciais e Bias
     printf("Pesos iniciais e Bias :\n");
     for(K=0; K<NCAMOCULTA+1; K++)
     {
         if (K==0)
         {
            printf("ENTRADA e 1 a. CAMADA OCULTA...\n");
            for(I=0; I<ENTRADA; I++)
                for(J=0; J<OCULTA; J++)
                     //printf("Peso entre o neuronio %d da camada de entrada e o neuronio %d 1a. camada oculta = %f\n", I+1, J+1, W[K][I][J]);
                     printf("W[%d][%d][%d] = %f\n", K+1, I+1, J+1, W[K][I][J] );
            for(I=0; I<OCULTA; I++)
                 //printf("Bias do neuronio %d da 1a. camada oculta = %f\n", I+1, K+1, BIAS[K][J]);
                 printf("BIAS[%d][%d] = %f\n", K+1, I+1, BIAS[K][I]);
         }
         else if (K==NCAMOCULTA)
              {
                 printf("%d a. CAMADA OCULTA e SAIDA\n", K);
                 for(I=0; I<OCULTA; I++)
                 {
                       for(J=0; J<SAIDA; J++)
                          //printf("Peso entre o neuronio %d da %d a. camada oculta e o neuronio %d da camada de saida = %f\n", I+1, K, J+1, W[K][I][J]);
                           printf("W[%d][%d][%d] = %f\n", K+1, I+1, J+1, W[K][I][J] );
                 }
                 for(I=0; I<SAIDA; I++)
                        //printf("Bias do neuronio %d da camada de saida = %f\n", I+1, BIAS[K][J]);
                        printf("BIAS[%d][%d] = %f\n", K+1, I+1, BIAS[K][I]);
              }
              else
              {
                  printf("%d a. CAMADA OCULTA e %d a. CAMADA OCULTA\n", K, K+1);
                  for(I=0; I<OCULTA; I++)
                  {
                      printf("Bias do neuronio %d da %d camada oculta = %f\n", I+1, K, BIAS[K][I]);
                      for(J=0; J<OCULTA; J++)
                           //printf("Peso entre o neuronio %d da %d a. camada oculta e o neuronio %d da %d a. camada oculta = %f\n", I+1, K, J+1, K+1, W[K][I][J]);
                            printf("W[%d][%d][%d] = %f\n", K+1, I+1, J+1, W[K][I][J] );
                  }
                  for(I=0; I<OCULTA; I++)
                      //printf("Bias do neuronio %d da %d camada oculta = %f\n", I+1, K+1, BIAS[K][J]);
                      printf("BIAS[%d][%d] = %f\n", K+1, I+1, BIAS[K][I]);
               }
}
*/
}
/* Término Função de Saída de Bias e Pesos Iniciais*/

void EntraEpocas()
{
/*
    printf("Entre com o numero de epocas de treinamento:\n");
    scanf("%d", &EPOCAS);
*/
  EPOCAS = 100;
  printf("NUMERO DE EPOCAS PREENCHIDOS AUTOMATICAMENTE\n");

}

void PreencheVetorEntrada()
{
/*
//Preenchimento dos vetores de entrada
    printf("Entre com os vetores de exemplos de treinamento de entrada:\n");
    for(I=0; I<EXEMPLOS; I++)
    {
        printf("Informe o numero binario que representa o %d o. elemento de treinamento\n", I+1);
        for(J=0; J<ENTRADA; J++)
        {
             printf("Informe o %d o. numero:", J+1);
             scanf("%f", &X[I][J]);
        }
    }

*/
  X[0][0] = 1;
  X[0][1] = 1;
  printf("VETORES DE ENTRADA PREENCHIDOS AUTOMATICAMENTE\n");

}

void PreencheVetorSaida()
{
/*
//Preenchimento dos vetores de saída
    printf("Entre com os vetores de exemplos de treinamento de saida:\n");

    for(I=0; I < EXEMPLOS; I++)
    {
         printf("Preencha os valores que representam as saidas esperadas do %d o.elemento\n", I+1);
         for (J=0; J < SAIDA; J++)
         {
             printf("Informe o %d o. numero:", J+1);
             scanf("%f", &D[I][J]);
         }
     }

*/
     D[0][0] = 0;
     printf("VETORES DE SAIDA PREENCHIDOS AUTOMATICAMENTE\n");
}

void PreencheOutrasConfiguracoes()
{
//Outras configurações

   /* printf("Entre com o valor da taxa de aprendizagem:\n");
    scanf("%f", &ETA);

    printf("Entre com o erro máximo desejado:\n");
    scanf("%f", &ERRODES);

    printf("Entre com função desejada[(1) degrau, (2)sigmoide]:\n");
    scanf("%d", &FUNCAO);
     */
     ETA = 0.8;
     ERRODES = -1;
     FUNCAO = 2;
}