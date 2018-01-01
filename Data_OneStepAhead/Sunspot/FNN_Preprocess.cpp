#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <ctime>

#include<stdio.h>
#include<stdlib.h>
#include<ctype.h>

using namespace::std;


int main(void)
{ 
const int alldatasize = 1000;
const int trainsize = 0.60 * alldatasize; //1200
const int validationsize = 0.0 * alldatasize; //400
const int testsize = 0.40 * alldatasize;//400

const int dimen = 7;
const int timelag = 2;

ifstream in;
in.open("sunspot.dat");


ofstream out;
out.open("train7.txt");

ofstream outvalidation;
outvalidation.open("validation.txt");

ofstream outtest;
outtest.open("test7.txt");

ofstream range;
range.open("scaled_dataset.txt"); //holds the scaled values

double Sunspot[10000]; //contains all scaled data set
double max = 0;


//input all values from sun_smoothed into an array
for (int i =0; i < alldatasize; i++)
   in>>Sunspot[i]; 
//------------------------make value between 0 and 1
       
					//find max
                    for (int i =0; i < alldatasize; i++) {
					 if(Sunspot[i]>max)
					   max = Sunspot[i];
					}
					
					//fill array with value between 0 and 1
                    for (int i =0; i < alldatasize; i++)
					{
					    Sunspot[i] =  (Sunspot[i])/(max+1);
                        range<<Sunspot[i]<<endl;                                
                     }

       
       int trainset = 0, testset = 0 , dpafter = 0, validationset=0; 
       
	int position = 0;
       //Fill up the input file with the values using D=5 and T =3
       for (int i =  dimen; i < trainsize; i+=timelag) 
       { 
      
	    for (int k = i - dimen ; k < i; k++) 
        {
            out<<Sunspot[k]<<" "; 
	    position = k;        
        } 
	    out<<Sunspot[position+1]<<endl; 
        trainset++;      
       }           
 		//fill the validation file with values using D=5 and T=3
        for(int i = trainsize + dimen ; i < (trainsize + validationsize); i+=timelag)
        {
               for (int k = i - dimen ; k < i; k++) 
               {
                   outvalidation<< Sunspot[k] <<" ";
                   position = k;   
	      }
               outvalidation<<Sunspot[position+1]<<endl;
               validationset++;	    
        }  		
	
	//fill the test file with values using D=5 and T=3
        for(int i = (alldatasize - testsize) + dimen ; i < alldatasize; i+=timelag)
        {
               for (int k = i - dimen ; k < i; k++) 
               {
                   outtest<< Sunspot[k] <<" ";
                   position = k;                   
               }
               outtest<<Sunspot[position+1]<<endl;
               testset++;	    
        }      
        

        //**************************SUMMARY***************************
        ofstream summary;
        summary.open("processedDataSummary.txt");
        
        summary<<"--------------------------------------------------------------"<<endl;
        summary<<"\t\t\tDATA PRE-PROCESSING SUMMARY"<<endl;
        summary<<"--------------------------------------------------------------"<<endl;
        summary<<"\nSize of training set:  "<<trainset<<endl; 
	summary<<"\nSize of validation set:  "<<validationset<<endl;        
        summary<<"Size of test set:  "<<testset<<endl;
        summary<<"Input Neuron:  "<<dimen<<endl;
        summary<<"--------------------------------------------------------------"<<endl;
        
        summary.close();
        //************************END SUMMARY**************************

        in.close();
        out.close();
        outtest.close();
        //out2.close();
        outvalidation.close();
    
         //system("PAUSE");
         //return 0;
       };
