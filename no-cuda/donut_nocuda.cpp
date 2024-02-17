#include <iostream>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

//define the donut 
const float R1 = 1;
const float R2 = 2;

//define the render grid
const float thetaStep = 0.07;
const float phiStep = 0.02;

//define the camera
const float K1 = 30; //focus to the screen
const int resH = 22;
const int resW = 80;


//define the pos
const float K2 = 5; //focus to the donut center

//define the light

const float lX = 0;
const float lY = 1;
const float lZ = -1;



void renderFrame(float A, float B, char* outputP){

    static float zBuf[resW][resH]; 

    memset(outputP, ' ', resW * resH * sizeof(char));
    memset(zBuf, 0, resW * resH * sizeof(float));

    float sinA = sin(A), sinB = sin(B), cosA = cos(A), cosB = cos(B);

    //iterate the donut for one frame
    for(float theta = 0; theta < 3.14 * 2; theta += thetaStep){ //the smaller circle

        float sinTheta = sin(theta), cosTheta = cos(theta);

        for(float phi = 0; phi < 3.14 * 2; phi += phiStep){ //larger circle
            float sinPhi = sin(phi), cosPhi = cos(phi);
            
            //for every point with theta, phi

            //without rotations, z = 0
            float x = R2 + R1 * cosTheta;
            float y = R1 * sinTheta;

            //rotate matrix
            float xRotatedo = x * (cosB * cosPhi + sinA * sinB * sinPhi) - y * cosA * sinB;
            float yRotatedo = x * (sinB * cosPhi - sinA * cosB * sinPhi) + y * cosA * cosB;

            // float xRotated = cosPhi * x * cosB - sinB * (sinPhi * x * cosA - sinTheta * sinA);
            // float yRotated = cosPhi * x * sinB + cosB * (sinPhi * x * cosA - sinTheta * sinA);

            float zRotated = K2 + cosA * x * sinPhi + y * sinA;

            float ooz = 1 / zRotated;

            int xP = (int) (resW / 2 + K1 * ooz * xRotatedo);
            int yP = (int) (resH / 2 - K1 * 0.5 * ooz * yRotatedo);

            if(xP >= resW || xP < 0 || yP >= resH || yP < 0){
                //out of the screen, pass
                continue;
            }

            float lum = lX * (cosTheta * cosPhi * cosB + 
                        sinPhi * cosTheta * sinA * sinB - 
                        sinTheta * cosA * sinB) +
                        lY * (cosTheta * cosPhi * sinB +
                        cosB * sinTheta * cosA -
                        cosB * sinPhi * cosTheta * sinA) +
                        lZ * (sinA * sinTheta + sinPhi * cosTheta * cosA);

            if( ooz > zBuf[xP][yP] && lum > 0){
                zBuf[xP][yP] = ooz;

                int lumIndex = lum * 8;
                outputP[xP + resW * yP] = ".,-~:;=!*#$@"[lumIndex];
            }


        }

    } 



}


int main(){

    static char output[resH][resW];
    float A = 3.14/2, B = 0;

    while(true){
        renderFrame(A, B, (char*)output);

        //output
        printf("\x1b[H");
        for (int k = 0; k < resH * resW + 1; k++) {
            putchar(k % resW ? ((char*)output)[k] : 10); //newline or char
        }
         A += 0.04;
         B += 0.02;
        usleep(30000);
    }

    

}






