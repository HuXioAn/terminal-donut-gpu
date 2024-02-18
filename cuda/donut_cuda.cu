
#include <stdio.h>
#include <math.h>
//#include <unistd.h>
#include <chrono>
#include <thread>
#include <cstring>

//define the donut 
const float R1 = 1;
const float R2 = 2;

//define the render grid
const float thetaStep = 0.07;
const float phiStep = 0.02;

//define the camera
const float K1 = 30; //focus(origin) to the screen
const int resH = 22; //render resolution
const int resW = 80;

//define the pos
const float K2 = 5; //focus(origin) to the donut center

//define the light, a opposite vecter of the light beem, simulating an infinite plane light source
//it should be a unit vector
const float lX = 0;
const float lY = 0.707;
const float lZ = -0.707;



__host__ void renderFrame(float A, float B, char* outputP){ //A: rotate about x-axis, B: rotate about z-axis

    static float zBuf[resW][resH]; 

    memset(outputP, ' ', resW * resH * sizeof(char));
    memset(zBuf, 0, resW * resH * sizeof(float));

    float sinA = sin(A), sinB = sin(B), cosA = cos(A), cosB = cos(B);

    //iterate the donut for one frame
    for(float theta = 0; theta < 3.14 * 2; theta += thetaStep){ //the smaller circle

        float sinTheta = sin(theta), cosTheta = cos(theta);

        for(float phi = 0; phi < 3.14 * 2; phi += phiStep){ //the larger circle
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

            //illumination, Lvec * Normalvec
            float lum = lX * (cosTheta * cosPhi * cosB + 
                        sinPhi * cosTheta * sinA * sinB - 
                        sinTheta * cosA * sinB) +
                        lY * (cosTheta * cosPhi * sinB +
                        cosB * sinTheta * cosA -
                        cosB * sinPhi * cosTheta * sinA) +
                        lZ * (sinA * sinTheta + sinPhi * cosTheta * cosA);

            if( ooz > zBuf[xP][yP] /* current point is closer to the viewer */&&
                lum > 0 /* the point is visible */){
                zBuf[xP][yP] = ooz;

                int lumIndex = lum * 11.3; //map the illuminance to the index
                outputP[xP + resW * yP] = ".,-~:;=!*#$@"[lumIndex];
            }


        }

    } 



}


__global__ void renderPointCuda(const float A, const float B, float* zBuf, char* output){

    //get the theta and phi from index
    float theta = blockIdx.x * thetaStep;
    float phi = threadIdx.x * phiStep;

    float sinTheta = sinf(theta);
    float cosTheta = cosf(theta);

    float sinPhi = sinf(phi);
    float cosPhi = cosf(phi);

    //common value
    float sinA = sinf(A);
    float cosA = cosf(A);
    float sinB = sinf(B);
    float cosB = cosf(B);


    //without rotations, z = 0
    float x = R2 + R1 * cosTheta;
    float y = R1 * sinTheta;

    //rotate matrix
    float xRotatedo = x * (cosB * cosPhi + sinA * sinB * sinPhi) - y * cosA * sinB;
    float yRotatedo = x * (sinB * cosPhi - sinA * cosB * sinPhi) + y * cosA * cosB;

    float zRotated = K2 + cosA * x * sinPhi + y * sinA;

    float ooz = 1 / zRotated;

    int xP = (int) (resW / 2 + K1 * ooz * xRotatedo);
    int yP = (int) (resH / 2 - K1 * 0.5 * ooz * yRotatedo);

    if(xP >= resW || xP < 0 || yP >= resH || yP < 0){
        //out of the screen, end thread for the point
        return;
    }

    //illumination, Lvec * Normalvec
    float lum = lX * (cosTheta * cosPhi * cosB + 
                sinPhi * cosTheta * sinA * sinB - 
                sinTheta * cosA * sinB) +
                lY * (cosTheta * cosPhi * sinB +
                cosB * sinTheta * cosA -
                cosB * sinPhi * cosTheta * sinA) +
                lZ * (sinA * sinTheta + sinPhi * cosTheta * cosA);

    if( ooz > zBuf[xP + yP * resW] /* current point is closer to the viewer */&&
        lum > 0 /* the point is visible */){
        zBuf[xP + yP * resW] = ooz;

        int lumIndex = lum * 11.3; //map the illuminance to the index
        output[xP + resW * yP] = ".,-~:;=!*#$@"[lumIndex];
    }



}


__host__ int main(){

    const size_t outputSize = resH * resW * sizeof(char);
    char outputF[outputSize];

    
    char* output;
    if(cudaSuccess != cudaMalloc(&output, outputSize)){
        //fail
        printf("[!]Unable to aloocate the output frame.");
        exit(-1);
    }

    float* zBuf;
    if(cudaSuccess != cudaMalloc(&zBuf, resH * resW * sizeof(float))){
        //fail
        printf("[!]Unable to aloocate the zBuf frame.");
        exit(-1);
    }


    //by default(A = 0), the donut is horizontally positioned, the hole will be invisible at the beginning
    float A = 3.14/2, B = 0; 
    
    while(true){
        //clear the buf on the gpu
        cudaMemset(output, ' ', resW * resH * sizeof(char));
        cudaMemset(zBuf, 0, resW * resH * sizeof(float));
        
        //thread for every point on the donut, grid for small circle, block for big circle
        renderPointCuda<<<6.28/thetaStep,6.28/phiStep>>>(A, B, zBuf, output);

        cudaDeviceSynchronize();    
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("[!]CUDA error: %s\n", cudaGetErrorString(error));
            exit(1);
        }
        
        //get the frame from gpu
        cudaMemcpy(outputF, output, outputSize, cudaMemcpyDeviceToHost);


        //output
        printf("\x1b[H");
        for (int k = 0; k < resH * resW + 1; k++) {
            putchar(k % resW ? outputF[k] : 10); //newline or char
        }

        //control the rotation speed
         A += 0.04;
         B += 0.02;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    

}






