
#include <stdio.h>
#include <math.h>
//#include <unistd.h>
#include <chrono>
#include <thread>
#include <cstring>



//define the donut 
#define R1 (1)
#define R2 (2)

//define the render grid
#define thetaStep (0.06)
#define phiStep (0.03)

//define the camera
#define K1 (30) //focus(origin) to the screen
#define resH (22) //render resolution
#define resW (80)

//define the pos
#define K2 (5) //focus(origin) to the donut center

//define the light, a opposite vecter of the light beem, simulating an infinite plane light source
//it should be a unit vector
//pls notice that it can not render the shade
#define lX (0)
#define lY (0.707)
#define lZ (-0.707)

//define buf depth for oozBuf and lumBuf, bigger if using smaller grid
#define BUF_DEPTH (200)




__global__ void calcPointCuda(const float A, const float B, float* oozBuf, float* lumBuf, int* depthBuf){

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

    int index = yP * resW + xP;
    int indexBuf = BUF_DEPTH * index;
    //take one slot for the point
    int depth = atomicAdd(depthBuf+index, 1);
    //if(depth > BUF_DEPTH)printf("%d\n", depth);
    //store the ooz and lum
    oozBuf[indexBuf + depth] = ooz;
    lumBuf[indexBuf + depth] = lum;


}


__global__ void renderPixCuda(float* oozBuf, float* lumBuf, int* depthBuf, char* output){
    //results from previous kernel
    int x = threadIdx.x;
    int y = blockIdx.x;

    int index = y * resW + x;
    int indexBuf = BUF_DEPTH * index;

    float oozMax = 0;
    

    for(int i = 0; i < depthBuf[index]; i++){
        //iterate each corresponding point
        float ooz = oozBuf[indexBuf + i];
        float lum = lumBuf[indexBuf + i];

        if(ooz > oozMax){/* current point is closer to the viewer */
            oozMax = ooz;
            if(lum > 0 ){/* the point is visible */
                int lumIndex = lum * 11.3; //map the illuminance to the index
                output[index] = ".,-~:;=!*#$@"[lumIndex];
            }else{
                output[index] = ' ';
            }
        }
    }

}



__host__ int main(){

    const size_t outputSize = resH * resW * sizeof(char);
    char outputF[outputSize]; //host buf 

    
    char* output; //device buf
    if(cudaSuccess != cudaMalloc(&output, outputSize)){
        //fail
        printf("[!]Unable to aloocate the output frame.");
        exit(-1);
    }

    float* oozBuf;
    if(cudaSuccess != cudaMalloc(&oozBuf, BUF_DEPTH * resH * resW * sizeof(float))){
        //fail
        printf("[!]Unable to aloocate the oozBuf frame.");
        exit(-1);
    }

    float* lumBuf;
    if(cudaSuccess != cudaMalloc(&lumBuf, BUF_DEPTH * resH * resW * sizeof(float))){
        //fail
        printf("[!]Unable to aloocate the lumBuf frame.");
        exit(-1);
    }

    int* depthBuf;
    if(cudaSuccess != cudaMalloc(&depthBuf, resH * resW * sizeof(int))){
        //fail
        printf("[!]Unable to aloocate the depthBuf frame.");
        exit(-1);
    }


    //by default(A = 0), the donut is horizontally positioned, the hole will be invisible at the beginning
    float A = 3.14/2, B = 0; 
    
    while(true){
        //clear the buf on the gpu
        cudaMemset(output, ' ', resW * resH * sizeof(char));
        cudaMemset(oozBuf, 0, BUF_DEPTH * resH * resW * sizeof(float));
        cudaMemset(lumBuf, 0, BUF_DEPTH * resH * resW * sizeof(float));
        cudaMemset(depthBuf, 0, resH * resW * sizeof(int));
        
        //thread for every point on the donut, grid for small circle, block for big circle
        calcPointCuda<<<6.28/thetaStep,6.28/phiStep>>>(A, B, oozBuf, lumBuf, depthBuf);

        cudaDeviceSynchronize();    
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("[!]CUDA error:calcpoint: %s\n", cudaGetErrorString(error));
            exit(1);
        }

        renderPixCuda<<<resH,resW>>>(oozBuf, lumBuf, depthBuf, output);

        cudaDeviceSynchronize();    
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("[!]CUDA error:renderpix: %s\n", cudaGetErrorString(error));
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






