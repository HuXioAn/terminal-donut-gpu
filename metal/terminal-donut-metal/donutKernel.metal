//
//  donut.metal
//  terminal-donut-metal
//
//  Created by Anton on 2024-06-02.
//

#include <metal_stdlib>
#include <metal_math>
#include <metal_atomic>
using namespace metal;

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


kernel void calcPointMetal(device const float& A , 
                           device const float& B,
                           device float* oozBuf,
                           device float* lumBuf,
                           device atomic_uint* depthBuf,
                           uint groupInGrid [[threadgroup_position_in_grid]],
                           uint threadInGroup [[thread_index_in_threadgroup]]
                           ){
    
    //get the theta and phi from index
    float theta = groupInGrid * thetaStep;
    float phi = threadInGroup * phiStep;

    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    //common value
    float sinA = sin(A);
    float cosA = cos(A);
    float sinB = sin(B);
    float cosB = cos(B);


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
    int depth = atomic_fetch_add_explicit(depthBuf+index, 1, memory_order_relaxed);
    //int depth = atomicAdd(depthBuf+index, 1);
    
    //if(depth > BUF_DEPTH)printf("%d\n", depth);
    //store the ooz and lum
    oozBuf[indexBuf + depth] = ooz;
    lumBuf[indexBuf + depth] = lum;



}


kernel void renderPixMetal(device float* oozBuf,
                           device float* lumBuf,
                           device uint* depthBuf,
                           device char* output,
                           uint groupInGrid [[threadgroup_position_in_grid]],
                           uint threadInGroup [[thread_index_in_threadgroup]]
                           ){
    //results from previous kernel
    int x = threadInGroup;
    int y = groupInGrid;

    int index = y * resW + x;
    int indexBuf = BUF_DEPTH * index;

    float oozMax = 0;
    
    output[index] = ' ';
    for(uint i = 0; i < depthBuf[index]; i++){
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
    depthBuf[index] = 0;
}


