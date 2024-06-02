//
//  metalInterface.swift
//  terminal-donut-metal
//
//  Created by Anton on 2024-06-02.
//

import Foundation
import CoreGraphics
import Metal


class MetalInterface{
    
    var device : MTLDevice
    var pipeLineState : MTLComputePipelineState!
    var commandQueue : MTLCommandQueue!
    var kernel : MTLFunction!
    
    init(device: MTLDevice, kernalName: String) {
        self.device = device
        
        guard let library = device.makeDefaultLibrary() else{
            print("[!]MetalInterface: Can not get the default library")
            return
        }
        
        guard let kernel = library.makeFunction(name: kernalName) else{
            print("[!]MetalInterface: Can not get the function: ", kernalName)
            return
        }
        self.kernel = kernel
        
        guard let pipeLineState = try? device.makeComputePipelineState(function: self.kernel!) else{
            print("[!]MetalInterface: Can not get the function")
            return
        }
        self.pipeLineState = pipeLineState
        
        guard let cmdQueue = device.makeCommandQueue() else{
            print("[!]MetalInterface: Can not get the command queue")
            return
        }
        self.commandQueue = cmdQueue
        
    }
    
    func compute(threadNum: Int) -> MTLCommandBuffer{
        let cmdBuffer = commandQueue.makeCommandBuffer()
        assert(cmdBuffer != nil)
        let computeEncoder = cmdBuffer?.makeComputeCommandEncoder()
        assert(computeEncoder != nil)
        
        computeEncoder?.setComputePipelineState(pipeLineState)
        
        //set buffer
        
        //thread alloc
        var threadGroupSizeMax = pipeLineState.maxTotalThreadsPerThreadgroup
        if(threadGroupSizeMax > threadNum){
            threadGroupSizeMax = threadNum
        }
        let threadGroupSize = MTLSizeMake(threadGroupSizeMax, 1, 1)
        let gridSize = MTLSizeMake(threadNum, 1, 1) // one-dimension
        computeEncoder?.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        computeEncoder?.endEncoding()
        cmdBuffer?.commit()
        
        return cmdBuffer!
    }
    
    
    
    func printDeviceinfo(){
        print(getGPUInfo(device: device))
    }
    
}


