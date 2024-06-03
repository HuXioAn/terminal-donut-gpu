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
    var cmdBuffer : MTLCommandBuffer!
    var computeEncoder : MTLComputeCommandEncoder!
    
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
        
        cmdBuffer = commandQueue.makeCommandBuffer()
        assert(cmdBuffer != nil)
        computeEncoder = cmdBuffer?.makeComputeCommandEncoder()
        assert(computeEncoder != nil)
        computeEncoder?.setComputePipelineState(pipeLineState)
        
    }
    
    func loadBuffer(length: Int, bufferOption: MTLResourceOptions, argIndex: Int) -> MTLBuffer?{
        
        let buffer = device.makeBuffer(length: length, options: bufferOption)
        if buffer == nil {
            print("[!]MetalInterface: Can not allocate the buffer")
            return nil
        }
        
        computeEncoder.setBuffer(buffer, offset: 0, index: argIndex)
        
        return buffer
    }
    
    func loadBuffer(fromPtr: UnsafePointer<Any>, length: Int, bufferOption: MTLResourceOptions, argIndex: Int) -> MTLBuffer?{
        
        let buffer = device.makeBuffer(bytes: fromPtr, length: length, options: bufferOption)
        if buffer == nil {
            print("[!]MetalInterface: Can not allocate the buffer")
            return nil
        }
        
        computeEncoder.setBuffer(buffer, offset: 0, index: argIndex)
        
        return buffer
    }
    
    func loadArg(fromPtr: UnsafePointer<Any>, length: Int, argIndex: Int){
        //used for small non-persisting mem, <4KB
        computeEncoder.setBytes(fromPtr, length: length, index: argIndex)
    }
    
    func compute(threadPerGroup: Int, groupPerGrid: Int) -> MTLCommandBuffer?{
        let threadNum = threadPerGroup * groupPerGrid
        //thread alloc
        var threadGroupSizeMax = pipeLineState.maxTotalThreadsPerThreadgroup
        if(threadGroupSizeMax < threadPerGroup){
            print("[!] threadPerGroup: ", threadPerGroup, " > maxTotalThreadsPerThreadgroup: ", threadGroupSizeMax)
            return nil
        }
        let threadGroupSize = MTLSizeMake(threadPerGroup, 1, 1)
        let gridSize = MTLSizeMake(threadNum, 1, 1) // one-dimension
        computeEncoder?.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        computeEncoder?.endEncoding()
        cmdBuffer?.commit()
        
        return cmdBuffer
    }
    
    
    
    func printDeviceinfo(){
        print(getGPUInfo(device: device))
    }
    
}


