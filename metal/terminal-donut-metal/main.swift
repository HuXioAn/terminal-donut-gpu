//
//  main.swift
//  terminal-donut-metal
//
//  Created by Anton on 2024-06-02.
//

import Foundation
import CoreGraphics
import Metal



func main() {
    let param = DonutParam()

    let outputSize = param.resW * param.resH
    
    var A: Float = .pi / 2
    var B: Float = 0
    
    //initianlize the device
    let device = MTLCopyAllDevices()[0]
    let calcPoint = MetalInterface(device: device, kernalName: "calcPointMetal")
    let renderPix = MetalInterface(device: device, kernalName: "renderPixMetal")
    //metal.printDeviceinfo()
    
    let oozBuf = calcPoint.createAndLoadBuffer(length: outputSize * MemoryLayout<Float>.size * param.BUF_DEPTH,
                                               bufferOption: MTLResourceOptions.storageModePrivate,
                                               argIndex: 2)
    let lumBuf = calcPoint.createAndLoadBuffer(length: outputSize * MemoryLayout<Float>.size * param.BUF_DEPTH,
                                               bufferOption: MTLResourceOptions.storageModePrivate,
                                               argIndex: 3)
    let depthBuf = calcPoint.createAndLoadBuffer(length: outputSize * MemoryLayout<UInt>.size,
                                               bufferOption: MTLResourceOptions.storageModePrivate,
                                               argIndex: 4)
    
    if oozBuf == nil || lumBuf == nil || depthBuf == nil {
        print("[!] Error creating buffer")
        return
    }
    
    renderPix.loadBuffer(buffer: oozBuf!, argIndex: 0)
    renderPix.loadBuffer(buffer: lumBuf!, argIndex: 1)
    renderPix.loadBuffer(buffer: depthBuf!, argIndex: 2)
    let outputBuf = renderPix.createAndLoadBuffer(length: outputSize * MemoryLayout<CChar>.size,
                                                  bufferOption: MTLResourceOptions.storageModeShared,
                                                  argIndex: 3)
    if outputBuf == nil{
        print("[!] Error creating buffer")
        return
    }
    let output = outputBuf?.contents().bindMemory(to: CChar.self, capacity: outputSize)
    
    while true {
        withUnsafePointer(to: A) { pointer in
            calcPoint.loadArg(fromPtr: UnsafeRawPointer(pointer).bindMemory(to: Any.self, capacity: 1), length: 4, argIndex: 0)
        }
        withUnsafePointer(to: B) { pointer in
            calcPoint.loadArg(fromPtr: UnsafeRawPointer(pointer).bindMemory(to: Any.self, capacity: 1), length: 4, argIndex: 1)
        }
        //device computation
        let calcCMDBuffer = calcPoint.compute(threadPerGroup: Int(6.28 / param.phiStep), groupPerGrid: Int(6.28 / param.thetaStep))
        calcCMDBuffer?.waitUntilCompleted()
        let renderCMDBuffer = renderPix.compute(threadPerGroup: param.resW, groupPerGrid: param.resH)
        renderCMDBuffer?.waitUntilCompleted()
        
        calcPoint.reLoad()
        renderPix.reLoad()
        
        calcPoint.loadBuffer(buffer: oozBuf!, argIndex: 2)
        calcPoint.loadBuffer(buffer: lumBuf!, argIndex: 3)
        calcPoint.loadBuffer(buffer: depthBuf!, argIndex: 4)
        
        renderPix.loadBuffer(buffer: oozBuf!, argIndex: 0)
        renderPix.loadBuffer(buffer: lumBuf!, argIndex: 1)
        renderPix.loadBuffer(buffer: depthBuf!, argIndex: 2)
        renderPix.loadBuffer(buffer: outputBuf!, argIndex: 3)
        
        //Print
        print("\u{1B}[H", terminator: "")
        var pointer = output!
        for k in 0..<(param.resH * param.resW) {
            if k % param.resW == 0 && k != 0 {
                print()
            }
            print(Character(UnicodeScalar(UInt8(pointer.pointee))), terminator: "")
            pointer += 1
        }
        
        A += 0.04
        B += 0.02
        usleep(10000)
    }
}

main()


