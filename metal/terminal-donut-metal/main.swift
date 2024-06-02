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
    var output = Array(repeating: Character(" "), count: param.resW * param.resH)
    
    var A: Float = .pi / 2
    var B: Float = 0
    
    //initianlize the device
    let device = MTLCopyAllDevices()[0]
    let metal = MetalInterface(device: device, kernalName: "donutKernel")
    //metal.printDeviceinfo()
    
    while true {
        
        //device computation
        
        //Print
        print("\u{1B}[H", terminator: "")
        for k in 0..<(param.resH * param.resW) {
            if k % param.resW == 0 && k != 0 {
                print()
            }
            print(output[k], terminator: "")
        }
        
        A += 0.04
        B += 0.02
        usleep(10000)
    }
}

main()


