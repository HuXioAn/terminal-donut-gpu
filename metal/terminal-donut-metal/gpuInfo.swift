import CoreGraphics
import Metal

func getGPUInfo(device: MTLDevice) -> String {
    var result = String()

    result.append("\n\(device.name)")

    // GPU location
    result.append("\n\t\(device.location) GPU")
    let properties = [(device.isLowPower, "low power"), (device.isHeadless, "headless"), (device.isRemovable, "removable")]
    .filter(\.0).map(\.1).joined(separator: ", ")
    if !properties.isEmpty {
    result.append(" (\(properties))")
    }

    // GPU memory
    if device.hasUnifiedMemory {
    result.append("\n\tUnified memory (shared with CPU)")
    } else {
    result.append("\n\tDiscrete memory")
    }

    result.append("\n\t\tmax recommended working set: \(device.recommendedMaxWorkingSetSize.formatted(.byteCount(style: .memory)))")
    if device.maxTransferRate > 0 {
    result.append("\n\t\tmax transfer rate: \(device.maxTransferRate.formatted(.byteCount(style: .memory)))/s")
    }

    // Computing
    result.append("\n\tGeneral Purpose Computing")
    result.append("\n\t\tmax threadgroup memory: \(device.maxThreadgroupMemoryLength.formatted(.byteCount(style: .memory)))")
    let t = device.maxThreadsPerThreadgroup
    result.append("\n\t\tmax threads per threadgroup: [\(t.width), \(t.height), \(t.depth)]")
    
    result.append("\n")
    return result
}




