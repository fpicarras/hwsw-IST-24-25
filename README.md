# CNN Accelerator

This repository provides a **CNN Accelerator**, a high-performance hardware module for accelerating convolutional neural network (CNN) inference on FPGA platforms. It includes HLS-based designs, software drivers, testbenches, and performance reports.

## Features

- **Configurable Convolution Core**: Parameterizable kernel size, stride, and channel dimensions.
- **Streaming Dataflow**: Fully pipelined architecture for low-latency processing.
- **DMA Interface**: IP Configurable via DMA AXI4-Lite Interface to increase throughput and reduce latency.
- **AXI4-Stream Data Interface**: High-throughput data movement between memory and accelerator.
- **Hide Latency Techniques**: Application hides the IP latency by operating in another image.
- **Performance Reporting**: Synthesis and implementation reports detailing latency, throughput, and resource utilization.

## Repository Structure

```
├── hls/  # HLS project and testbenches for the CNN Accelerator
├── app/  # Software drivers and example applications
├── tb/   # Testbenches and input datasets
├── doc/  # Project Documentation
├── images/  # Input images for the CNN
├── LICENSE   # MIT License file
└── README.md # Project overview and instructions
```

## Prerequisites

- **Vivado HLS** (version 2023.2 or later)
- **Vivado Design Suite** (for implementation reports, optional)
- **GNU Make**
- **ARM GCC Toolchain** (or equivalent, for host applications)

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

## Contact

For questions or feedback, open an issue or reach out to the maintainers via GitHub Discussions.
