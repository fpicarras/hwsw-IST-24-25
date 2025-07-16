import struct
import sys

FORMAT = 16

def read_floats_from_binary(filename):
    floats = []
    with open(filename, 'rb') as f:
        while True:
            bytes_read = f.read(4)
            if not bytes_read:
                break
            value = struct.unpack('f', bytes_read)[0]
            floats.append(value)
    return floats

def convert_to_Q15(floats):
    q15_values = []
    for value in floats:
        if value < -1.0 or value > 1.0:
            raise ValueError("Value out of range for Q15 conversion: {}".format(value))
        q15_value = int(value * (2**(FORMAT-1)) + 0.5)  # Convert to Q15
        q15_values.append(q15_value)
    return q15_values

def quantization_error(original, quantized):
    float_quantized = [q / (2**(FORMAT-1)) for q in quantized]
    error = [100*abs(o - q) for o, q in zip(original, float_quantized)]
    max_error = max(error)
    print(f"Max quantization error: {max_error:.6f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <binary_file>")
        sys.exit(1)
    filename = sys.argv[1]
    float_values = read_floats_from_binary(filename)
    q15_values = convert_to_Q15(float_values)

    errors = quantization_error(float_values, q15_values)
    
    # Genmerate the .bin file with the weights
    # Write the 2**FORMAT BITS of the value to the file
    output_name = filename.replace('.bin', '_q' + str(FORMAT-1) + '.bin')
    with open(output_name, 'wb') as f:
        for value in q15_values:
            f.write(struct.pack('h', value))
        
    # Valideate the written file
    with open(output_name, 'rb') as f:
        read_values = f.read()
        if len(read_values) != len(q15_values) * 2:
            print("Error: The written file size does not match the expected size.")
        else:
            print("File written successfully with quantized values.")