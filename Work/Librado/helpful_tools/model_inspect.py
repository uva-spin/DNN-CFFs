import tensorflow as tf
import argparse
from BHDVCS_tf_modified import TotalFLayer
import numpy as np
import matplotlib.pyplot as plt

def plot_model_architecture(model, output_file='model_plot.png'):
    tf.keras.utils.plot_model(model, to_file=output_file, show_shapes=True)
    print(f"Model architecture plot saved as {output_file}")

def interactive_mode(model):
    while True:
        cmd = input("Enter command (type 'help' for options or 'exit' to quit): ").strip()
        if cmd == 'exit':
            break
        elif cmd == 'help':
            print("Available commands: summary, plot [filename], input_output, layer_info, parameters, exit")
        elif cmd == 'summary':
            model.summary()
        elif cmd.startswith('plot'):
            _, output_file = cmd.split()
            plot_model_architecture(model, output_file)
        elif cmd == 'input_output':
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
        elif cmd == 'layer_info':
            for layer in model.layers:
                print(f"Layer: {layer.name} | Type: {layer.__class__.__name__} | Output Shape: {layer.output_shape}")
        elif cmd == 'parameters':
            print(f"Total parameters: {model.count_params()}")
            print(f"Trainable parameters: {np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])}")
            print(f"Non-trainable parameters: {np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])}")
        else:
            print("Unknown command.")

def main():
    parser = argparse.ArgumentParser(description='Inspect TensorFlow model file.')
    parser.add_argument('model_path', help='Path to the TensorFlow model file (.h5)')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, custom_objects={'TotalFLayer': TotalFLayer})

    interactive_mode(model)

if __name__ == "__main__":
    main()
