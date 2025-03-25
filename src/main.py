import os
import argparse
from data_loader import get_data_generators, IMG_HEIGHT, IMG_WIDTH
from model import build_model
from train import train_model, evaluate_model, plot_history

def main():
    parser = argparse.ArgumentParser(description="Train Emotion Detection Model")
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default=os.path.join(os.path.dirname(__file__), 'dataset'),
        help="Path to the dataset folder containing 'train' and 'test' subdirectories"
    )
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--save_model', type=str, default='emotion_model.h5', help="Filename to save the trained model")
    args = parser.parse_args()

    train_dir = os.path.join(args.dataset_path, 'train')
    test_dir = os.path.join(args.dataset_path, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Dataset directories not found in {args.dataset_path}")
        return

    train_generator, test_generator = get_data_generators(train_dir, test_dir)
    print("Detected Classes:", train_generator.class_indices)

    input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
    num_classes = len(train_generator.class_indices)
    model = build_model(input_shape, num_classes)
    model.summary()


    history = train_model(model, train_generator, test_generator, args.epochs)

    evaluate_model(model, test_generator)


    plot_history(history)

    model.save(args.save_model)
    print(f"Model saved to {args.save_model}")

if __name__ == '__main__':
    main()
