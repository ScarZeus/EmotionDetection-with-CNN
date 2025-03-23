import os
from data_loader import get_data_generators, IMG_HEIGHT, IMG_WIDTH
from model import build_model
from train import train_model, evaluate_model, plot_history

def main():


    base_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    
    train_generator, test_generator = get_data_generators(train_dir, test_dir)
    

    num_classes = len(train_generator.class_indices)
    print("Detected Classes:", train_generator.class_indices)
    
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 1) 
    model = build_model(input_shape, num_classes)
    model.summary()
    
    EPOCHS = 20
    history = train_model(model, train_generator, test_generator, EPOCHS)
    
    evaluate_model(model, test_generator)
    
    plot_history(history)

if __name__ == '__main__':
    main()
