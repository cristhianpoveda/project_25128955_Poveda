# RGB-D Hand Gesture Recognition

This project implements a deep learning pipeline to recognize and segment hand gestures using RGB-D (Red, Green, Blue + Depth) data. It uses a custom **Multi-Task U-Net** to simultaneously predict the gesture class (e.g., *Like, Peace, Stop*) and generate a binary segmentation mask of the hand.

The model was trained on 32 students and achieved **90.48% Accuracy** and **86.09% IoU** on the test set.

## Project Structure

```text
.
├── check_data_quality.py        # Utility to scan the dataset for corrupt or missing files.
├── check_loader.py              # Verifies that the dataloader and augmentations are working correctly.
├── requirements.txt             # List of Python dependencies required to run the project.
├── results/                     # Directory containing training logs, plots, and visualization outputs.
│   ├── prediction_stack.png     # Visual sample of model predictions (RGB vs Ground Truth vs Prediction).
│   └── training_curves.png      # Graphs showing Loss and Accuracy over epochs.
├── src/
│   ├── dataloader_augmentation.py # Custom Dataset class that handles loading and real-time data augmentation.
│   ├── evaluate.py              # Script to evaluate the trained model on the held-out Test Set.
│   ├── model.py                 # PyTorch definition of the Multi-Task U-Net architecture (4-channel input).
│   ├── train_validation.py      # Main script to train the model, handling train/val splits and saving weights.
│   ├── utils.py                 # Helper functions for file path management and data loading.
│   └── visualise.py             # Script to generate qualitative visualization images from the test set.
├── test_model_architecture.py   # Unit tests to verify model input/output tensor shapes.
└── weights/                     # Directory where the best trained model checkpoints are saved.
