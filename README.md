# Asaphus Vision Code Challenge 🚀

## Challenge description 📝
Write a Python script that reads an image from a file as grayscale, and finds the four non-overlapping 5x5 patches with highest average brightness. Take the patch centers as corners of a quadrilateral, calculate its area in pixels, and draw the quadrilateral in red into the image and save it in PNG format. Use the opencv-python package for image handling. Write test cases.

## Installation 🔧
Install the required Python dependencies by running:
```sh
pip install -r requirements.txt
```

### Building Cython Module (Optional) 🏗️

This project uses a Cython module for performance optimization. Make sure that a C compiler is installed on your system. Linux or Mac systems typically use GCC, while Windows systems can use Microsoft Visual C++.

After ensuring the C compiler is installed, navigate to the project directory and run the following command to build the Cython module:

```sh
python setup.py build_ext --inplace
```

### Usage 💻
After installation, you can use the script as follows:
```sh
python solution.py --input_image_path <path-to-your-image>
```

**Example**: You can use the `Lenna_top.jpg` in the `assets` directory:
```sh
python solution.py --input_image_path assets/Lenna_top.jpg
```

### Running tests 🧪
This project uses pytest for testing. To run the tests, use the following command:
```sh
pytest test_solution.py -vv
```

---
