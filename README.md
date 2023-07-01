# Asaphus Vision Code Challenge

## Challenge description
Write a Python script that reads an image from a file as grayscale, and finds
the four non-overlapping 5x5 patches with highest average brightness. Take
the patch centers as corners of a quadrilateral, calculate its area in
pixels, and draw the quadrilateral in red into the image and save it in PNG
format. Use the opencv-python package for image handling. Write test cases.


### Install the dependencies (using Python 3.8):
```sh
pip install -r requirements.txt
```

### Run the script
```sh
python solution.py --image_path <path-to-your-image>
```

### Run tests

```sh
pytest test_solution.py -vv
```

---
