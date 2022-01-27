# Whale Shark View Classification

![](https://user-images.githubusercontent.com/11778655/151292900-dcdb5be7-6c7d-4464-a44b-cf796566764c.jpg)

```
1. Download and prepare data.
2. Train CNNs:
python train.py
3. Test metrics:
python test.py
4. Use ViewClassifier:
```
```
from inference import ViewClassifier

vc = ViewClassifier()
image = cv2.imread("path/to/image.png")
result = vc.predict(image)

if result >= 0.5:
    print("GOOD: {0:.2f}".format(result))
else:
    print("BAD: {0:.2f}".format(result))
```