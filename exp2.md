import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Quadtree Node
# ===============================
class QuadTree:
    def __init__(self, img, x, y, size, threshold):
        self.x = x
        self.y = y
        self.size = size
        self.threshold = threshold
        self.children = []
        self.img = img

        self.build()

    def is_homogeneous(self):
        region = self.img[self.y:self.y+self.size, self.x:self.x+self.size]
        return np.std(region) < self.threshold

    def build(self):
        if self.size <= 8 or self.is_homogeneous():
            return

        half = self.size // 2

        self.children = [
            QuadTree(self.img, self.x, self.y, half, self.threshold),
            QuadTree(self.img, self.x+half, self.y, half, self.threshold),
            QuadTree(self.img, self.x, self.y+half, half, self.threshold),
            QuadTree(self.img, self.x+half, self.y+half, half, self.threshold)
        ]

    def draw(self, output):
        if not self.children:
            cv2.rectangle(output,
                          (self.x, self.y),
                          (self.x+self.size, self.y+self.size),
                          (0, 255, 0), 1)
        else:
            for child in self.children:
                child.draw(output)


# ===============================
# Main Program
# ===============================
def main():
    image = cv2.imread("image.jpg")
    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = 10  # Variance threshold
    qt = QuadTree(gray, 0, 0, 256, threshold)

    output = image.copy()
    qt.draw(output)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Quadtree Decomposition")
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
    output
<img width="543" height="615" alt="image" src="https://github.com/user-attachments/assets/0a0ade8a-5765-4444-b00e-f6abc53ab888" />
