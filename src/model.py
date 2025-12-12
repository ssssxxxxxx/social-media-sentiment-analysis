import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img, title="Image", cmap=None):
    """Utility to show images in Jupyter"""
    plt.figure(figsize=(6, 6))
    if len(img.shape) == 2:  # Grayscale
        plt.imshow(img, cmap=cmap if cmap else "gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


class ComputerVisionAssignment:
    def __init__(self, image_path, binary_image_path):
        self.image = cv2.imread(image_path)
        self.binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    def load_and_analyze_image(self):
        Image_data_type = type(self.image)
        Pixel_data_type = self.image.dtype
        Image_shape = self.image.shape

        print(f"Image data type: {Image_data_type}")
        print(f"Pixel data type: {Pixel_data_type}")
        print(f"Image dimensions: {Image_shape}")

        show_image(self.image, "Original Image")
        return Image_data_type, Pixel_data_type, Image_shape

    def create_red_image(self):
        red_image = self.image.copy()
        red_image[:, :, 1] = 0  # Green = 0
        red_image[:, :, 2] = 0  # Blue = 0
        show_image(red_image, "Red Image")
        return red_image

    def create_photographic_negative(self):
        negative_image = 255 - self.image
        show_image(negative_image, "Photographic Negative")
        return negative_image

    def swap_color_channels(self):
        swapped_image = self.image.copy()
        swapped_image[:, :, [0, 2]] = swapped_image[:, :, [2, 0]]
        show_image(swapped_image, "Swapped R and B")
        return swapped_image

    def foliage_detection(self):
        B, G, R = cv2.split(self.image)
        mask = (G >= 50) & (R < 50) & (B < 50)
        foliage_image = np.zeros_like(G)
        foliage_image[mask] = 255
        show_image(foliage_image, "Foliage Detection", cmap="gray")
        return foliage_image

    def shift_image(self):
        rows, cols = self.image.shape[:2]
        M = np.float32([[1, 0, 200], [0, 1, 100]])
        shifted_image = cv2.warpAffine(self.image, M, (cols, rows))
        show_image(shifted_image, "Shifted Image")
        return shifted_image

    def rotate_image(self):
        rotated_image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        show_image(rotated_image, "Rotated 90° Clockwise")
        return rotated_image

    def similarity_transform(self, scale, theta, shift):
        rows, cols = self.image.shape[:2]
        theta_rad = np.deg2rad(theta)

        # Scale + Rotate matrix
        M = np.array([
            [scale * np.cos(theta_rad), scale * np.sin(theta_rad), 0],
            [-scale * np.sin(theta_rad), scale * np.cos(theta_rad), 0]
        ])
        M[:, 2] = shift

        transformed_image = cv2.warpAffine(
            self.image, M, (cols, rows), flags=cv2.INTER_NEAREST
        )
        show_image(transformed_image, "Similarity Transform")
        return transformed_image

    def convert_to_grayscale(self):
        B, G, R = cv2.split(self.image)
        gray = (3 * R + 6 * G + 1 * B) / 10
        gray_image = gray.astype(np.uint8)
        show_image(gray_image, "Custom Grayscale", cmap="gray")
        return gray_image

    def compute_moments(self):
        moments = cv2.moments(self.binary_image)

        m00 = moments["m00"]
        m10 = moments["m10"]
        m01 = moments["m01"]

        x_bar = m10 / m00
        y_bar = m01 / m00

        mu20 = moments["mu20"]
        mu02 = moments["mu02"]
        mu11 = moments["mu11"]

        print("First-Order Moments:")
        print(f"M00={m00}, M10={m10}, M01={m01}")
        print("Centralized:")
        print(f"x̄={x_bar:.2f}, ȳ={y_bar:.2f}")
        print("Second-Order Centralized:")
        print(f"mu20={mu20}, mu02={mu02}, mu11={mu11}")

        show_image(self.binary_image, "Binary Image")
        return m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11

    def compute_orientation_and_eccentricity(self):
        moments = cv2.moments(self.binary_image)

        x_bar = moments["m10"] / moments["m00"]
        y_bar = moments["m01"] / moments["m00"]

        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]

        theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
        orientation = np.degrees(theta)

        numerator = (mu20 - mu02) ** 2 + 4 * mu11 ** 2
        lambda1 = (mu20 + mu02 + np.sqrt(numerator)) / 2
        lambda2 = (mu20 + mu02 - np.sqrt(numerator)) / 2
        eccentricity = np.sqrt(1 - lambda2 / lambda1)

        glasses_with_ellipse = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
        center = (int(x_bar), int(y_bar))
        axes = (int(np.sqrt(lambda1)), int(np.sqrt(lambda2)))
        angle = -orientation  # cv2 angle is CCW
        cv2.ellipse(glasses_with_ellipse, center, axes, angle, 0, 360, (0, 0, 255), 1)

        print(f"Orientation: {orientation:.2f}°")
        print(f"Eccentricity: {eccentricity:.4f}")

        show_image(glasses_with_ellipse, "Ellipse Fit")
        return orientation, eccentricity, glasses_with_ellipse


# Example Run
if __name__ == "__main__":
    assignment = ComputerVisionAssignment("picket_fence.jpg", "glasses_outline.png")

    assignment.load_and_analyze_image()
    assignment.create_red_image()
    assignment.create_photographic_negative()
    assignment.swap_color_channels()
    assignment.foliage_detection()
    assignment.shift_image()
    assignment.rotate_image()
    assignment.similarity_transform(scale=2.0, theta=45.0, shift=[100, 100])
    assignment.convert_to_grayscale()
    assignment.compute_moments()
    assignment.compute_orientation_and_eccentricity()
