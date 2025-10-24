import cv2, glob
import matplotlib.pyplot as plt


def visualize_result(input_path: str, res_path: str, c_img: int = 5):
    input_paths = sorted(glob.glob(f"{input_path}/*.png"))
    print("Number of input images:", len(input_paths))

    res_paths = sorted(glob.glob(f"{res_path}/*.png"))
    print("Number of result images:", len(res_paths))

    for i, (inp, res) in enumerate(zip(input_paths[:c_img], res_paths[:c_img])):
        img_in = cv2.imread(inp)[:, :, ::-1]  # BGR -> RGB
        img_res = cv2.imread(res)[:, :, ::-1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(img_in)
        axes[0].set_title(f"Input - Image {i+1}")
        axes[0].axis("off")

        axes[1].imshow(img_res)
        axes[1].set_title(f"RLP Output - Image {i+1}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
