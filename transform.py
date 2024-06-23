import os
import argparse
from PIL import Image

def process_images(input_dir, output_dir_4, output_dir_8):
    # 确保输出目录存在
    os.makedirs(output_dir_4, exist_ok=True)
    os.makedirs(output_dir_8, exist_ok=True)

    # 遍历输入目录中的所有 JPG 文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # 构建完整的输入路径和输出路径
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path_4 = os.path.join(output_dir_4, output_filename)
            output_path_8 = os.path.join(output_dir_8, output_filename)

            # 打开图像
            with Image.open(input_path) as img:
                # 计算新尺寸
                size_4 = (int(img.width * 0.25), int(img.height * 0.25))
                size_8 = (int(img.width * 0.125), int(img.height * 0.125))

                # 调整图像大小
                resized_img_4 = img.resize(size_4, Image.Resampling.LANCZOS)
                resized_img_8 = img.resize(size_8, Image.Resampling.LANCZOS)

                # 保存为 PNG 格式
                resized_img_4.save(output_path_4, 'PNG')
                resized_img_8.save(output_path_8, 'PNG')

            print(f"Processed {filename} -> {output_filename}")

    print("All images have been processed.")

def main():
    # 设置参数解析
    parser = argparse.ArgumentParser(description="Process images: resize and convert to PNG.")
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., orange).')

    args = parser.parse_args()

    # 固定路径的前缀和后缀部分
    base_path = '/mnt/ly/models/FinalTerm/mission3/nerf-pytorch/data/nerf_llff_data'
    
    input_dir = os.path.join(base_path, args.dataset, 'images')
    output_dir_4 = os.path.join(base_path, args.dataset, 'images_4')
    output_dir_8 = os.path.join(base_path, args.dataset, 'images_8')

    # 调用处理图像的函数
    process_images(input_dir, output_dir_4, output_dir_8)

if __name__ == '__main__':
    main()