import argparse
import ffmpeg
# python transform_mov2gif.py /path/to/input.mov /path/to/output.gif
def convert_mov_to_gif(input_path, output_path):
    try:
        # 使用 ffmpeg 将视频转换为 GIF
        ffmpeg.input(input_path).output(output_path, vf='fps=10,scale=320:-1:flags=lanczos', loop=0).run()
        print(f'转换完成: {output_path}')
    except ffmpeg.Error as e:
        print(f'转换失败: {e}')

def main():
    parser = argparse.ArgumentParser(description="将 MOV 文件转换为 GIF 文件")
    parser.add_argument('input_mov', type=str, help='输入 MOV 文件路径')
    parser.add_argument('output_gif', type=str, help='输出 GIF 文件路径')
    
    args = parser.parse_args()
    
    convert_mov_to_gif(args.input_mov, args.output_gif)

if __name__ == '__main__':
    main()