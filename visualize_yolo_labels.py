import cv2
import os

def draw_yolo_labels(image_path, label_path, output_path):
    # 检查输入文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"标签文件不存在: {label_path}")
        
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    height, width, _ = image.shape

    # 读取标签文件
    try:
        with open(label_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        raise Exception(f"读取标签文件时出错: {str(e)}")

    for line in lines:
        try:
            # 解析标签
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

            # 转换为图像坐标
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            # 计算边界框的左上角和右下角坐标
            x1 = int(max(0, x_center - bbox_width / 2))
            y1 = int(max(0, y_center - bbox_height / 2))
            x2 = int(min(width, x_center + bbox_width / 2))
            y2 = int(min(height, y_center + bbox_height / 2))

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在边界框上绘制类别ID
            cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"处理标签时出错: {str(e)}")
            continue

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存带有标签的图像
    try:
        cv2.imwrite(output_path, image)
    except Exception as e:
        raise Exception(f"保存图像时出错: {str(e)}")

if __name__ == "__main__":
    # 示例图像和标签路径
    image_path = r'.\wheat_dataset\train\0a3cb453f.jpg'
    label_path = r'.\wheat_dataset\labels\0a3cb453f.txt'
    # 修改输出路径，确保包含完整的文件名
    output_path = r'.\wheat_dataset\output\0a3cb453f_labeled.jpg'

    try:
        draw_yolo_labels(image_path, label_path, output_path)
        print(f"已成功保存标注后的图像到: {output_path}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")