import os

base_dir = r"D:\.kaggle\文本情绪反应预测"

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            filepath = os.path.join(root, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f'{subindent}{f} ({size_mb:.2f} MB)')

if __name__ == '__main__':
    list_files(base_dir)