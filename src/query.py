import pickle
# import os  <-- os 在这里没用到，可以省去
from pathlib import Path

dataset_dir = Path('.') / 'dataset'

# 【雷达 1】看看 Python 到底去电脑的哪个绝对路径下找文件夹了
print(f"🔍 正在搜索的绝对路径是: {dataset_dir.resolve()}")

pkl_files = list(dataset_dir.glob('*.pkl'))

# 【雷达 2】看看到底找到了几个文件
print(f"📦 共找到了 {len(pkl_files)} 个 .pkl 文件\n")
print("-" * 30)

for file_path in pkl_files:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
        if isinstance(data, dict):
            # .name 可以只打印文件名，更干净
            print(f"✅ 文件名: {file_path.name}") 
            print("前 5 个键值对如下：")
            first_5_items = list(data.items())[:5]
            for key, value in first_5_items:
                print(f"  键 : {key}  ==>  值: {value}")
                
        else:
            # 【雷达 3】如果不是字典，告诉我是什么类型
            print(f"⚠️ 文件 {file_path.name} 被打开了，但它不是字典，它的类型是：{type(data)}")
            
    print("-" * 30)