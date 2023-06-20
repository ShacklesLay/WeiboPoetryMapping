import os 
import json
import jsonlines

def extract_content():
    # 取出各个csv文件中的正文部分，组成一个json文件
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/data/weibo'
    for name in os.listdir(dir_path):
        content = []
        file_path = os.path.join(dir_path, name)
        for subname in os.listdir(file_path):
            sub_file_path = os.path.join(file_path, subname)
            
            raw_data_path = os.path.join(sub_file_path, os.listdir(sub_file_path)[0])
            
            with open(raw_data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    text = line.split(',',maxsplit=3)[2]
                    content.append({"Province":name, "City":subname, "Content":text})
            
        # 将内容写入文件
        with jsonlines.open(os.path.join(file_path,'data.json'), 'w') as f:
            f.write_all(content)
        

if __name__ == '__main__':
    extract_content()

