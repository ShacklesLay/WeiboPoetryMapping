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
            if not os.path.isdir(sub_file_path):
                continue
            raw_data_path = os.path.join(sub_file_path, os.listdir(sub_file_path)[0])
            
            with open(raw_data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    text = line.split(',',maxsplit=3)[2]
                    id  = line.split(',',maxsplit=3)[0]
                    content.append({"Province":name, "City":subname, "Content":text, "ID":id})
            
        # 将内容写入文件
        with jsonlines.open(os.path.join(file_path,'data.json'), 'w') as f:
            f.write_all(content)

def extract_groundtruth():
    # 提取出csv文件中的groundtruth部分，组成一个json文件
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/data/groundtruth.csv'
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            pro, gdp, med, engel = line.split(',')
            data[pro] = {"GDP":float(gdp), "Med":float(med), "Engel":float(engel)}
    # 将内容写入文件
    with open('./data/label.json','w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))
        

if __name__ == '__main__':
    # extract_content()
    extract_groundtruth()

