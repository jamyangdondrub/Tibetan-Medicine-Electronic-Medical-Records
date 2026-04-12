#以下脚本将藏医电子病历原始标注格式转换为 BIOES 序列标注格式：
import re
data=[]
with open('merge_test.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.replace('།','་།་')
        line = line.rstrip('\n').split('/')
        first = line[0]
        #print(first)
        gidr = re.findall(r'[\u0F00-\u0FFF]+|.',first)
        #print(gidr)
        dire = '་'.join(gidr).replace('་་','་').split('་')
        # first = first.split('་')
        # print(first)
        for i in dire:
            #print(i)
            if i != '':
                data.append(i+'\tO')
        for part in range(1,len(line)):
            jia = line[part]
            if re.search(r'[\w\W/-a-zA-Z]+', jia):  
                    jia = re.split(r'([\w\W/-a-zA-Z]+)', jia) 
                    ner = jia[1].split('-')
                    #print(ner)
                    #匹配藏文子符和任意符号
                    yang= r'[\u0F00-\u0FFF]+|.'
                    temp = re.findall(yang,ner[0])
                    new_list = ['་'.join(temp).replace('་་','་')]
                    #print(new_list)
                    #切分英文和藏文字符
                    pattern = r'[\u0F00-\u0FFF]+|[0-9a-zA-Z]+' 
                    un_ner = re.findall(pattern,ner[1])
                    #print(un_ner)
                    jiaa = un_ner[1:]
            
                    cond = [''.join(jiaa)]
                    cons = re.findall(yang,cond[0])
                    endi = ['་'.join(cons)]
                    #print(endi)
                    accd = [un_ner[0]]+endi
                    #print(accd)
                    adlist = new_list + accd
                    #print(adlist)
                    ner = adlist[0].split('་')
                    un_ner = adlist[2].split('་')
                  
            for i in range(len(ner)):
                if i == 0:
                    if len(ner)==1:
                        data.append(ner[i]+'\tS-'+adlist[1]) 
                    else:
                        data.append(ner[i]+'\tB-'+adlist[1])
                elif i == len(ner)-1:
                    data.append(ner[i]+'\tE-'+adlist[1])
                else:
                    data.append(ner[i]+'\tI-'+adlist[1])
            for i in un_ner:
                    if i != '':
                        data.append(i+'\tO')


for i in data:
     print(i)
with open('ner.txt', 'w', encoding='utf-8') as output_file:
    for item in data:
        output_file.write('{}\t\n'.format(item))

