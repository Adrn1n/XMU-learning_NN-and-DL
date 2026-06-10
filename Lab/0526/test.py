import requests
#提交文件格式，文件序号\t类别序号
#1	0
#2	2
#3	4
f = open(r"19720212203789.txt","rb")#提交的结果以自己的学号命名
files = {'file': f}
r = requests.post(url="http://101.34.251.69:5005/detectfile",files=files)#服务器地址不要修改
print(r.text)
