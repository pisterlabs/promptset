from OpenAIAuth import Auth0
import sys
import yaml
# 第一个参数是脚本名称，因此需要从第二个参数开始进行处理
email = sys.argv[1]
password = sys.argv[2]
your_web_password = sys.argv[3]
auth = Auth0(email, password)
access_token = auth.get_access_token()


## 将access_token 写入到yml文件中
with open('docker-compose.yml', 'r') as file:
    data = yaml.safe_load(file)
data['services']['app']['environment']['OPENAI_ACCESS_TOKEN']=access_token
data['services']['app']['environment']['AUTH_SECRET_KEY']=your_web_password

# 将更改保存回文件
with open('docker-compose.yml', 'w') as file:
    yaml.safe_dump(data, file,)
