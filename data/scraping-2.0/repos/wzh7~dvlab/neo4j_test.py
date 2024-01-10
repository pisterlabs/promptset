from py2neo import Graph, Node, Relationship
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import openai


#########################################################################################
#######################              NEO4J               ################################
#########################################################################################

# 连接到Neo4j数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "1234567890"))

# 创建Flask应用
app = Flask(__name__)


# socketio 用于解决数据同步问题
socketio = SocketIO(app)

# 生成知识图谱
def gen_graph(result):

    # 创建一个空的知识图谱
    knowledge_graph = {"nodes": [], "links": []}

    for record in result:
        project = record.get("p")
        category = record.get("c")
        protection_unit = record.get("pu")
        region_or_unit = record.get("ru")
        
        # 添加节点
        if project:
            knowledge_graph["nodes"].append({"id": project["名称"], "label": "Project"})
        if category:
            knowledge_graph["nodes"].append({"id": category["类别名称"], "label": "Category"})
        if protection_unit:
            knowledge_graph["nodes"].append({"id": protection_unit["单位名称"], "label": "ProtectionUnit"})
        if region_or_unit:
            knowledge_graph["nodes"].append({"id": region_or_unit["地区或单位名称"], "label": "RegionOrUnit"})
        
        # 添加关系
        if project and category:
            knowledge_graph["links"].append({"source": project["名称"], "target": category["类别名称"], "label": "BELONGS_TO"})
        if project and protection_unit:
            knowledge_graph["links"].append({"source": project["名称"], "target": protection_unit["单位名称"], "label": "PROTECTED_BY"})
        if project and region_or_unit:
            knowledge_graph["links"].append({"source": project["名称"], "target": region_or_unit["地区或单位名称"], "label": "DECLARED_IN"})

    return knowledge_graph

# 查询不多于20条Project实体及其相关实体和关系
query = """
MATCH (p:Project)-[:BELONGS_TO]->(c:Category),
      (p)-[:PROTECTED_BY]->(pu:ProtectionUnit),
      (p)-[:DECLARED_IN]->(ru:RegionOrUnit)
RETURN p,c,pu LIMIT 40
"""
result = graph.run(query)

# 知识图谱必须是全局变量
global_knowledge_graph = gen_graph(result)

# 创建Flask路由
@app.route('/')
def index():
    return render_template('index.html', knowledge_graph=global_knowledge_graph)


#########################################################################################
#######################              GPT               ##################################
#########################################################################################

# ChatGPT API密钥
openai.api_key = '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'  # 替换为你的API密钥

# prompt
prompt = """
这是我的neo4j知识图谱结构，根据问题，写一个cypher查询语句，
我希望返回的是完整的节点，而不只是节点的名字。
图谱结构：p:Project (序号: row.`项目序号`, 名称: row.`名称`, 
类别: row.`类别`, 公布时间: row.`公布时间`, 类型: row.`类型`, 
申报地区或单位: row.`申报地区或单位`, 描述: row.`描述`)
pu:ProtectionUnit (单位名称: row.`保护单位`)
c:Category (类别名称: row.`类别`)
ru:RegionOrUnit (地区或单位名称: row.`申报地区或单位`)
MERGE (p)-[:BELONGS_TO]->(c)
MERGE (p)-[:PROTECTED_BY]->(pu)
MERGE (p)-[:DECLARED_IN]->(ru)
"""

prompt2 = """
下面是一个neo4j数据库的返回结果，请用30到50字来总结一下
"""

# 聊天消息的POST路由
@app.route('/chatgpt-api', methods=['POST'])
def chatgpt_api():
    # 全局化
    global global_knowledge_graph

    message = request.json.get('message')

    # 调用OpenAI的chat.completion方法来与ChatGPT进行对话 
    # 【第一次】生成cypher语句
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ]
    )

    # 提取ChatGPT的回复
    chatgpt_response = response['choices'][0]['message']['content']
    
    # 打印cypher内容并尝试运行
    try:
        # 打印cypher语句
        print("[cypher]" + chatgpt_response)

        # 用gpt来查询
        result = graph.run(chatgpt_response)

        # 获取新的数据（这里假设你从某个地方获取新数据）
        new_knowledge_graph = gen_graph(result)

        # 更新全局变量
        global_knowledge_graph = new_knowledge_graph

        # 触发 SocketIO 事件通知客户端数据已更新
        socketio.emit('data_updated', new_knowledge_graph, broadcast=True)

        print(global_knowledge_graph)

        # 再问一次
        # 【第二次】生成cypher语句
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt2},
                {"role": "user", "content": str(global_knowledge_graph)}
            ]
        )

        # 提取ChatGPT的回复
        chatgpt_response = response['choices'][0]['message']['content']
    except:
        print("没有检测到cypher语句")

    return jsonify({'message': chatgpt_response})

if __name__ == '__main__':
    app.run(debug=True)
    socketio.run(app, debug=True)
