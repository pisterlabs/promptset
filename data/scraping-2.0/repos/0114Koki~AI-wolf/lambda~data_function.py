import boto3
import credentials
import random
from datetime import datetime
import string
import json
from decimal import Decimal
import openai

table_name = 'game_manager_2'

dynamodb = boto3.resource(
        "dynamodb", 
        region_name='ap-northeast-3',
        aws_access_key_id=credentials.aws_access_key,
        aws_secret_access_key=credentials.aws_secret_key,
        #aws_session_token='YOUR_SESSION_TOKEN
        )
game_manager = dynamodb.Table(table_name)

def make_game_table():
    
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'RoomID',
                'KeyType': 'HASH'  # Partition key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'RoomID',
                'AttributeType': 'N'
            },
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        }
    )

def new_item():
    table = dynamodb.Table(table_name)
    table.put_item(
        Item={
            'RoomID': 419211,
            'Password': "alaksa",
            'N_mem': 4,
            'Members': ["asae", "asda"],
            'N_hacked': 1,
            'Hacked': [],
            'GameState': 0,
            'Created-at': '20230931-09-12-43',
            'Current_mem': len([])
        }
    )
    return  0

# ゲーム管理テーブルの全項目取得
def scan_game_manager():
    response = game_manager.scan()
    items = response['Items']
    return  items

def create_room(n_mem, user_name):
    # ゲーム管理テーブルのスキャン
    items = scan_game_manager()
    
    # ルームIDの発行
    ## 既存のルームIDのリスト化
    existing_room_ids = [item['RoomID'] for item in items]
    # ランダムなルームIDの作成とかぶりの確認
    roomid = random.randint(100000, 999999)
    while (roomid in existing_room_ids):
        roomid = random.randint(100000, 999999)

    # パスワード(６桁string)の発行
    alphabet = string.ascii_lowercase  # 小文字のアルファベット
    password = ''.join(random.choice(alphabet) for _ in range(6))   
    
    # 現在の日時を取得
    current_datetime = datetime.now()
    # 日時を指定したフォーマットに変換
    timestamp = current_datetime.strftime("%Y%m%d-%H-%M-%S")
    
    # その他の属性
    members = [user_name]
    gamestate = 0

    # テーブルへのアイテムの追加
    game_manager.put_item(
        Item = {
            'RoomID': roomid,
            'Password': password,
            'N_mem': n_mem,
            'Members': members,
            'N_hacked': 1,
            'Hacked': [],
            'AI_msgs': {},
            'Dead': [],
            'GameState': gamestate,
            'Created-at': timestamp,
            'Current_mem': len(members),
        }
    )
    print("Room Created| roomid:", roomid, ", password:", password)
    response = {
        'statusCode': 200,
        "body": json.dumps({'message': "OK", "roomid": roomid, "password": password}),
        'headers': {'Access-Control-Allow-Origin': '*'}
    }
    return response


def get_item(roomid):
    table = dynamodb.Table(table_name)
    response = table.get_item(
        Key={
            'RoomID': roomid,
        }
    )
    item = response['Item']
    #print(item)
    return item

# 数値をDecimalからintに変換する関数
def convert_decimal_to_int(value):
    if isinstance(value, Decimal):
        return int(value)
    return value

def join_room(roomid, password, user_name):
    # ゲーム管理テーブルのスキャン
    items = scan_game_manager()

    # ルームIDの存在確認
    existing_room_ids = [item['RoomID'] for item in items]
    if roomid not in existing_room_ids:
        print("RoomID", roomid, "is not found")
        response = {
            'statusCode': 404,
            "body": json.dumps({'message': "RoomID is not found"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # 該当ルーム情報の取得
    room_info = [item for item in items if item['RoomID'] == roomid][0]
    # パスワードの確認
    if room_info['Password'] != password:
        print("Password is incorrect")
        response = {
            'statusCode': 401,
            "body": json.dumps({'message': "Password is incorrect"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # GameStateの確認
    if room_info['GameState'] != 0:
        print("Room is not available")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "Room is not available"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # メンバー数の確認
    if room_info['Current_mem'] >= room_info['N_mem']:
        print("Room is full")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "Room is full"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # 名前の重複確認
    if user_name in room_info['Members']:
        print("User name is already used")
        response = {
            'statusCode': 409,
            "body": json.dumps({'message': "User name is already used"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # メンバーの追加
    members = room_info['Members']
    members.append(user_name)

    # データベースの更新
    game_manager.update_item(
        Key={
            'RoomID': roomid,
        },
        UpdateExpression='SET Members = :val1, Current_mem = :val2',
        ExpressionAttributeValues={
            ':val1': members,
            ':val2': len(members),
        }
    )
    print(user_name, "joined", roomid)
    response = {
        'statusCode': 200,
        "body": json.dumps({'message': "OK"}),
        'headers': {'Access-Control-Allow-Origin': '*'}
    }
    return response
    
def close_room(roomid, owner_name):
    # 該当ルーム情報の取得
    room_info = get_item(roomid)
    # オーナー名の確認
    if room_info['Members'][0] != owner_name:
        print("You are not owner")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "You are not owner"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    
    # GameStateの更新(3: 取り消し)
    game_manager.update_item(
        Key={
            'RoomID': roomid,
        },
        UpdateExpression='SET GameState = :val1',
        ExpressionAttributeValues={
            ':val1': 3,
        }
    )
    print("Room", roomid, "is closed by owner")
    response = {
        'statusCode': 200,
        "body": json.dumps({'message': "OK"}),
        'headers': {'Access-Control-Allow-Origin': '*'}
    }
    return response

def leave_room(roomid, user_name):
    # 該当ルーム情報の取得
    room_info = get_item(roomid)
    # メンバーの確認
    if user_name not in room_info['Members']:
        print("User name is not found in the room")
        response = {
            'statusCode': 404,
            "body": json.dumps({'message': "User name is not found"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # オーナーでないことの確認
    if user_name == room_info['Members'][0]:
        print("You are owner, so you cannot leave the room")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "You are owner, so you cannot leave the room"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    
    # GameStateの確認
    if room_info['GameState'] != 0:
        print("Room is not in Waiting Mode")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "Room is not in Waiting Mode"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # メンバーの削除
    members = room_info['Members']
    members.remove(user_name)
    # データベースの更新
    game_manager.update_item(
        Key={
            'RoomID': roomid,
        },
        UpdateExpression='SET Members = :val1, Current_mem = :val2',
        ExpressionAttributeValues={
            ':val1': members,
            ':val2': len(members),
        }
    )
    print(user_name, "left room", roomid)
    response = {
        'statusCode': 200,
        "body": json.dumps({'message': "OK"}),
        'headers': {'Access-Control-Allow-Origin': '*'}
    }
    return response


def start_game(roomid, owner_name, n_hacked):
    # 部屋の情報を取得
    room_info = get_item(roomid)
    # オーナー名の確認
    if room_info['Members'][0] != owner_name:
        print("You are not owner")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "You are not owner"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # GameStateの確認
    if room_info['GameState'] != 0:
        print("Room is not in Waiting Mode")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "Room is not in Waiting Mode"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # メンバー数の確認
    if room_info['Current_mem'] != room_info['N_mem']:
        print("Room is not full")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "Room is not full"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # n_hackedの数の確認
    if n_hacked >= room_info['N_mem']/2 or n_hacked <= 0:
        print("n_hacked is out of range")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "n_hacked is out of range"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    
    # Hackedの選択
    members = room_info['Members']
    hacked = random.sample(members, n_hacked)

    # HackedにAIを割り当て
    AI_msgs = {}
    for person in hacked:
        message = []
        # 年齢・性別のランダム作成
        # 代のリストと性別のリストを定義します
        generations = ["10代前半", "10代後半", "20代前半", "20代後半", "30代", "40代", "50代", "60代",  "90代"]
        genders = ["男", "女"]
        # ランダムに代と性別を選択します
        random_generation = random.choice(generations)
        random_gender = random.choice(genders)
        # 文字列を生成します
        result = f"{random_generation}{random_gender}性"

        # 生成した人格をAIに付与
        message.append({
            "role": "system",
            "content": f"あなたは{result}で、友達とアイスブレイクゲームをしています。簡単なクイズに対して、簡潔に答えます。必ずしも文章で答える必要はありません。敬語は禁止で、できれば体現止めで答えてください。'答え：'などの形式も不要です。AIであると気づかれないように回答してください。です、ますは禁止です。"
        },)
        AI_msgs[person] = message

    # GameStateの更新(1: ゲーム中)
    game_manager.update_item(
        Key={
            'RoomID': roomid,
        },
        UpdateExpression='SET GameState = :val1, Hacked = :val2, AI_msgs = :val3',
        ExpressionAttributeValues={
            ':val1': 1,
            ':val2': hacked,
            ':val3': AI_msgs,
        }
    )
    print("Game started in room", roomid, "| Hacked:" ,hacked)
    response = {
        'statusCode': 200,
        "body": json.dumps({'message': "OK", "hacked": hacked}),
        'headers': {'Access-Control-Allow-Origin': '*'}
    }
    return response


# Deadの追加
def add_dead(roomid, user_name):
    # 該当ルーム情報の取得
    room_info = get_item(roomid)
    # メンバーの確認
    if user_name not in room_info['Members']:
        print("User name is not found in the room")
        response = {
            'statusCode': 404,
            "body": json.dumps({'message': "User name is not found in the room"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # GameStateの確認
    if room_info['GameState'] != 1:
        print("Room is not in Game Mode")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "Room is not in Game Mode"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # メンバーをDeadに追加
    dead = room_info['Dead']
    dead.append(user_name)
    # データベースの更新
    game_manager.update_item(
        Key={
            'RoomID': roomid,
        },
        UpdateExpression='SET Dead = :val1',
        ExpressionAttributeValues={
            ':val1': dead,
        }
    )
    print(user_name, "is dead in room", roomid)
    response = {
        'statusCode': 200,
        "body": json.dumps({'message': "OK"}),
        'headers': {'Access-Control-Allow-Origin': '*'}
    }
    return response




# ルーム情報の取得
def get_room_info(roomid):
    # 該当ルーム情報の取得
    room_info = get_item(roomid) # 数値がDecimalで入ってる
    converted = {key: convert_decimal_to_int(value) for key, value in room_info.items()} # Decimalをintに変換

    # 送信
    response = {
        'statusCode': 200,
        "body": json.dumps({'message': "OK", "room_info": converted}),
        'headers': {'Access-Control-Allow-Origin': '*'}
    }
    return response

    
# ゲーム終了をDBに反映
def end_game(roomid):
    # 該当ルーム情報の取得
    room_info = get_item(roomid)
    
    # GameStateの確認
    if room_info['GameState'] != 1:
        print("Room is not in Game Mode")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "Room is not in Game Mode"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # GameStateの更新(2: ゲーム終了)
    game_manager.update_item(
        Key={
            'RoomID': roomid,
        },
        UpdateExpression='SET GameState = :val1',
        ExpressionAttributeValues={
            ':val1': 2,
        }
    )
    print("Game ended in room", roomid)
    response = {
        'statusCode': 200,
        "body": json.dumps({'message': "OK"}),
        'headers': {'Access-Control-Allow-Origin': '*'}
    }
    return response


# ChatGPTに問い合わせ
def chatgpt(messages):
    openai.api_key = credentials.chatgpt_secret_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return(response['choices'][0]['message']['content'])

# 回答の生成
def get_ai_answer(roomid, user_name, question):
    # 該当ルーム情報の取得
    room_info = get_item(roomid)
    # Hackedであることの確認
    if user_name not in room_info['Hacked']:
        print("User name is not found in Hacked list")
        response = {
            'statusCode': 404,
            "body": json.dumps({'message': "User name is not found in Hacked list"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # GameStateの確認
    if room_info['GameState'] != 1:
        print("Room is not in Game Mode")
        response = {
            'statusCode': 403,
            "body": json.dumps({'message': "Room is not in Game Mode"}),
            'headers': {'Access-Control-Allow-Origin': '*'}
        }
        return response
    # 対象のmessagesの取得と質問の追加
    messages = room_info["AI_msgs"][user_name]
    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # ChatGPTへの問い合わせ
    ans = chatgpt(messages)

    # 回答のmsgsへの追加
    messages.append(
        {
            "role": "assistant",
            "content": ans
        }
    )
    updated_AI_msgs = room_info["AI_msgs"]
    updated_AI_msgs[user_name] = messages

    # Databeseへの保存
    game_manager.update_item(
        Key={
            'RoomID': roomid,
        },
        UpdateExpression='SET AI_msgs = :val1',
        ExpressionAttributeValues={
            ':val1': updated_AI_msgs,
        }
    )
    print("AI answered in room", roomid, "| user:", user_name, ", question:", question, ", answer:", ans)
    response = {
        'statusCode': 200,
        "body": json.dumps({'message': "OK", "answer": ans}),
        'headers': {'Access-Control-Allow-Origin': '*'}
    }
    return response

