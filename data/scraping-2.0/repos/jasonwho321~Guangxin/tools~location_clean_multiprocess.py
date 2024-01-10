from location_clean import *
import openai
import csv
import pandas as pd
def gpt_clean(input_file='/Users/huzhang/Documents/我的 Tableau Prep 存储库/数据源/输出 2.csv'):
    openai.api_key = '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'

    # 读取城市名
    city_data = []
    with open(input_file, newline='',
              encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            city_data.append(row[0])

    result_df = pd.DataFrame(columns=['city', 'city_name', 'full_name', 'latitude', 'longitude'])
    total_tokens_used = 0
    un_line = ''
    # 批量处理，例如每40个城市发送一次请求
    batch_size = 50
    for i in range(0, len(city_data), batch_size):
        cities_batch = city_data[i:i + batch_size]

        messages = [
            {"role": "system", "content": "我需要你扮演一个地理专家"},

            {
                "role": "user",
                "content": "我会给你一个未清洗的城市或地理名数据，我需要你尽你最大的可能去猜测这个数据指代的，最有可能的城市，保证每一个名称都有一行对应的数据，我需要你以表格的形式，直接给我5列数据：1，我给你的城市名，不要做任何删减，保留所有符号，尤其是+号，2，你猜测的最有可能的城市名，3，包含城市名在内的完美的地址，4，5经纬度信息，以下是需要你直接返回给我的范例：city\tcity_name\tfull_name\tlatitude\tlongitude\nJuan Aldama\tJuan Aldama\tJuan Aldama, Zacatecas, México\t24.2556571\t-103.3069236\nWESTFORD\tWestford\tWestford, Middlesex County, Massachusetts, 01886, United States\t42.5792583\t-71.4378411， 以下是需要你匹配并输出的城市地理数据，不需要你告诉我操作步骤，我只要最终的数据，输出格式按照：Juan Aldama｜Juan Aldama｜Juan Aldama, Zacatecas, México｜24.2556571｜-103.3069236，我需要你清洗的城市名单是Wulkow + OT Hohenbell｜ Mexioe City"
            },
            {
                "role": "assistant",
                "content": "city\tcity_name\tfull_name\tlatitude\tlongitude\nWulkow + OT Hohenbell\tWulkow\tWulkow OT Hohenbellin, Brandenburg, Germany\t52.7402797\t12.1585299\nMexice City\tMexico City\tMexico City, Mexico City, Mexico\t19.4326296\t-99.1331785"
            },
            {
                "role": "user",
                "content": "很好！我需要你继续清洗的城市名单是：" + "｜ ".join(cities_batch)
            },
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=3000  # 根据批量大小调整最大令牌数
        )
        tokens_used = response['usage']['total_tokens']
        total_tokens_used += tokens_used
        print(f'Tokens used in this batch: {tokens_used}')
        print(f'Total tokens used so far: {total_tokens_used}')

        # 从API响应中提取内容
        content = response.choices[0].message['content']
        # 分割内容为各行
        lines = content.split('\n')[1:]  # 排除第一行，因为它是标题

        # 解析每一行并保存到DataFrame
        for line in lines:
            values = line.split('\t')
            if len(values) != 5:
                un_line = un_line+line
                continue
            city, city_name, full_name, latitude, longitude = values

            new_row = pd.DataFrame({
                'city': [city],
                'city_name': [city_name],
                'full_name': [full_name],
                'latitude': [latitude],
                'longitude': [longitude]
            })

            result_df = pd.concat([result_df, new_row], ignore_index=True)

        # 保存结果到CSV
        result_df.to_csv('/Users/huzhang/Documents/我的 Tableau Prep 存储库/数据源/城市_gpt_cleaned.csv', index=False,
                         encoding='utf-8-sig')
        with open('un_line.txt', 'w') as f:
            f.write(un_line)

def load_checkpoint():
    try:
        with open('../backup_data1.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def main():
    # 尝试从pickle文件中加载数据
    df = load_checkpoint()

    # 如果pickle文件不存在，从csv文件中读取数据
    if df is None:
        df = pd.read_csv('/Users/huzhang/Documents/我的 Tableau Prep 存储库/数据源/城市1.csv')
        df['city_name'] = None
        df['full_name'] = None
        df['latitude'] = None
        df['longitude'] = None

    # 筛选出还没有处理的城市
    unfinished_cities = df[pd.isna(df['city_name'])]['city']

    save_frequency = 100  # 设置每处理100条数据就保存一次
    results = []

    # 使用多进程处理未完成的城市
    with Pool(cpu_count()) as pool:
        for i, result in enumerate(
                tqdm(pool.imap(get_location, unfinished_cities), total=unfinished_cities.shape[0])):
            results.append(result)

            # 如果已经处理了100条数据，就保存到pickle文件
            if (i + 1) % save_frequency == 0:
                city_names, full_names, latitudes, longitudes = zip(*results)
                data = {'city_name': city_names,
                        'full_name': full_names,
                        'latitude': latitudes,
                        'longitude': longitudes}
                df_temp = pd.DataFrame(data)
                df = pd.concat([df, df_temp], ignore_index=True)
                with open('../backup_data.pkl', 'wb') as f:
                    pickle.dump(df, f)
                results = []

    # 更新数据框的列
    city_names, full_names, latitudes, longitudes = zip(*results)
    df.loc[df['city'].isin(unfinished_cities), ['city_name', 'full_name', 'latitude', 'longitude']] = list(zip(city_names, full_names, latitudes, longitudes))

    # 再次保存数据到pickle文件（覆盖旧的）
    with open('../backup_data1.pkl', 'wb') as f:
        pickle.dump(df, f)

    # 保存清洗后的数据到csv
    df.to_csv('/Users/huzhang/Documents/我的 Tableau Prep 存储库/数据源/城市1_cleaned.csv', index=False)


if __name__ == '__main__':
    gpt_clean()