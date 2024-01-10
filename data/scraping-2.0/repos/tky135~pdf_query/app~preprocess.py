def get_chunks():
    import pypdf
    CHUNK_LIMIT = 600
    sections = [
        "本手册相关的重要信息",
        "敬告用户",
        "联系Lynk&Co领克",
        "事件数据记录系统",
        "远程监控系统",
        "原厂精装附件、选装装备和改装",
        "无线电设备",
        "所有权变更",
        "动力电池回收",
        "车辆报废",
        "隐私告知",
        "远程查询车辆状况",
        "安全检查",
        "车辆装载",
        "前排储物空间",
        "前排储物空间",
        "第二排储物空间",
        "后备厢储物空间",
        "折叠后排座椅",
        "使用手套箱密码保护",
        "后备厢载物",
        "遮物帘",
        "车内打开/关闭尾门",
        "车内打开/关闭尾门",
        "车外打开/关闭尾门",
        "设置尾门开启角度",
        "车辆锁止/解锁状态",
        "使用遥控钥匙解锁和闭锁",
        "使用Lynk&CoApp解锁和闭锁",
        "无钥匙进入系统",
        "车内解锁和闭锁",
        "车内打开/关闭车门",
        "车外打开/关闭车门",
        "主驾座椅迎宾",
        "开门预警系统",
        "防盗系统",
        "开启/关闭防盗系统",
        "调节驾驶员座椅",
        "调节驾驶员座椅",
        "位置记忆功能",
        "位置记忆功能",
        "方向盘介绍",
        "方向盘介绍",
        "调整方向盘",
        "调节外后视镜",
        "调节内后视镜",
        "内后视镜自动防眩目",
        "胎压监测系统",
        "前雨刮和洗涤器",
        "后雨刮和洗涤器",
        "组合仪表",
        "指示灯和警告灯",
        "指示灯和警告灯",
        "查看组合仪表信息",
        "查看组合仪表信息",
        "打开/关闭远近光灯",
        "打开/关闭远近光灯",
        "打开/关闭自动大灯",
        "智能远近光控制系统",
        "大灯随动转向功能",
        "打开/关闭后雾灯",
        "打开/关闭位置灯",
        "打开/关闭转向指示灯",
        "使用阅读灯",
        "使用超车灯",
        "使用危险警告灯",
        "调节背光亮度",
        "调节背光亮度",
        "设置车内氛围灯",
        "使用接近照明灯",
        "使用伴我回家灯",
        "欢迎灯和欢送灯",
        "安全系统",
        "安全带",
        "使用安全带",
        "安全气囊",
        "颈椎撞击保护系统",
        "打开/关闭车窗",
        "打开/关闭全景天窗",
        "调节副驾驶座椅",
        "调节副驾驶座椅",
        "前排座椅加热",
        "前排座椅通风",
        "前排座椅通风",
        "头枕",
        "方向盘加热",
        "方向盘加热",
        "儿童锁",
        "儿童座椅固定装置",
        "推荐的儿童安全座椅规格",
        "语音助手",
        "车载12V电源",
        "车载12V电源",
        "智能设备充电",
        "智能设备充电",
        "智能设备充电",
        "智能设备充电",
        "外接行车记录仪",
        "遮阳板",
        "驾驶须知",
        "点火模式",
        "通过手机APP启动车辆",
        "通过遥控钥匙启动车辆",
        "车辆熄火",
        "换挡",
        "驾驶模式",
        "方向盘助力与驾驶模式联动",
        "抬头显示",
        "能量回收系统",
        "低速行驶提示音",
        "转向助力系统",
        "制动系统",
        "车身稳定控制系统",
        "制动防抱死系统",
        "电子驻车制动（EPB）",
        "自动驻车系统",
        "坡道辅助系统",
        "陡坡缓降系统",
        "加油",
        "车辆排放",
        "涉水驾驶",
        "节能驾驶",
        "冬季驾驶",
        "斜坡驻车",
        "制动防抱死系统",
        "驾驶辅助系统",
        "驾驶辅助系统传感器",
        "驾驶辅助系统传感器",
        "超速报警",
        "最高限速辅助系统",
        "自适应巡航系统",
        "高级智能驾驶",
        "高级智能驾驶",
        "交通标志识别系统",
        "驾驶员状态监测系统",
        "前方交叉路口预警系统",
        "后方横向来车预警系统",
        "车道辅助系统",
        "车道辅助系统",
        "变道辅助系统",
        "前向碰撞减缓系统",
        "后方碰撞预警系统",
        "紧急转向避让辅助系统",
        "生命体检测系统",
        "泊车辅助传感器",
        "泊车辅助传感器",
        "泊车辅助系统",
        "泊车辅助系统",
        "360°全景影像",
        "泊车紧急制动",
        "全自动泊车",
        "遥控泊车",
        "外后视镜倒车自动调节",
        "折叠/展开外后视镜",
        "开启/关闭空调",
        "调节空调温度",
        "调节空调风量",
        "空调模式",
        "调节空调出风方向",
        "主动式座舱清洁系统",
        "香氛系统",
        "空气质量管理系统",
        "空调除霜/除雾",
        "中央显示屏",
        "设置中央显示屏显示状态",
        "设置中央显示屏显示状态",
        "应用程序",
        "多媒体",
        "车辆功能界面",
        "连接设置",
        "系统设置",
        "账户设置",
        "账户设置",
        "检查车辆网络连接状态",
        "操作行车记录仪",
        "查看行车记录仪视频",
        "行车记录仪内存卡",
        "相机",
        "Lynk&CoApp",
        "创建和删除蓝牙钥匙",
        "高压警告标签",
        "混合动力电池",
        "充电安全警告",
        "车载充电设备充电",
        "随车设备快速充电",
        "预约充电",
        "车辆供电",
        "存放车辆",
        "更换遥控钥匙电池",
        "保养和维护动力电池",
        "保养和维护低压蓄电池",
        "新车磨合",
        "更换保险丝",
        "使用诊断工具读取VIN码",
        "打开前机舱盖",
        "检查发动机机油",
        "检查制动液",
        "检查冷却液",
        "添加洗涤液",
        "更换雨刮片",
        "胎压标签",
        "保养轮胎",
        "清洁车辆",
        "保养漆面",
        "车身防腐",
        "保养内饰",
        "保养项目",
        "车辆远程升级（OTA)",
        "车辆检测",
        "处理车辆故障",
        "处理车辆故障",
        "紧急救援",
        "道路救援求助服务指导",
        "应急解锁和锁止车门",
        "应急打开尾门",
        "应急解锁充电枪",
        "牵引车辆",
        "安全背心和三角警示牌",
        "补胎套装",
        "电池电量较低",
        "车辆标识",
        "车辆参数",
        "缩略语和术语"
    ]
    largesections = {'前言': ['本手册相关的重要信息', '敬告用户', '联系Lynk & Co领克', '事件数据记录系统', '远程监控系统', '原厂精装附件、选装装备和改装', '无线电设备', '所有权变更', '动力电池回收', '车辆报废', '隐私告知'], '用车前准备': ['远程查询车辆状况', '安全检查'], '装载货物': ['车辆装载', '前排储物空间', '前排储物空间', '第二排储物空间', '后备厢储物空间', '折叠后排座椅', '使用手套箱密码保护', '后备厢载物', '遮物帘', '车内打开/关闭尾门', '车内打开/关闭尾门', '车外打开/关闭尾门', '设置尾门开启角度', '车辆锁止/解锁状态'], '上车和下车': ['使用遥控钥匙解锁和闭锁', '使用Lynk&CoApp解锁和闭锁', '无钥匙进入系统', '车内解锁和闭锁', '车内打开/关闭车门', '车外打开/关闭车门', '主驾座椅迎宾', '开门预警系统', '防盗系统', '开启/关闭防盗系统'], '驾驶前的准备': ['调节驾驶员座椅', '调节驾驶员座椅', '位置记忆功能', '位置记忆功能', '方向盘介绍', '方向盘介绍', '调整方向盘', '调节外后视镜', '调节内后视镜', '内后视镜自动防眩目', '胎压监测系统', '前雨刮和洗涤器', '后雨刮和洗涤器'], '仪表和灯光': ['组合仪表', '指示灯和警告灯', '指示灯和警告灯', '查看组合仪表信息', '查看组合仪表信息', '打开/关闭远近光灯', '打开/关闭远近光灯', '打开/关闭自动大灯', '智能远近光控制系统', '大灯随动转向功能', '打开/关闭后雾灯', '打开/关闭位置灯', '打开/关闭转向指示灯', '使用阅读灯', '使用超车灯', '使用危险警告灯', '调节背光亮度', '调节背光亮度', '设置车内氛围灯', '使用接近照明灯', '使用伴我回家灯', '欢迎灯和欢送灯'], '安全出行': ['安全系统', '安全带', '使用安全带', '安全气囊', '颈椎撞击保护系统', '打开/关闭车窗', '打开/关闭全景天窗', '调节副驾驶座椅', '调节副驾驶座椅', '前排座椅加热', '前排座椅通风', '前排座椅通风', '头枕', '方向盘加热', '方向盘加热', '儿童锁', '儿童座椅固定装置', '推荐的儿童安全座椅规格', '语音助手', '车载12V电源', '车载12V电源', '智能设备充电', '智能设备充电', '智能设备充电', '智能设备充电', '外接行车记录仪', '遮阳板'], '启动和驾驶': ['驾驶须知', '点火模式', '通过手机APP启动车辆', '通过遥控钥匙启动车辆', '车辆熄火', '换挡', '驾驶模式', '方向盘助力与驾驶模式联动', '抬头显示', '能量回收系统', '低速行驶提示音', '转向助力系统', '制动系统', '车身稳定控制系统', '制动防抱死系统', '电子驻车制动（EPB）', '自动驻车系统', '坡道辅助系统', '陡坡缓降系统', '加油', '车辆排放', '涉水驾驶', '节能驾驶', '冬季驾驶', '斜坡驻车', '制动防抱死系统'], '驾驶辅助': ['驾驶辅助系统', '驾驶辅助系统传感器', '驾驶辅助系统传感器', '超速报警', '最高限速辅助系统', '自适应巡航系统', '高级智能驾驶', '高级智能驾驶', '交通标志识别系统', '驾驶员状态监测系统', '前方交叉路口预警系统', '后方横向来车预警系统', '车道辅助系统', '车道辅助系统', '变道辅助系统', '前向碰撞减缓系统', '后方碰撞预警系统', '紧急转向避让辅助系统', '生命体检测系统'], '泊车': ['泊车辅助传感器', '泊车辅助传感器', '泊车辅助系统', '泊车辅助系统', '360°全景影像', '泊车紧急制动', '全自动泊车', '遥控泊车', '外后视镜倒车自动调节', '折叠/展开外后视镜'], '空调': ['开启/关闭空调', '调节空调温度', '调节空调风量', '空调模式', '调节空调出风方向', '主动式座舱清洁系统', '香氛系统', '空气质量管理系统', '空调除霜/除雾'], '中央显示屏': ['中央显示屏', '设置中央显示屏显示状态', '设置中央显示屏显示状态', '应用程序', '多媒体', '车辆功能界面', '连接设置', '系统设置', '账户设置', '账户设置', '检查车辆网络连接状态', '操作行车记录仪', '查看行车记录仪视频', '行车记录仪内存卡', '相机'], 'Lynk&CoApp': ['Lynk&CoApp', '创建和删除蓝牙钥匙'], '高压系统': ['高压警告标签', '混合动力电池', '充电安全警告', '车载充电设备充电', '随车设备快速充电', '预约充电', '车辆供电'], '保养和维护': ['存放车辆', '更换遥控钥匙电池', '保养和维护动力电池', '保养和维护低压蓄电池', '新车磨合', '更换保险丝', '使用诊断工具读取VIN码', '打开前机舱盖', '检查发动机机油', '检查制动液', '检查冷却液', '添加洗涤液', '更换雨刮片', '胎压标签', '保养轮胎', '清洁车辆', '保养漆面', '车身防腐', '保养内饰', '保养项目'], 'OTA升级': ['车辆远程升级（OTA)'], '紧急情况下': ['车辆检测', '处理车辆故障', '处理车辆故障', '紧急救援', '道路救援求助服务指导', '应急解锁和锁止车门', '应急打开尾门', '应急解锁充电枪', '牵引车辆', '安全背心和三角警示牌', '补胎套装', '电池电量较低']}

    pdf = pypdf.PdfReader("dataset.pdf")
    current_largesection = '前言'
    current_section = None
    log = open("pre_log.txt", "w")
    chunks = []
    chunk_text = []
    for page in pdf.pages:
        print(page.page_number, end="\r")
        display_page_number = page.page_number + 1
        if display_page_number < 8: # skip the contents pages
            continue
        page_text = page.extract_text()
        if not page_text:    # empty page
            continue
        if page_text.startswith("123\n"):   # new large section
            current_largesection = page_text.split("\n")[1]
            if current_largesection == "技术资料":  # for now this is not considered
                chunks.append(chunk_text)
                chunk_text = []
                break
            assert current_largesection in largesections
            current_section = None
            # wrap up last section
            chunks.append(chunk_text)
            chunk_text = []
            continue
        if not page_text.startswith(current_largesection + '\n' + str(display_page_number)):
            # print(page_text)
            raise Exception("unexpected page")
        page_text = page_text[len(current_largesection) + 1 + len(str(display_page_number)):]
        if current_section is None:     # initial condition
            current_section = sections.pop(0)
            # print(sections)
            # print(current_section)
            assert page_text.startswith(current_section + "\n")
            chunk_text.append(current_section + "\n")
            page_text = page_text[len(current_section) + 1:]

        # get content into chunk_text until we reach the next section(or end of page)
        if not page_text:   # empty page except for section header and page number
            continue
        while True:
            line = page_text.split("\n", 1)[0]
            page_text = page_text.split("\n", 1)[1] if len(page_text.split("\n", 1)) > 1 else ""
            if not line and not page_text:
                break
            # print(line, file=log)
            if line == sections[0]: # reached next section
                chunks.append(chunk_text)
                chunk_text = []
                current_section = sections.pop(0)
                assert current_section == line
                chunk_text.append(line + "\n")
                continue
            if line.endswith("。" + sections[0]): # special case where the next section is not preceeded with a new line
                print(line)
                chunk_text.append(line.rsplit("。", 1)[0] + "。\n")
                chunks.append(chunk_text)
                chunk_text = []
                current_section = sections.pop(0)
                assert current_section == line.rsplit("。", 1)[1]
                chunk_text.append(current_section + "\n")
                continue
            # TODO: handle other special cases about text here:
            #
            #
            # normal case
            if line.endswith("。") or line.endswith("：") or line.endswith("！"):
                # check if length in one chunk is higher than limit
                chunk_text.append(line + "\n")
                if len("".join(chunk_text)) > CHUNK_LIMIT:
                    new_chunk_text = []
                    new_chunk_text = chunk_text[len(chunk_text) // 2:].copy()
                    new_chunk_text = [current_section + "\n"] + new_chunk_text
                    chunks.append(chunk_text)
                    chunk_text = new_chunk_text
            else:
                chunk_text.append(line)
            
        # print(sections[0], file=log)
        # print(chunks)
        print(chunks, file=log)
        print(chunk_text, end='\n\n', file=log)
        # raise Exception("break")

        # output_lines.append(line)

    # with open("dataset_upda.txt", "w") as f:
    #     for chunk in chunks:
    #         for line in chunk:
    #             f.write(line)
    #         f.write("\n")
    chunks = ["".join(chunk) for chunk in chunks]
    return chunks

def get_chunks_naive(chunk_size, chunk_overlap):
    import pypdf
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["(?<=。)","(?<=，)", ""], 
        is_separator_regex=True
    )
    pdf = pypdf.PdfReader("dataset.pdf")
    all_text = ""
    for page in pdf.pages:
        all_text += page.extract_text()
    chunks = r_splitter.split_text(all_text)
    return chunks


if __name__ == "__main__":
    get_chunks_naive()