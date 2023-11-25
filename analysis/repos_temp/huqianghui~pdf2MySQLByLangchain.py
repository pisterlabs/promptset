'''You are an assistant designed to extract entities from text. Users will paste in a string of text and you will respond with entities you've extracted from the text as a JSON object.
Here's your output format:
{sample}
''''''
{
  "限额项目": "",
  "销售方式": "",
  "是否含申购费": "",
  "金额数": "",
  "单位": ""
}
''''''
{{
	"限额项目": "申购最低额",
	"销售方式": "直销中心柜台",
	"是否含申购费": "含",
	"金额数": "10万",
	"单位": "元"
}}
''''''
{{
	"限额项目": "追加申购最低额",
	"销售方式": "直销中心柜台",
	"是否含申购费": "含",
	"金额数": "10万",
	"单位": "元"
}}
''''''
{{
	"限额项目": "申购最低额",
	"销售方式": "网上直销系统",
	"是否含申购费": "含",
	"金额数": "10",
	"单位": "元"
}}
''''''
{{
	"限额项目": "追加申购最低额",
	"销售方式": "网上直销系统",
	"是否含申购费": "含",
	"金额数": "10",
	"单位": "元"
}}
''''''
{{
	"限额项目": "申购最低额",
	"销售方式": "其他销售机构",
	"是否含申购费": "含",
	"金额数": "0.1",
	"单位": "元"
}}
''''''
{{
	"限额项目": "追加申购最低额",
	"销售方式": "其他销售机构",
	"是否含申购费": "含",
	"金额数": "0.1",
	"单位": "元"
}}
'''f'''
[
	{result1},
	{result2},
	{result3},
	{result4},
	{result5},
	{result6}
]
''''''
{{
	"限额项目": "申购最低额",
	"销售方式": "直销中心柜台",
	"是否含申购费": "含",
	"金额数": "10000",
	"单位": "元"
}}
''''''
{{
	"限额项目": "追加申购最低额",
	"销售方式": "直销中心柜台",
	"是否含申购费": "含",
	"金额数": "1000",
	"单位": "元"
}}
''''''
{{
	"限额项目": "申购最低额",
	"销售方式": "电子直销交易系统/其他销售机构",
	"是否含申购费": "含",
	"金额数": "1",
	"单位": "元"
}}
''''''
{{
	"限额项目": "追加申购最低额",
	"销售方式": "电子直销交易系统/其他销售机构",
	"是否含申购费": "含",
	"金额数": "1",
	"单位": "元"
}}
''''''
{{
	"限额项目": "赎回最低额",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1",
	"单位": "份"
}}
''''''
{{
	"限额项目": "账户持有份额下限",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1",
	"单位": "份"
}}
'''f'''
[
	{result1},
	{result2},
	{result3},
	{result4},
	{result5},
	{result6}
]
''''''
{{
	"限额项目": "申购最低额",
	"销售方式": "销售机构/直销中心/网上直销系统",
	"是否含申购费": "含",
	"金额数": "0.01",
	"单位": "元"
}}
''''''
{{
	"限额项目": "追加申购最低额",
	"销售方式": "销售机构/直销中心/网上直销系统",
	"是否含申购费": "含",
	"金额数": "0.01",
	"单位": "元"
}}
'''f'''
[
	{result1},
	{result2}
]

''''''
{{
	"限额项目": "最小申购赎回单位",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1万",
	"单位": "份"
}}
'''f'''
[
	{result1}
]
''''''
{{
	"限额项目": "赎回最低额",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1",
	"单位": "份"
}}
''''''
{{
	"限额项目": "转换最低额",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1",
	"单位": "份"
}}
''''''
{{
	"限额项目": "账户持有份额下限",
	"销售方式": "",
	"是否含申购费": "",
	"金额数": "1",
	"单位": "份"
}}
'''f'''
[
	{result1},
	{result2},
	{result3}
]
'''