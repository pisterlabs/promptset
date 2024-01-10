from utils import logger
from langchain import LLMChain, PromptTemplate
from kor import create_extraction_chain, Object, Text


class Parser:

    def __init__(self, llm):
        self.llm = llm

    def extract_mini_info(self, text: str):
        """
        extract id, timestamp and title
        """
        patient_id_obj = Object(
            id='patient_id',
            description='人工编制的表示患者身份的号码，姓名、性别等信息不属于身份id',
            attributes=[
                Text(id='id_name', description='号码名称'),
                Text(id='id_value', description='号码值'),
            ],
            many=True
        )
        record_obj = Object(
            id='record',
            description='医疗记录的记录日期时间或者检查日期时间和对应的内容标题',
            attributes=[
                Text(id='record_datetime', description='记录的日期时间，保留原始形式'),
                Text(id='title', description='记录内容标题，每段文本只有一个标题')
            ],
            many=True
        )
        mini_info_schema = Object(
            id="input_note",
            description=(
                "这是一份医疗文档，包含患者id、记录时间和记录内容"
            ),
            attributes=[
                patient_id_obj,
                record_obj
            ],
            many=False,
        )
        chain = create_extraction_chain(self.llm, mini_info_schema, encoder_or_encoder_class='json')
        prompt = chain.prompt.format_prompt(text=text).to_string()
        logger.info(prompt)
        output = chain.predict_and_parse(text=text)
        if len(output['data']) == 0:
            data = chain.prompt.output_parser.encoder.decode(output['raw'])
        else:
            data = output['data']
        return data

    def generate_abstract(self, content):
        """
        generate abstract in 50 chars
        """
        template = '''
        请将下面这段三个反引号内的病历记录总结为50字以内的摘要，需包含重要的异常表现和临床检查结果，不要加入未在原文中提及的内容。
        ```
        {content}
        ```
        '''
        prompt = PromptTemplate(
            input_variables=["content"],
            template=template,
        )
        prompt_str = prompt.format(content=content)
        logger.info(prompt_str)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.run(content=content)
        return output

    def query_lexical_chain(self, timeline, condition):
        """
        query record number by complex condition
        """
        template = '''
        你是个管理电子病历的AI助手，你需要从给你的病历记录中找出符合要求的一条记录，输出其记录编号。
        下面这段三个反引号内的内容为一个病人住院期间的所有病历记录，每条记录包括编号、记录时间、标题和摘要。
        ```
        {timeline}
        ```
        下面这段三个反引号内的描述是和病历记录有关的查询条件，请输出符合要求的记录编号，如果找不到输出-1。
        第一步，先考虑这个记录会出现在哪些记录类型里。
        第二步，判断条件里的时间是否存在于在第一步的结果范围内，如果超出了结果范围，就找不到符合要求的记录，直接输出-1。
        如果第二步通过，第三步，对第一步的结果找出符合时间先后顺序要求的记录。
        不用考虑记录里是否出现准确结果，只需要考虑记录类型和时间关系即可。
        ```
        {condition}
        ```
        需要在输出里单独输出结果编号，封装在<res>标签里。
        '''
        prompt = PromptTemplate(
            input_variables=["timeline", "condition"],
            template=template,
        )
        prompt_str = prompt.format(timeline=timeline, condition=condition)
        logger.info(prompt_str)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.run(timeline=timeline, condition=condition)
        return output

    def query_answer(self, content, question):
        """
        query answer from record
        """
        template = '''
        你是个管理电子病历的AI助手，你需要根据给你的病历记录和查询条件回答正确的数据查询结果。
        下面这段三个反引号内的内容为一段病历记录。
        ```
        {content}
        ```
        下面这段三个反引号内的描述是和病历记录有关的查询条件，请忽略前面的修饰条件，仅保留核心数据项，然后在病历记录里寻找核心数据项对应的结果。
        如果不存在请如实回答，不要回答原始记录以外的内容。
        ```
        {question}
        ```
        '''
        prompt = PromptTemplate(
            input_variables=["content", "question"],
            template=template,
        )
        prompt_str = prompt.format(content=content, question=question)
        logger.info(prompt_str)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.run(content=content, question=question)
        return output
