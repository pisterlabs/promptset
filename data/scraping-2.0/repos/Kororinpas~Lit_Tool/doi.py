def get_doi(abstract):
  from kor.extraction import create_extraction_chain
  from kor.nodes import Object, Text, Number
  from langchain.chat_models import ChatOpenAI

  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # type: ignore
  schema = Object(
      id="doi",
      description="doi is a digital identifier.It typically starts with 10. followed by a numeric prefix, such as 10.1000/182.",
      attributes=[
          Text(
              id="doi",
              description='doi is a digital identifier. It typically starts with "10." followed by a numeric prefix, such as 10.1000/182.',
              examples=[
                  ('American Economic Journal: Economic Policy 2015, 7(4): 223–242  http://dx.doi.org/10.1257/pol.20130367 223 Water Pollution Progress at Borders: The','http://dx.doi.org/10.1257/pol.20130367'),
                  ('Environment and Development Economics (2020), 1–17 doi:10.1017/S1355770X2000025X EDE RESEARCH ARTICLE Political incentives, Party Congress, and pollution cycle: empirical evidence from China Zhihua Tian,1 and Yanfang Tian2* 1School of Economics, Zhejiang University of Technology, Hangzhou','10.1017/S1355770X2000025X')
                  ],
               many=True
               )
          ],
          many=False
          )
  chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
  output = chain.predict_and_parse(text=abstract.page_content)
  if 'doi' not in output['data']:
    print(f"LLM strategy failed!!{abstract.metadata['source']} Please manually add it!!")
    source = 'None'
    
    return source
    
  else:
    doi = output['data']['doi']['doi'][0]
    if 'doi=' in doi:
      doi = doi.split('doi=')[1]
    return doi
