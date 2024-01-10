from langchain.prompts import PromptTemplate
from viewmodel.model import CompanyAnalysis
from utils.prompt_utils import add_analysis_info, add_language, add_call_to_action, add_output_post_constraint
from langchain.schema import AIMessage, HumanMessage, SystemMessage

class ContentGeneratorPrompt:
    def __init__(self):
        self.templates =  PromptTemplate(
                input_variables=["company", "social_media", "tone"],
                template='''
                    I'm looking for some creative ways to promote our {company} through content marketing and engage our target audience.
                    Write a post in {social_media} with {tone} tone style, including hashtag and other additional information that is common on this media.            
                '''
            )

    def get_content_generator_prompt(
            self, 
            companyAnalysis : CompanyAnalysis
    ):
        

        base_prompt = self.templates.format(
            company=companyAnalysis.name,
            social_media=companyAnalysis.social_media,
            tone=companyAnalysis.tone
        )

        analysis_info = add_analysis_info(
            market_analysis=companyAnalysis.market_analysis,
            competitor=companyAnalysis.competitor,
            key_selling_point=companyAnalysis.key_selling_point,
            base_prompt=""
        )

        message = [
            SystemMessage(
                content="You are professional content creator on social network."
            ),
            HumanMessage(
                content=base_prompt
            ),
            SystemMessage(
                content= analysis_info
            ),
        ]

        if companyAnalysis.website is not None:
            cta = add_call_to_action(
                website_url=companyAnalysis.website, 
                base_prompt="")
            message.append(
                SystemMessage(
                    content=cta
                )
            )


        message.append(
            SystemMessage(
                content=add_output_post_constraint()
            )   
        )

        language = add_language(
            language=companyAnalysis.language,
            base_prompt=""
        )

        message.append(
            SystemMessage(
                content=language
            )
        )
        
        return message
