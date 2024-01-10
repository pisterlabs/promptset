import os
import logging
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
from func_timeout import func_timeout, FunctionTimedOut

from backend import config


class ResumeGenerator():
    """
    Resume generator with user query in memory
    """
    def __init__(self, llm, readonlymemory):
        self.verbose = False
        self.llm = llm
        self.readonlymemory = readonlymemory

    def generate_job_summary(self):
        """
        Summarize the job description.
        To address the token limitations, extract the only useful information from the job description, in case the job description is too long.
        """
        job_template = PromptTemplate(
            input_variables=[
                "job",
            ],
            template=self.JOB_SUMMARY_PROMPT,
        )
        job_summary_chain = LLMChain(llm=self.llm,
                                     prompt=job_template,
                                     verbose=self.verbose,
                                     memory=self.readonlymemory)
        self.job_summary = job_summary_chain.run(job=self.job_description)
        logging.info('The job summary is generated.')

    def generate_background_summary(self):
        """
        Summarize/rephrase the background based on the job summary.
        Extract the related and strong background information using the job summary and rephrase it well.
        """
        # load the background from doc file
        loader = Docx2txtLoader(self.background_path)
        background_info = loader.load()

        # split the background
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        background_info_doc = text_splitter.split_documents(background_info)

        rephrase_template = PromptTemplate(
            input_variables=[
                "background",
                "job",
            ],
            template=self.REPHRASE_PROMPT_TEMPLATE,
        )
        rephrase_chain = LLMChain(llm=self.llm,
                                  prompt=rephrase_template,
                                  verbose=self.verbose,
                                  memory=self.readonlymemory)

        # summarize the background
        results = []
        for part in background_info_doc:
            res = rephrase_chain.run(background=part, job=self.job_summary)
            results.append(res)

        self.background_info_generated = '\n\n'.join(results)
        logging.info('The background summary is generated.')

    def generate_latex(self, template, latex, background):
        """
        Generate the latex code based on the given template and background summary.
        Fill the latex template with the selected background information.
        """
        # generate latex code
        reformat_template = PromptTemplate(
            input_variables=[
                "template",
                "background",
            ],
            template=template,
        )
        reformat_chain = LLMChain(llm=self.llm,
                                  prompt=reformat_template,
                                  verbose=self.verbose,
                                  memory=self.readonlymemory)
        res = reformat_chain.run(background=background, template=latex)
        
        # make sure it only contains utf-8 characters
        return bytes(res, 'utf-8').decode('utf-8', 'ignore').strip()
    
    def read_files(self):
        """
        Read all prompt templates and latex code templates
        """
        with open(config.REPHRASE_PROMPT_TEMPLATE_PATH) as f:
            self.REPHRASE_PROMPT_TEMPLATE = f.read()
        with open(config.JOB_SUMMARY_PROMPT_PATH) as f:
            self.JOB_SUMMARY_PROMPT = f.read()

        with open(config.LATEX_PROMPT_TEMPLATE_1_PATH) as f:
            self.LATEX_PROMPT_TEMPLAT_1 = f.read()
        with open(config.LATEX_PROMPT_TEMPLATE_2_PATH) as f:
            self.LATEX_PROMPT_TEMPLAT_2 = f.read()

        with open(config.LATEX_HEADER_PATH) as f:
            self.latex_header = f.read()
        with open(config.LATEX_PART1_PATH) as f:
            self.part1 = f.read()
        with open(config.LATEX_PART2_PATH) as f:
            self.part2 = f.read()
        
        with open(config.BACKGROUND_SECTION_TEMPLATE_1_PATH) as f:
            self.BACKGROUND_SECTION_TEMPLATE_1 = f.read()
        with open(config.BACKGROUND_SECTION_TEMPLATE_2_PATH) as f:
            self.BACKGROUND_SECTION_TEMPLATE_2 = f.read()

    def generate_background_section(self, template):
        """
        Select the background information based on the latex template sections.
        To address the token limitations, further split the generated background according to the sections of the latex template.
        """
        # generate the background section
        background_template = PromptTemplate(
            input_variables=[
                "background",
            ],
            template=template,
        )
        background_section_chain = LLMChain(llm=self.llm,
                                            prompt=background_template,
                                            verbose=self.verbose,
                                            memory=self.readonlymemory)
        background_section = background_section_chain.run(background=self.background_info_generated)

        return background_section

    def run(self, query, output_path=config.output_path, verbose=False):
        """
        run the resume generator
        (The query is just a placeholder)
        """
        self.background_path = config.background_path
        self.job_description = config.job_description
        self.verbose = verbose
        
        self.read_files()

        self.generate_job_summary()
        self.generate_background_summary()

        background_section_1 = self.generate_background_section(self.BACKGROUND_SECTION_TEMPLATE_1)
        background_section_2 = self.generate_background_section(self.BACKGROUND_SECTION_TEMPLATE_2)
        logging.info('The backgroun section is generated.')

        results = []
        results.append(self.generate_latex(self.LATEX_PROMPT_TEMPLAT_1, self.part1, background_section_1))
        results.append(self.generate_latex(self.LATEX_PROMPT_TEMPLAT_2, self.part2, background_section_2))

        # concat the latex results
        generated_resume = self.latex_header + '\n' + \
            results[0] + '\n' + results[1] + '\n\n \end{document}'
        logging.info('The latex code is generated.')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(generated_resume)

        # compile the latex code
        command = f"pdflatex {output_path}"
        def wrapper():
            os.system(command)
        try:
            func_timeout(5, wrapper)
        except FunctionTimedOut:
            print("\n\nLATEX compile failed\n\n")
            self.run('')
        
        # remove the extra files
        filenames = ['resume.aux', 'resume.log', 'resume.out']
        for fn in filenames:
            if os.path.exists(fn):
                os.remove(fn)
