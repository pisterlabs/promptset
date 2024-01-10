
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table
from reportlab.lib import colors
import matplotlib.colors as mcolors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Table, Spacer, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PyPDF2 import PdfMerger, PdfReader
from reportlab.lib.utils import ImageReader
import datetime
import os
import re
import markdown
import variables

from report_fig import prepare_disease_data
from report_text import openai_single, bing_analysis, update_markdown_file, openai_abstract

def translate_html(element, styles, text):
    """
    This function is used to trans markdown to reportlab supported html.
    """
    # split text by \n
    contents = text.split("\n")
    for content in contents:
        if content.startswith("#"):
            # remove all # and following spaces
            content = content.replace("#", "").strip()
            element.append(Spacer(1, 12))
            element.append(Paragraph(content, styles['Hed1']))
            element.append(Spacer(1, 12))
        else:
            content = markdown.markdown(content)
            element.append(Paragraph(content, styles['Normal']))
    return element

def create_report(disease_order):
    """
    This function is used to merge created pdf files into one pdf file.

    Parameters:
    - disease_order (list): the order of diseases
    """

    filenames = [f"./temp/{disease}.pdf" for disease in disease_order]
    merger = PdfMerger()
    merger.append("./temp/cover.pdf")
    merger.append("./temp/cover_summary.pdf")
    # Loop through all the filenames and merge them
    for filename in filenames:
        try:
            # Open the PDF file and append it to the merger
            with open(filename, 'rb') as pdf_file:
                merger.append(pdf_file)
        except FileNotFoundError:
            print(f"File {filename} not found. Skipping.")
        except Exception as e:
            print(f"An error occurred while merging {filename}: {e}")

    # Write out the merged PDF to a new file
    output_filename = "./temp/Report.pdf"
    with open(output_filename, 'wb') as output_pdf:
        merger.write(output_pdf)

    merger.close()
    return output_filename

def create_report_page(df,
                       disease_name,
                       analysis_YearMonth,
                       report_date,
                       page_number,
                       page_total):
    """
    This function is used to process disease data.

    Parameters:
    - df: the dataframe of disease data
    - disease_name: the name of disease
    - 
    - report_date: the date of report, format: "June 2023"
    - page_number: the number of page
    - page_total: the total number of page
    - foot_left_content: the content of left footer, format: "Page 1 of 1"
    """

    links_app = variables.links_app
    links_web = variables.links_web

    # prepare data
    disease_data = prepare_disease_data(df, disease_name)
    table_data_str = disease_data[['YearMonth', 'Cases', 'Deaths']].to_markdown(index=False)

    
    pdfmetrics.registerFont(TTFont('Helvetica', './WeeklyReport/font/Helvetica.ttf'))
    pdfmetrics.registerFont(TTFont('Helvetica-Bold', './WeeklyReport/font/Helvetica-Bold.ttf'))

    introduction_box_content = openai_single(
        os.environ['REPORT_INTRODUCTION_CREATE'],
        os.environ['REPORT_INTRODUCTION_CHECK'],
        variables.introduction_create.format(disease_name=disease_name), 
        variables.introduction_check.format(disease_name=disease_name),
        variables.introduction_words,
        "Introduction",
        disease_name
        )
    update_markdown_file(disease_name, "Introduction", introduction_box_content, analysis_YearMonth)

    highlights_box_content = openai_single(
        os.environ['REPORT_HIGHLIGHTS_CREATE'],
        os.environ['REPORT_HIGHLIGHTS_CHECK'],
        variables.highlights_create.format(disease_name=disease_name, report_date=report_date, table_data_str=table_data_str),
        variables.highlights_check.format(disease_name=disease_name),
        variables.highlights_words,
        "Highlights",
        disease_name
        )
    update_markdown_file(disease_name, "Highlights", highlights_box_content, analysis_YearMonth)

    analy_box_content = openai_single(
        os.environ['REPORT_ANALYSIS_CREATE'],
        os.environ['REPORT_ANALYSIS_CHECK'],
        variables.analysis_create.format(disease_name=disease_name, table_data_str=table_data_str),
        variables.analysis_check.format(disease_name=disease_name),
        variables.analysis_words,
        "Analysis",
        disease_name
        )
    update_markdown_file(disease_name, "Analysis", analy_box_content, analysis_YearMonth)
    cases_box_content = analy_box_content.split("### Cases Analysis")[1].split("### Deaths Analysis")[0]
    death_box_content = analy_box_content.split("### Deaths Analysis")[1]
    # remove all \n
    cases_box_content = cases_box_content.replace("\n", "")
    death_box_content = death_box_content.replace("\n", "")

    foot_left_content = f"Page {page_number} of {page_total}"
    info_box_content = '<font color="red"><b>' + variables.alert_content + '</b></font>'
    foot_right_content = ""
    set_report_title = variables.cover_title_1 + " " + variables.cover_title_2

    add_disease(disease_name,
                report_date, 
                introduction_box_content,
                highlights_box_content,
                cases_box_content,
                death_box_content,
                foot_left_content,
                links_app,
                links_web,
                info_box_content,
                foot_right_content,
                set_report_title)

def add_disease(set_disease_name,
                set_report_date, 
                introduction_box_content,
                highlights_box_content,
                incidence_box_content,
                death_box_content,
                foot_left_content,
                links_app,
                links_web,
                info_box_content, 
                foot_right_content,
                set_report_title):
    # setting function description
    """
    This function is used to generate pdf report page for single disease.

    Parameters:
    - set_disease_name (str): the name of disease
    - set_report_date (str): the date of report, format: "June 2023"
    - introduction_box_content (str): the content of introduction box, generated by AI
    - highlights_box_content (str): the content of highlights box, generated by AI
    - incidence_box_content (str): the content of incidence box, generated by AI
    - death_box_content (str): the content of death box, generated by AI
    - foot_left_content (str): the content of left footer, format: "Page 1 of 1"
    - links_app (str): the link of app
    - links_web (str): the link of web
    - info_box_content (str): the content of informaton box, wanring users that the content of boxs is generated automatically by AI
    """

    # create pdf
    pdf_filename =  f"./temp/{set_disease_name}.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=A4)

    # define page size
    page_width, page_height = A4

    # define title box
    title_box_x = 20
    title_box_y = 730
    title_box_space = 5
    title_box_width = page_width - title_box_x * 2
    title_box_height = 80
    title_box_color = colors.HexColor("#1E5E84")
    title_font_sizes = [18, 16, 12]
    title_line_spacing = [1.5, 1, 1]
    title_font_family = ['Helvetica-Bold', 'Helvetica', 'Helvetica']
    title_text_color = colors.white

    # draw title box
    c.setFillColor(title_box_color)
    c.rect(title_box_x, title_box_y, title_box_width, title_box_height, fill=True, stroke=False)
    title_text_x = title_box_x + title_box_space * 2
    title_text_y = title_box_y + title_box_height - 25
    title_lines = [
        set_report_title,
        set_disease_name,
        set_report_date
    ]
    for i, line in enumerate(title_lines):
        c.setFont(title_font_family[i], title_font_sizes[i])
        c.setFillColor(title_text_color)
        c.drawString(title_text_x, title_text_y, line)
        if i < len(title_lines) - 1:
            title_text_y -= title_font_sizes[i] * title_line_spacing[i] + 2
        else:
            break

    # define content box
    content_box_title_color = colors.HexColor("#2E2F5C")

    introduction_box_height = 120
    introduction_box_y = title_box_y - introduction_box_height - title_box_space * 3
    introduction_box_width = (title_box_width - title_box_space * 2) * 2 / 3

    temporal_trend_box_height = 195
    temporal_trend_box_y = introduction_box_y - temporal_trend_box_height - title_box_space * 2
    temporal_trend_box_width = introduction_box_width

    highlights_box_height = introduction_box_height + temporal_trend_box_height + title_box_space*2
    highlights_box_y = temporal_trend_box_y
    highlights_box_width = (title_box_width - title_box_space*2) * 1 / 3

    incidence_box_width = (title_box_width - title_box_space*2) * 1 / 2
    incidence_box_height = 165
    incidence_box_y = temporal_trend_box_y - incidence_box_height - title_box_space * 2

    month_box_width = (title_box_width - title_box_space*2) * 3 / 4
    month_box_height = 150
    month_box_y = incidence_box_y - month_box_height - title_box_space * 2

    info_box_height = 30

    # draw content box
    box_positions = [
        (title_box_x, introduction_box_y, introduction_box_width, introduction_box_height, "#E6E6E6", introduction_box_content),
        (title_box_x, temporal_trend_box_y, temporal_trend_box_width, temporal_trend_box_height, "#E6E6E6", "figure1"),
        (title_box_x + introduction_box_width + title_box_space*2, highlights_box_y, highlights_box_width, highlights_box_height, "#E6E6E6", highlights_box_content),
        (title_box_x, incidence_box_y, incidence_box_width, incidence_box_height, "#E6E6E6", incidence_box_content),
        (title_box_x + incidence_box_width + title_box_space*2, incidence_box_y, incidence_box_width, incidence_box_height, "#E6E6E6", death_box_content),
        (title_box_x , month_box_y, month_box_width, month_box_height, "#E6E6E6", "figure2"),
        (title_box_x+ month_box_width + title_box_space*2, month_box_y, month_box_width/3, month_box_height, "#D8E6E8", 'figure3'),
        (title_box_x, month_box_y - info_box_height - title_box_space*2, title_box_width, info_box_height, "#E6E6E6", info_box_content)
    ]
    box_titles = [
        "Introduction", 
        "Temporal Trend",
        "Highlights",
        "Cases Analysis",
        "Deaths Analysis",
        "Distribution",
        "",
        None
    ]
    box_links =[
        links_app,
        links_app,
        links_app,
        links_app,
        links_app,
        links_app,
        links_web,
        links_web
    ]
    styles = getSampleStyleSheet()

    for i, (x, y, width, height, color, content) in enumerate(box_positions):
        c.setFillColor(colors.HexColor(color))
        c.rect(x, y, width, height, fill=True, stroke=False)

        if box_titles[i] is None:
            para = Paragraph(content, styles['Normal'])
            para.wrapOn(c, width - 10, height - 10)
            para.drawOn(c, x + 10, y + height - para.height - 8)
        else:
            c.setFont(title_font_family[0], 14)
            c.setFillColor(content_box_title_color)
            c.drawString(x + title_box_space*2, y + height - 15, box_titles[i])

        if content.startswith('figure'):
            if content == 'figure3':
                image_path = f'./{content}.png'
            else:
                image_path = f'temp/{set_disease_name} {content}.png'
            image = ImageReader(image_path)
            iw, ih = image.getSize()
            scale_w = (width - 20) / iw
            scale_h = (height - 30) / ih
            scale = min(scale_w, scale_h)
            new_width = iw * scale
            new_height = ih * scale
            new_x = x + 10 + (width - 20 - new_width) / 2
            new_y = y + 10 + (height - 30 - new_height) / 2
            c.drawImage(image_path, new_x, new_y, new_width, new_height, mask='auto')
            c.linkURL(url=box_links[i], rect=(x + 10, y + 10, x + 10 + width - 20, y + 10 + height - 30))
        elif box_titles[i] is not None:
            content = content_clean(content)
            para = Paragraph(content, styles['Normal'])
            para.wrapOn(c, width - 15, height - 30)
            para.drawOn(c, x + 10, y + height - para.height - 20)

    # define copy right
    copy_right_font_size = 8
    copy_right_font_family = 'Helvetica'
    copy_right_text_color = colors.black

    ## draw copy right
    c.setFont(copy_right_font_family, copy_right_font_size)
    c.setFillColor(copy_right_text_color)
    foot_right_content_width = c.stringWidth(foot_right_content, copy_right_font_family, copy_right_font_size)
    c.drawString(title_box_x + month_box_width + title_box_space*2 + month_box_width/6 - foot_right_content_width/2,
                month_box_y + month_box_height/4,
                foot_right_content)

    current_time = datetime.datetime.now()
    version_content = f"Version: {current_time.strftime('%Y-%m-%d')} ({current_time.astimezone().strftime('%Z%z').replace('00','')})"
    version_content_width = c.stringWidth(version_content, copy_right_font_family, copy_right_font_size)
    c.drawString(title_box_x + month_box_width + title_box_space*2 + month_box_width/6 - version_content_width/2,
                month_box_y + month_box_height/4 - 15,
                version_content)

    # define page number
    foot_left_y = 8
    foot_font_size = 8
    foot_font_family = 'Helvetica'
    foot_text_color = colors.HexColor("#606060")

    # draw page number
    c.setFont(foot_font_family, foot_font_size)
    c.setFillColor(foot_text_color)
    c.drawString(title_box_x, foot_left_y, foot_left_content, )

    # save pdf
    c.showPage()
    c.save()

    print(f"{set_disease_name} report is generated successfully!")

def create_report_cover(analysis_MonthYear):
    """
    This function is used to generate pdf report cover.

    Parameters:
    - analysis_MonthYear (str): the date of report, format: "June 2023"
    """

    # create pdf
    pdf_filename =  f"./temp/cover.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=A4)

    # define page size
    page_width, page_height = A4
    styles = getSampleStyleSheet()

    # add background image
    image_path = './temp/cover.jpg'
    image = ImageReader(image_path)
    iw, ih = image.getSize()
    scale_w = page_width / iw
    scale_h = page_height / ih
    scale = max(scale_w, scale_h)
    new_width = iw * scale
    new_height = ih * scale
    new_x = (page_width - new_width) / 2
    new_y = (page_height - new_height) / 2
    c.drawImage(image_path, new_x, new_y, new_width, new_height, mask='auto')

    # define project location
    project_box_x = 20
    project_box_y = page_height - 50
    project_text_color = colors.white
    project_font_sizes = 10
    project_font_family = 'Helvetica-Bold'

    # dram project
    c.setFont(project_font_family, project_font_sizes)
    c.setFillColor(project_text_color)
    c.drawString(project_box_x, project_box_y, variables.cover_project)

    # define title location
    title_box_x = 20
    title_box_y = 650
    title_text_color = colors.white
    title_font_sizes = 28
    title_font_family = 'Helvetica-Bold'

    # draw title
    text = [variables.cover_title_1, 
            variables.cover_title_2,
            analysis_MonthYear]
    position = [(title_box_x, title_box_y),
                (title_box_x, title_box_y - 40),
                (title_box_x, title_box_y - 80)]
    c.setFont(title_font_family, title_font_sizes)
    c.setFillColor(title_text_color)
    for i, line in enumerate(text):
        c.drawString(position[i][0], position[i][1], line)

    # define author location
    author_box_x = 20
    author_box_y = 90
    author_text_color = colors.white
    author_font_sizes = 10
    author_font_family = 'Helvetica'

    # draw author
    date_now = datetime.datetime.now()
    date_now = date_now.strftime("%Y-%m-%d")
    text = [variables.cover_info_1,
            variables.cover_info_2,
            variables.cover_info_3,
            variables.cover_info_4.format(date_now=date_now),
            variables.cover_info_5]
    styles.add(ParagraphStyle(name='covernormal', parent=styles['Normal'],
                              fontSize=author_font_sizes, textColor=author_text_color, fontName=author_font_family))
    for i, line in enumerate(text):
        para = Paragraph(line, styles["covernormal"])
        para.wrapOn(c, page_width - 40, 20)
        para.drawOn(c, author_box_x, author_box_y - i * 15)

    # save pdf
    c.showPage()
    c.save()

# create_report_cover("September 2023")

def create_report_summary(table_data, table_data_str, analysis_MonthYear, legend, file_name = "cover_summary"):

    elements = []
    analysis_MonthYear = analysis_MonthYear

    # setting style
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Subtitle', parent=styles['Normal'], fontSize=12, textColor=colors.gray))
    styles.add(ParagraphStyle(name='Notice', parent=styles['Normal'], fontSize=14, textColor=colors.red, alignment=TA_CENTER, borderWidth=3))
    styles.add(ParagraphStyle(name="Cite", alignment=TA_LEFT, fontSize=10, textColor=colors.gray))
    styles.add(ParagraphStyle(name="Author", alignment=TA_LEFT, fontSize=10, textColor=colors.black))
    styles.add(ParagraphStyle(name='Hed1', alignment=TA_LEFT, fontSize=14, fontName='Helvetica-Bold', textColor=colors.black))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER, fontName='Helvetica-Bold', fontSize=14, textColor=colors.white))
    styles.add(ParagraphStyle(name='Hed0', fontSize=16, alignment=TA_LEFT, borderWidth=3, textColor=colors.HexColor("#1E5E84")))
    styles.add(ParagraphStyle(name='foot', fontName='Helvetica', fontSize=10, textColor=colors.black))
    styles.add(ParagraphStyle(name='pg', fontName='Helvetica', fontSize=8, textColor=colors.HexColor("#606060")))
    styles.add(ParagraphStyle(name='break', fontName='Helvetica', fontSize=10, textColor=colors.darkred))
    styles.add(ParagraphStyle(name='TOC', fontName='Helvetica-Bold', fontSize=10, textColor=colors.navy, alignment=TA_LEFT, leading=20))
    
    pdfmetrics.registerFont(TTFont('Helvetica', './WeeklyReport/font/Helvetica.ttf'))
    pdfmetrics.registerFont(TTFont('Helvetica-Bold', './WeeklyReport/font/Helvetica-Bold.ttf'))


    # add table of content
    disease_order = table_data['Diseases'][:-1].tolist()
    elements = add_toc(elements, disease_order, styles)
    # add table
    elements = add_table(elements, table_data, analysis_MonthYear, styles)
    # add monthly analysis
    analysis_content = openai_abstract(os.environ['REPORT_ABSTRACT_CREATE'],
                                       os.environ['REPORT_ABSTRACT_CHECK'],
                                       variables.abstract_create.format(analysis_MonthYear=analysis_MonthYear, table_data_str=table_data_str, legend=legend),
                                       variables.abstract_check,
                                       4096)
    elements = add_analysis(elements, analysis_content, styles)

    # update README.md
    with open('../docs/README.md', 'r') as file:
        readme = file.read()
    pattern = r"(# Introduction\n(?:.*?(?=\n# |\Z))*)"
    replacement = f"# Introduction\n\n{analysis_content}\n\n"
    readme_new = re.sub(pattern, replacement, readme, flags=re.DOTALL)
    with open('../docs/README.md', 'w') as file:
        file.writelines(readme_new)

    # add table legend
    elements = add_legend(elements, legend, styles)
    # add new page
    bing_content = bing_analysis(os.environ['REPORT_NEWS_CREATE'],
                                 os.environ['REPORT_NEWS_CLEAN'],
                                 os.environ['REPORT_NEWS_CHECK'],
                                 variables.news_create_nation.format(analysis_MonthYear=analysis_MonthYear),
                                 variables.news_clean_nation,
                                 variables.news_check_nation)
    elements = add_news(elements, bing_content, analysis_MonthYear, "in Chinese Mainland", styles)

    bing_content = bing_analysis(os.environ['REPORT_NEWS_CREATE'],
                                 os.environ['REPORT_NEWS_CLEAN'],
                                 os.environ['REPORT_NEWS_CHECK'],
                                 variables.news_create_global.format(analysis_MonthYear=analysis_MonthYear),
                                 variables.news_clean_global,
                                 variables.news_check_global)
    elements = add_news(elements, bing_content, analysis_MonthYear, "around world", styles)

    # pre build
    doc = SimpleDocTemplate(f"./temp/{file_name}.pdf",
                            pagesize=A4,
                            topMargin=20,
                            leftMargin=20,
                            rightMargin=20,
                            bottomMargin=20)
    elements_copy = elements.copy()
    doc.build(elements_copy)
    existing_pdf = PdfReader(f"./temp/{file_name}.pdf", 'rb')
    custom_total_num = len(existing_pdf.pages) + len(disease_order) - 1
    # build report
    doc = SimpleDocTemplate(f"./temp/{file_name}.pdf",
                            pagesize=A4,
                            topMargin=20,
                            leftMargin=20,
                            rightMargin=20,
                            bottomMargin=20)
    # doc.build(elements)
    doc.build(
        elements,
        onFirstPage=lambda canvas, doc: add_page_number(canvas, doc, custom_total_num),
        onLaterPages=lambda canvas, doc: add_page_number(canvas, doc, custom_total_num)
    )

    return custom_total_num

def add_page_number(canvas, doc, total_num):
    page_num = canvas.getPageNumber() - 1
    text = f"Page {page_num} of {total_num}"
    if page_num > 0:
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor("#606060"))
        canvas.drawString(20, 8, text)
        canvas.restoreState()

def content_clean(content):
    # Replace <br> with <br/> and remove new lines
    content = content.replace('<br>', '<br/>').replace('\n', '<br/>')
    # Remove HTML tags but preserve the numbered list
    numbered_list_pattern = re.compile(r'(\d+\.\s)')
    parts = numbered_list_pattern.split(content)
    cleaned_parts = [re.sub(r'(?i)<(?!br\s*/?>)[^>]+>', '', part) for part in parts]
    content = ''.join(cleaned_parts)

    # Remove extra <br/> tags
    content = re.sub(r'(?i)(<br\s*/?>){2,}', '<br/>', content)
    # content = markdown.markdown(content)
    return content

def add_analysis(elements, text, styles):
    elements = translate_html(elements, styles, text)
    elements.append(Spacer(12, 12))
    return elements

def add_news(elements, content, analysis_MonthYear, location, styles):
    title = f"News information since {analysis_MonthYear} {location}"
    paragraphReportHeader = Paragraph(title, styles['Hed0'])
    elements.append(paragraphReportHeader)
    elements.append(Spacer(12, 12))
    elements = translate_html(elements, styles, content)
    elements.append(PageBreak())
    return elements

def add_toc(elements, diseases_order, styles):
    doc = SimpleDocTemplate(f"./temp/temp.pdf",
                            pagesize=A4,
                            topMargin=20,
                            leftMargin=20,
                            rightMargin=20,
                            bottomMargin=0)
    # add title
    title = variables.alert_title
    title = Paragraph(title, styles['Center'])
    title_table = Table([[title]], colWidths=[doc.width], rowHeights=[80])
    title_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.red),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ])
    elements.append(title_table)
    elements.append(Spacer(50, 12))
    text = variables.alert_content
    paragraphReportSummary = Paragraph(text, styles["Notice"])
    elements.append(paragraphReportSummary)

    # # add links
    # data = [[Paragraph('<a href="3">Report Summary</a>', styles['TOC']),
    #          Paragraph('<a href="5">News Information</a>', styles['TOC'])]]
    # link_list = list(range(7, 7 + len(diseases_order)))
    # for i, disease in enumerate(diseases_order):
    #     link = Paragraph(f'<a href="{link_list[i]}">{disease}</a>', styles['TOC'])
    #     if i % 2 == 0:
    #         data.append([link, ""])
    #     else:
    #         data[-1][1] = link
    # if len(diseases_order) % 2 != 0:
    #     data[-1].append(Paragraph("", styles['TOC']))
    # # add table
    # toc_table = Table(data, colWidths=[doc.width/2]*2)
    # toc_table.setStyle([
    #     ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    #     ('TOPPADDING', (0, 0), (-1, -1), 0),
    #     ('LEFTPADDING', (0, 0), (-1, -1), 0),
    #     ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    #     ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    # ])
    # elements.append(toc_table)
    elements.append(PageBreak())
        
    return elements

def add_table(elements, table_data, analysis_MonthYear, styles):
    doc = SimpleDocTemplate(f"./temp/temp.pdf",
                            pagesize=A4,
                            topMargin=20,
                            leftMargin=20,
                            rightMargin=20,
                            bottomMargin=0)
    # add title
    title = variables.cover_title_1 + " " + variables.cover_title_2
    title = f'{title}<br/><br/>{analysis_MonthYear}'
    title = Paragraph(title, styles['Center'])
    title_table = Table([[title]], colWidths=[doc.width], rowHeights=[80])
    title_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1E5E84")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ])
    elements.append(title_table)
    elements.append(Spacer(1, 12))

    # add table
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 1), colors.HexColor('#2DA699')),
        ('TEXTCOLOR', (0, 0), (-1, 1), colors.white),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 2), (-1, -1), 7),
        ('LEADING', (0, 2), (-1, -1), 7),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#2DA699')),
        ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 1), 10),
        ('VALIGN', (0, 0), (-1, 1), 'MIDDLE'),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
        ('LINEABOVE', (0, 2), (-1, 2), 1, colors.black),
        ('LINEABOVE', (1, 1), (-1, 1), 0.5, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        ('LINEBELOW', (0, -2), (-1, -1), 1, colors.black),
        ('SPAN', (0, 0), (0, 1)),
        ('SPAN', (1, 0), (3, 0)),
        ('SPAN', (4, 0), (6, 0))
    ]

    data = [
        ('Disease', 'Cases', '', '', 'Deaths', '', ''),
        ('', 'Reported', 'MoM*', 'YoY**', 'Reported', 'MoM*', 'YoY**')
    ] + table_data.values.tolist()

    # setting width of table
    table = Table(data)
    table.setStyle(table_style)
    # prebuild
    elements_temp = [table]
    doc.build(elements_temp)
    # remove temp file
    os.remove(f"./temp/temp.pdf")

    colWidths = sum(table._colWidths)
    col_widths = [width * doc.width/colWidths for width in table._colWidths]
    table = Table(data, colWidths=col_widths)
    table.setStyle(table_style)

    ## setting color fill
    change_data = table_data.copy()
    change_data = change_data[['Diseases', 'Cases', 'Deaths']]
    change_data['Cases'] = change_data['Cases'].str.replace(',', '').astype(float)
    change_data['Deaths'] = change_data['Deaths'].str.replace(',', '').astype(float)
    min_case = min(change_data['Cases'][:-1])
    max_case = max(change_data['Cases'][:-1])
    norm_case = mcolors.Normalize(vmin=min_case, vmax=max_case)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#FFFFFFFF", "#088096FF"])
    for i in range(2, len(data)-1):
        value = change_data['Cases'][i-2]
        cell_color = mcolors.to_hex(cmap(norm_case(value)))
        table.setStyle([('BACKGROUND', (1, i), (1, i), colors.HexColor(cell_color))])

    min_death = min(change_data['Deaths'][:-1])
    max_death = max(change_data['Deaths'][:-1])
    norm_death = mcolors.Normalize(vmin=min_death, vmax=max_death)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#FFFFFFFF", "#019875FF"])
    for i in range(2, len(data)-1):
        value = change_data['Deaths'][i-2]
        cell_color = mcolors.to_hex(cmap(norm_death(value)))
        table.setStyle([('BACKGROUND', (4, i), (4, i), colors.HexColor(cell_color))])
    elements.append(table)

    # add footnote of table
    footnote = f'*MoM: Month on Month change, **YoY: Year on Year change.'
    footnote = Paragraph(footnote, styles['foot'])
    elements.append(footnote)
    elements.append(PageBreak())

    return elements

def add_legend(elements, legend, styles):
    elements.append(Spacer(36, 12))
    legend = legend.replace('\ufeff', '')
    paragraphReportSummary = Paragraph("<b>Notation from Data Source:</b><br/>" + legend, styles["break"])
    elements.append(paragraphReportSummary)
    elements.append(PageBreak())
    return elements


# Example usage:
# disease_name = "Hand foot and mouth disease"
# report_date = "June 2023"
# introduction_box_content = "box information"
# highlights_box_content = "**Prevalence**: this is test information"
# incidence_box_content = "this is test information"
# death_box_content = "this is test information"
# links_app = "https://lkg1116.shinyapps.io/CNIDS/"
# links_web = "https://github.com/xmusphlkg/CNID"
# foot_left_content = "Page 1 of 1"

# add_disease(disease_name, report_date, 
#             introduction_box_content, highlights_box_content,
#             incidence_box_content, death_box_content,
#             foot_left_content, links_app, links_web)
