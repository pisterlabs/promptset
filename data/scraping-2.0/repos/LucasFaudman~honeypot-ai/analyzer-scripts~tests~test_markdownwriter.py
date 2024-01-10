from analyzerbase import *
from markdownwriters.markdownwriterbase import *

from markdownwriters.ipmarkdownwriter import IPAnalyzerMarkdownWriter
from markdownwriters.cowrieattackmarkdownwriter import CowrieAttackMarkdownWriter


from markdownwriters.visualizer import CounterGrapher
from osintanalyzers.ipanalyzer import IPAnalyzer
from loganalyzers.cowrieloganalyzer import CowrieLogAnalyzer 


from openaianalyzers.openaianalyzer import OpenAIAnalyzer, OPENAI_API_KEY




from main import AttackAnalyzer


class TestMarkdownWriterBasics(TestCase):

    def test_basic_md(self):
        mdw = MarkdownWriter('test.md')
        mdw.write_md(h1('h1'))
        mdw.write_md(h2('h2'))
        mdw.write_md(h3('h3'))
        mdw.write_md(h4('h4'))
        mdw.write_md("\n"+italic('italic')+"\n")
        mdw.write_md("\n"+bold('bold')+"\n")
        mdw.write_md("\n"+link('Google.com', 'https://www.google.com')+"\n")
        mdw.write_md("\n"+image('image', 'https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png')+"\n")
        mdw.write_md("\n"+code('code')+"\n")
        mdw.write_md("\n"+codeblock('code block'))
        mdw.write_md("\n"+blockquote('blockquote'))
        mdw.write_md(unordered_list(['item1', 'item2', 'item3']))
        mdw.write_md(ordered_list(['item1', 'item2', 'item3']))
        mdw.write_md(hline())
        mdw.write_md(table(['header1', 'header2', 'header3'], [['row1col1', 'row1col2', 'row1col3'], ['row2col1', 'row2col2', 'row2col3']]))



    def test_convert_md_to_mdtxt_for_canvas(self):
        filepath = '/Users/lucasfaudman/Documents/SANS/internship/BACS-4498/attack-observations/attack-1/observation1.md'
        github_url = 'https://github.com/LucasFaudman/BACS-4498/blob/main/attack-observations/attack-1/observation1.md'
        convert_md_to_mdtxt_for_canvas(filepath, github_url)




    def test_counter_grapher(self):
        counter = Counter(['A', 'B', 'A', 'C', 'B', 'A', 'D', 'C', 'A', 'B', 'E'])
        grapher = CounterGrapher("/Users/lucasfaudman/Documents/SANS/internship/tests/observations", counter, n=10)
        grapher.plot()
        print(grapher.labels)



class TestMarkdownWriter(TestCase):


    @classmethod
    def setUpClass(cls):
        # Only run this once
        if hasattr(cls, 'cla'):
            return None


        cls.test_keys = [
        '713ca6a961a02c78b95decc18a01c69606d112c77ffc9f8629eb03ac39e7a22b',
        '346442d765fd49fd142ef69309ac870ac9076ece44dcb07a2769e9b2942de8e3',
        '8a57f997513e762dec5cd58a2de822cdf3d2c7ef6372da6c5be01311e96e8358',
        #'ea40ecec0b30982fbb1662e67f97f0e9d6f43d2d587f2f588525fae683abea73'
       #'440e8a6e0ddc0081c39663b5fcc342a6aa45185eb53c826d5cf6cddd9b87ea64',
       #'0229d56a715f09337b329f1f6ced86e68b6d0125747faafdbdb3af2f211f56ac'
       #"fe9291a4727da7f6f40763c058b88a5b0031ee5e1f6c8d71cc4b55387594c054",
        #'7bb46aa291cc9ca205b3b181532609eb24c24f05f31923f8d165a322a864b48f',
        #"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        #"440e8a6e0ddc0081c39663b5fcc342a6aa45185eb53c826d5cf6cddd9b87ea64",
        #"0229d56a715f09337b329f1f6ced86e68b6d0125747faafdbdb3af2f211f56ac",
        #"04a9aabb18e701dbe12c2606538202dc02156f480f3d58d926d20bd9bc613451",
        #"275776445b4225c06861b2f6f4e2ccf98e3f919583bddb9965d8cf3d4f6aa18f",
        #"c41b0875c506cc9421ae26ee43bd9821ccd505e9e24a732c8a9c0180eb34a5a8",
        
        ]

        cls.analyzer = AttackAnalyzer()
        cls.analyzer.load_attacks_from_attack_dir(only_attacks=cls.test_keys)
        cls.analyzer.postprocess_attacks()

    
    def test_cowrie_md(self):

        for key in self.test_keys:
            attack = self.analyzer.attacks[key]
            key = key.replace('/', '_')
            mdw = CowrieAttackMarkdownWriter(f'/Users/lucasfaudman/Documents/SANS/internship/tests/attacks/' + key + '.md', 
                                             mode="w+", 
                                             data_object=attack)
            mdw.update_md()


    
    def test_ipanalyzer_md(self):
        for key in self.test_keys:
            attack = self.analyzer.attacks[key]
                
            ips = attack.uniq_ips#.union({'80.94.92.20'})
            ips.difference_update(("127.0.0.1", "8.8.8.8"))

            #ipdata = self.analyzer.ipanalyzer.get_data(ips)
            

            key = key.replace('/', '_')
            fpath = f'/Users/lucasfaudman/Documents/SANS/internship/tests/attacks/' + key +'.md'
            ipmdw = IPAnalyzerMarkdownWriter(filepath=fpath,
                                             mode="a+", 
                                             data_object=attack)
            ipmdw.update_md()
            
        
        print("done")





    